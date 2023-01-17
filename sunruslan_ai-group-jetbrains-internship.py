import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import clear_output
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import spacy
from tqdm import tqdm, tqdm_notebook, tnrange
import seaborn as sns
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

SEED = 43
np.random.seed(SEED)
tqdm.pandas(desc='Progress')
columns = ["text", "parent_text", "score"]
df = pd.concat([
    pd.read_csv("../input/reddit-comment-score-prediction/comments_positive.csv", usecols=columns, na_filter=False),
    pd.read_csv("../input/reddit-comment-score-prediction/comments_negative.csv", usecols=columns, na_filter=False)
], ignore_index=True)
y = df['score']
df.drop(columns='score', inplace=True)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=SEED)

# To be sure we don't use indices to predict something
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
X_train['text'] = X_train.text.progress_apply(lambda x: x.strip())
X_train['parent_text'] = X_train.parent_text.progress_apply(lambda x: x.strip())
words = Counter()
for sent in tqdm(X_train.values):
    words.update(w.text.lower() for w in nlp(sent[0]))
    words.update(w.text.lower() for w in nlp(sent[1]))
   
words = sorted(words, key=words.get, reverse=True)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}
def indexer(s): 
    return [word2idx[w.text.lower()] for w in nlp(s)]
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=0.1, random_state=SEED)
class DataframeDataset(Dataset):
    
    def __init__(self, X, y, maxlen=10):
        self.maxlen = maxlen
        self.X = X
        self.y = y
        self.X['text_idx'] = self.X.text.progress_apply(indexer)
        self.X['text_lengths'] = self.X.text_idx.progress_apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        self.X['text_padded'] = self.X.text_idx.progress_apply(self.pad_data)
        self.X['parent_text_idx'] = self.X.parent_text.progress_apply(indexer)
        self.X['parent_text_lengths'] = self.X.parent_text_idx.progress_apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        self.X['parent_text_padded'] = self.X.parent_text_idx.progress_apply(self.pad_data)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        row = self.X.iloc[index, :]
        x1 = row.text_padded
        l1 = row.text_lengths
        x2 = row.parent_text_padded
        l2 = row.parent_text_lengths
        y = self.y[index]
        return x1, l1, x2, l2, y
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: 
            padded[:] = s[:self.maxlen]
        else: 
            padded[:len(s)] = s
        return padded
class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        self.criterion = nn.MSELoss()
        self.emb_dim = 128
        self.hidden_size = 64
        self.num_layers = 2
        self.emb = nn.Embedding(vocab_size, self.emb_dim)
        self.rnn_comment = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, 
                          dropout=0.5,batch_first=True)
        
        self.rnn_parent = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, 
                          dropout=0.5,batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(2*self.hidden_size, 128), 
            nn.LeakyReLU(0.01), 
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        
    def forward(x1, x1_lengths, x2, x2_lengths):
        bs = x1.size(1)
        embs = self.emb(seq)
        embs = pack_padded_sequence(embs, x1_lengths)
        self.h = self.init_hidden(bs) 
        _, h1 = self.rnn_comment(embs, self.h)
        
        bs = x2.size(1)
        embs = self.emb(seq)
        embs = pack_padded_sequence(embs, x2_lengths)
        self.h = self.init_hidden(bs) 
        _, h2 = self.rnn_parent(embs, self.h)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.model(x)
        return x
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1,batch_size,self.hidden_size)))
        
    def loss(x1, x1_lengths, x2, x2_lengths, y):
        prediction = self.forward(x1, x1_lengths, x2, x2_lengths)
        return self.criterion(prediction, y)
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = DataframeDataset(x_train.head(5), y_train[:5])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataset = DataframeDataset(x_val.head(5), y_val[:5])
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)
def train(model, loader, optimizer, device):
    model.train()
    losses = []
    for x1, x1_lengths, x2, x2_lengths, y in loader:
        optimizer.zero_grad()
        loss = model.loss(x1, x1_lengths, x2, x2_lengths, y)
        loss.backward()
        opimizer.step()
        losses.append(loss.item())
    return np.mean(losses)
    
def test(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x1, x1_lengths, x2, x2_lengths, y in loader:
            prediction = model.forward(x1, x1_lengths, x2, x2_lengths)
            losses.append(mean_squared_error(prediction.cpu().data.numpy(), y))
    return np.mean(losses)
epochs = 1000
vocab_size = len(words)
model = Net(vocab_size)
optimizer = torch.optim.Adam(model.parameters())
best_loss = 99999.0
endure = 0
train_history = []
val_history = []

for e in range(epochs):
    train_loss = train(model, train_loader, optimizer, device)
    val_loss = test(model, val_loader, device)
    
    train_history.append(train_loss)
    val_loss.append(val_loss)
    
    clear_output(True)
    plt.plot(train_history, label="train")
    plt.plot(val_history, label="val")
    plt.legend()
    plt.show()
    
    if best_loss > val_loss:
        endure = 0
        best_loss = val_loss
    else:
        endure += 1
    if endure == 5:
        break
    
model.eval()
y_pred = []
test_dataset = DatframeDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
for x1, x2, _ in test_loader:
    x1, x2 = x1.to(device), x2.to(device)
    prediction = model.forward(x1, x2)
    y_pred.extend(list(prediction.cpu().data))
mean_squared_error(y_test, y_pred)