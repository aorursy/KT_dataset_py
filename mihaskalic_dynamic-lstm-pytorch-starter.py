import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.utils.data_utils import GeneratorEnqueuer  # We only want this for multithreaded 

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch import Tensor
train_df = pd.read_csv("../input/train.csv")

keep = [len(x[:-1].split()) > 0 for x in train_df["question_text"]]
train_df = train_df[keep]
train_df, val_df = train_test_split(train_df, test_size=0.1)
# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:120]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    seq_len = len(embeds)
    embeds+= [empyt_emb] * (120 - len(embeds))
    return np.array(embeds), seq_len

embeddings = [text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:5000])]
val_x, val_xlen = zip(*embeddings)

# For dynamic LSTM the values have to be ordered.
sorder = np.argsort(val_xlen)[::-1]

val_xlen = np.array(val_xlen)[sorder]
val_x = np.array(val_x)[sorder]
val_y = np.array(val_df["target"][:5000])[sorder]
# Train data providers
batch_size = 128
def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr, text_len = zip(*[text_to_array(text) for text in texts])
            sorder = np.argsort(text_len)[::-1]
            text_arr = np.array(text_arr)[sorder]
            text_len = np.array(text_len)[sorder]
            yield text_arr, text_len, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])[sorder]
class BiLSTM_model(nn.Module):
    def __init__(self, embedding_size=300, hidden_size=128):
        super(BiLSTM_model, self).__init__()
        self.lstm = nn.LSTM(300, hidden_size, 
                            batch_first=True, num_layers=2,
                            bidirectional=True)
        
        self.fc = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, input_lengths):
        x = pack_padded_sequence(x, input_lengths, batch_first=True)
        out = self.lstm(x)
        out = out[1][1][2:] # TODO check if :2 really is FW and BW of last layer!
        out = out.transpose(0,1)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        return self.sigmoid(out).flatten()
        # return memory

model = BiLSTM_model()
model.cuda()
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
bcloss = nn.BCELoss()

my_generator = GeneratorEnqueuer(batch_gen(train_df))
my_generator.start()
mg =  my_generator.get()


def val_gen(batch_size=256):
    n_batches = math.ceil(len(val_x) / batch_size)
    for idx in range(n_batches):
        xb = val_x[idx *batch_size:(idx+1) * batch_size]
        xlb = val_xlen[idx *batch_size:(idx+1) * batch_size]
        yb = val_y[idx *batch_size:(idx+1) * batch_size]
        yield xb, xlb, yb
losses = []
for i, (x, xlen, y) in tqdm(enumerate(mg)):
    optimizer.zero_grad()
    
    x = Tensor(x).cuda()
    y_pred = model(x, xlen)
    loss = bcloss(y_pred, Tensor(y).cuda())
    loss.backward()
    
    losses.append(loss.data.cpu().numpy())
    optimizer.step()
    
    if (i + 1) % 1000 == 0:
        print("Iter: {}".format(i+1))
        print("\tAverage training loss: {:.5f}".format(np.mean(losses)))
        losses = []
        # Evalute F1 on validation set
        model.eval()
        all_preds = []
        for x, xlen, y in val_gen():
            y_pred = model(Variable(Tensor(x)).cuda(), xlen)
            all_preds.extend(y_pred.cpu().data.numpy())
        score = f1_score(val_y, np.array(all_preds).flatten() > 0.5)
        print("\tVal F1 score: {:.5f}".format(score))
        model.train()
    if  i == 10000:
        print("Reducing LR")
        for g in optimizer.param_groups:
            g['lr'] = 0.0001
    if (i + 1) % 12000 == 0:  # We are done
        break

batch_size = 256

def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        ids = test_df["qid"][i*batch_size:(i+1)*batch_size]
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr, text_len = zip(*[text_to_array(text) for text in texts])
        sorder = np.argsort(text_len)[::-1]
        
        text_arr = np.array(text_arr)[sorder]
        text_len = np.array(text_len)[sorder]
        ids = np.array(ids)[sorder]
        yield text_arr, text_len, ids
    
test_df = pd.read_csv("../input/test.csv")

all_preds = []
all_ids = []

model.eval()
for x, xlen, xid in tqdm(batch_gen(test_df)):
    preds = model(Variable(Tensor(x).cuda()), xlen)
    preds = preds.data.cpu().numpy()
    all_preds.extend(preds)
    all_ids.extend(xid)
y_te = (np.array(all_preds).flatten() > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": all_ids, "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)