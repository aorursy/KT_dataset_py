import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from nltk.tokenize import TweetTokenizer
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
import time
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.utils.data
import random
import warnings
warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples.")
np.seterr(divide='ignore')
import re
import os
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
path = '../input/'
train = pd.read_csv(os.path.join(path,"train.csv"))
test = pd.read_csv(os.path.join(path,"test.csv"))
sub = pd.read_csv(os.path.join(path,'sample_submission.csv'))
print('Available embeddings:',  os.listdir(os.path.join(path,"embeddings/")))
train["target"].value_counts()
train.head()
print('Average word length of questions in train is {0:.0f}.'.format(np.mean(train['question_text'].apply(lambda x: len(x.split())))))
print('Average word length of questions in test is {0:.0f}.'.format(np.mean(test['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of questions in train is {0:.0f}.'.format(np.max(train['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of questions in test is {0:.0f}.'.format(np.max(test['question_text'].apply(lambda x: len(x.split())))))
print('Average character length of questions in train is {0:.0f}.'.format(np.mean(train['question_text'].apply(lambda x: len(x)))))
print('Average character length of questions in test is {0:.0f}.'.format(np.mean(test['question_text'].apply(lambda x: len(x)))))
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
                "can't" : "cannot",
                "couldn't" : "could not",
                "didn't" : "did not",
                "doesn't" : "does not",
                "don't" : "do not",
                "hadn't" : "had not",
                "hasn't" : "has not",
                "haven't" : "have not",
                "he'd" : "he would",
                "he'll" : "he will",
                "he's" : "he is",
                "i'd" : "I would",
                "i'd" : "I had",
                "i'll" : "I will",
                "i'm" : "I am",
                "isn't" : "is not",
                "it's" : "it is",
                "it'll":"it will",
                "i've" : "I have",
                "let's" : "let us",
                "mightn't" : "might not",
                "mustn't" : "must not",
                "shan't" : "shall not",
                "she'd" : "she would",
                "she'll" : "she will",
                "she's" : "she is",
                "shouldn't" : "should not",
                "that's" : "that is",
                "there's" : "there is",
                "they'd" : "they would",
                "they'll" : "they will",
                "they're" : "they are",
                "they've" : "they have",
                "we'd" : "we would",
                "we're" : "we are",
                "weren't" : "were not",
                "we've" : "we have",
                "what'll" : "what will",
                "what're" : "what are",
                "what's" : "what is",
                "what've" : "what have",
                "where's" : "where is",
                "who'd" : "who would",
                "who'll" : "who will",
                "who're" : "who are",
                "who's" : "who is",
                "who've" : "who have",
                "won't" : "will not",
                "wouldn't" : "would not",
                "you'd" : "you would",
                "you'll" : "you will",
                "you're" : "you are",
                "you've" : "you have",
                "'re": " are",
                "wasn't": "was not",
                "we'll":" will",
                "didn't": "did not",
                "tryin'":"trying",
               '\u200b': '',
                '…': '',
                '\ufeff': '',
                'करना': '',
                'है': ''}

for coin in ['Litecoin', 'altcoin', 'altcoins', 'coinbase', 'litecoin', 'Unocoin', 'Dogecoin', 'cryptocoin', 'Altcoins', 'filecoin', 'Altcoin', 'cryptocoins',
             'Altacoin', 'Dentacoin', 'Bytecoin', 'Siacoin', 'Onecoin', 'dogecoin', 'unocoin', 'siacoin', 'litecoins', 'Filecoin', 'Buyucoin', 'Litecoins',
             'Laxmicoin', 'shtcoins', 'Sweatcoin', 'Skycoin', 'vitrocoin', 'Monacoin', 'Litcoin', 'reddcoin', 'freebitcoin', 'Namecoin', 'plexcoin', 'Onecoins',
             'daikicoin', 'Gainbitcoin', 'Gatecoin', 'Plexcoin', 'peercoin', 'coinsecure', 'dogecoins', 'cointries', 'Zcoin', 'genxcoin', 'Frazcoin', 'frazcoin',
             'coinify', 'Nagricoin', 'OKcoin', 'Presscoins', 'Dagcoin', 'batcoin', 'Spectrocoin', 'Travelflexcoin', 'ecoin', 'Minexcoin', 'Kashhcoin', 'coinone',
             'octacoin', 'coinsides', 'zabercoin', 'ADZcoin', 'cyptocoin', 'bitecoin', 'Bitecoin', 'Emercoin', 'tegcoin', 'flipcoin', 'Gridcoin', 'Facecoin',
             'Ravencoins', 'digicoin', 'bitcoincash', 'Vitrocoin', 'Livecoin', 'dashcoin', 'Fedcoin', 'litcoins', 'Webcoin', 'coinspot', 'bitoxycoin', 'peercoins',
             'Ucoin', 'ALTcoins', 'coincidece', 'dagcoin', 'Giracoin', 'coincheck', 'Swisscoin', 'butcoin', 'neocoin', 'mintcoin', 'Myriadcoin', 'Viacoin', 'jiocoin',
             'Potcoin', 'bibitcoin', 'gainbitcoin', 'altercoins', 'coinburn', 'Kodakcoin', 'Bcoin', 'Kucoin', 'Operacoin', 'Lomocoin', 'dentacoin', 'Nyancoin',
             'Jiocoin', 'Indicoin', 'coinsidered', 'Vertcoin', 'Maidsafecoin', 'coindelta', 'coinfirm', 'coinvest', 'bixcoin', 'litcoin', 'Dogecoins', 'Unicoin',
             'Rothscoin', 'localbitcoins', 'groestlcoin', 'sibcoin', 'Travelercoin', 'Vericoin', 'bytecoin', 'Bananacoin', 'PACcoin']:
    mispell_dict[coin] = 'bitcoin'

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

# Clean the text
train["question_text"] = train["question_text"].apply(lambda x: clean_text(x.lower()))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x.lower()))

# Clean numbers
train["question_text"] = train["question_text"].apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))

# Clean speelings
train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))
max_features = 120000
tk = Tokenizer(lower = True, filters='', num_words=max_features)
full_text = list(train['question_text'].values) + list(test['question_text'].values)
tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train['question_text'].fillna('missing'))
test_tokenized = tk.texts_to_sequences(test['question_text'].fillna('missing'))
train['question_text'].apply(lambda x: len(x.split())).plot(kind='hist');
plt.yscale('log');
plt.title('Distribution of question text length in characters');
max_len = 72
X_train = pad_sequences(train_tokenized, maxlen=max_len)
X_test = pad_sequences(test_tokenized, maxlen=max_len)
y_train = train['target'].values
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(X_train, y_train))
embed_size = 300
embedding_path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
# all_embs = np.stack(embedding_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std = -0.005838499, 0.48782197
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_path = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore') if len(o)>100)
# all_embs = np.stack(embedding_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std = -0.0053247833, 0.49346462
embedding_matrix1 = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix1[i] = embedding_vector
embedding_path = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore') if len(o)>100)
# all_embs = np.stack(embedding_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
# print(emb_mean,emb_std)
emb_mean,emb_std = -0.0033469985, 0.109855495
embedding_matrix2 = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix2[i] = embedding_vector
embedding_matrix = np.mean([embedding_matrix, embedding_matrix1, embedding_matrix2], axis=0)
print(embedding_matrix.shape)
del embedding_matrix1, embedding_matrix2
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
    
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        hidden_size = 128
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(hidden_size*2, max_len)
        self.gru_attention = Attention(hidden_size*2, max_len)
        
        self.linear = nn.Linear(1536, 256)
        self.linear1 = nn.Linear(256, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        
        avg_pool_g = torch.mean(h_gru, 1)
        max_pool_g, _ = torch.max(h_gru, 1)
        
        avg_pool_l = torch.mean(h_lstm, 1)
        max_pool_l, _ = torch.max(h_lstm, 1)
        
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool_g, max_pool_g, avg_pool_l, max_pool_l), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.relu(self.linear1(conc))
        out = self.out(conc)
        
        return out
m = NeuralNet()
x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
batch_size = 512
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
seed=1029
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)
def train_model_full(X_train=X_train, y_train=y_train, splits=splits, n_epochs=5, batch_size=batch_size, validate=False):
    train_preds = np.zeros(len(X_train))
    test_preds = np.zeros((len(test), len(splits)))
    scores = []
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {i + 1}. {time.ctime()}')
        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
        
        seed_everything(seed + i)
        model = NeuralNet()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        
        train_dataset = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid_dataset = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        best_f1 = 0
        best_model_name = ''
        
        for epoch in range(n_epochs):
            print()
            print(f'Epoch {epoch}. {time.ctime()}')
            model.train()
            avg_loss = 0.

            for x_batch, y_batch in train_loader:
                # print(x_batch.shape)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()

            valid_preds = np.zeros((x_val_fold.size(0)))

            if validate:
                avg_val_loss = 0.
                for j, (x_batch, y_batch) in enumerate(valid_loader):
                    y_pred = model(x_batch).detach()

                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds[j * batch_size:(j+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

                best_th, score = scoring(y_val_fold.cpu().numpy(), valid_preds, verbose=True)

#                 if score > best_f1:
#                     best_f1 = score
#                     torch.save(model.state_dict(), f'model_{epoch}.pt')
#                     best_model_name = f'model_{epoch}.pt'
#                 else:
#                     print('Stopping training on this fold')
#                     break
        
#         if score < best_f1:
#             checkpoint = torch.load(best_model_name)
#             model.load_state_dict(checkpoint)
#             model.eval()

        valid_preds = np.zeros((x_val_fold.size(0)))

        avg_val_loss = 0.
        for j, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds[j * batch_size:(j+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        best_th, score = scoring(y_val_fold.cpu().numpy(), valid_preds, verbose=True)

        scores.append(score)

        test_preds_fold = np.zeros((len(test_loader.dataset)))

        for j, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[j * batch_size:(j+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds
        test_preds[:, i] = test_preds_fold
    print(f'Finished training at {time.ctime()}')
    print(f'Mean validation f1-score: {np.mean(scores)}. Std: {np.std(scores)}')
    
    return train_preds, test_preds
def scoring(y_true, y_proba, verbose=True):
    # https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76391
    
    def threshold_search1(y_true, y_proba):
        precision , recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001) 
        F = 2 / (1/precision + 1/recall)
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        return best_th 

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    # rkf = StratifiedKFold(n_splits=5)

    scores = []
    ths = []
    for train_index, test_index in rkf.split(y_true, y_true):
        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        # determine best threshold on 'train' part 
        best_threshold = threshold_search1(y_true_train, y_prob_train)

        # use this threshold on 'test' part for score 
        sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))
        scores.append(sc)
        ths.append(best_threshold)

    best_th = np.mean(ths)
    score = np.mean(scores)

    if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score,5)}')

    return best_th, score
train_preds, test_preds = train_model_full(X_train=X_train, y_train=y_train, splits=splits, n_epochs=5, batch_size=batch_size, validate=True)
best_th, score = scoring(y_train, train_preds)
sub['prediction'] = test_preds.mean(1) > best_th
sub.to_csv("submission.csv", index=False)