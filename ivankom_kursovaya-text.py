import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

!ls '/kaggle/input'
import nltk

from collections import Counter

import itertools

import torch

from sklearn import model_selection

imdb_df = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv', encoding='latin-1')

dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]

test_df = imdb_df[(imdb_df.type == 'test')]

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)

class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, label_id):

        self.input_ids = input_ids

        self.label_id = label_id

class Vocab:

    def __init__(self, itos, unk_index):

        self._itos = itos

        self._stoi = {word:i for i, word in enumerate(itos)}

        self._unk_index = unk_index

        

    def __len__(self):

        return len(self._itos)

    

    def word2id(self, word):

        idx = self._stoi.get(word)

        if idx is not None:

            return idx

        return self._unk_index

    

    def id2word(self, idx):

        return self._itos[idx]

from tqdm import tqdm_notebook
class TextToIdsTransformer:

    def transform():

        raise NotImplementedError()

        

    def fit_transform():

        raise NotImplementedError()

class SimpleTextTransformer(TextToIdsTransformer):

    def __init__(self, max_vocab_size):

        self.special_words = ['<PAD>', '</UNK>', '<S>', '</S>']

        self.unk_index = 1

        self.pad_index = 0

        self.vocab = None

        self.max_vocab_size = max_vocab_size

        

    def tokenize(self, text):

        return nltk.tokenize.word_tokenize(text.lower())

        

    def build_vocab(self, tokens):

        itos = []

        itos.extend(self.special_words)

        

        token_counts = Counter(tokens)

        for word, _ in token_counts.most_common(self.max_vocab_size - len(self.special_words)):

            itos.append(word)

            

        self.vocab = Vocab(itos, self.unk_index)

    

    def transform(self, texts):

        result = []

        for text in texts:

            tokens = ['<S>'] + self.tokenize(text) + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

    

    def fit_transform(self, texts):

        result = []

        tokenized_texts = [self.tokenize(text) for text in texts]

        self.build_vocab(itertools.chain(*tokenized_texts))

        for tokens in tokenized_texts:

            tokens = ['<S>'] + tokens + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result
def build_features(token_ids, label, max_seq_len, pad_index, label_encoding):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [pad_index for _ in range(max_seq_len - len(token_ids))]

    return InputFeatures(ids, label_encoding[label])

def features_to_tensor(list_of_features):

    text_tensor = torch.tensor([example.input_ids for example in list_of_features], dtype=torch.long)

    labels_tensor = torch.tensor([example.label_id for example in list_of_features], dtype=torch.long)

    return text_tensor, labels_tensor
max_seq_len=200

classes = {'neg': 0, 'pos' : 1}

text2id = SimpleTextTransformer(10000)



train_ids = text2id.fit_transform(train_df['review'])

val_ids = text2id.transform(val_df['review'])

test_ids = text2id.transform(test_df['review'])
train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(train_ids, train_df['label'])]



val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(val_ids, val_df['label'])]



test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(test_ids, test_df['label'])]
from torch.utils.data import TensorDataset,DataLoader

from torch.utils import data

batch_size = 64





train_tensor, train_labels = features_to_tensor(train_features)

val_tensor,     val_labels = features_to_tensor(val_features)

test_tensor,   test_labels = features_to_tensor(test_features)
train_dataset = TensorDataset(train_tensor, train_labels)

val_dataset   = TensorDataset(val_tensor, val_labels)

test_dataset  = TensorDataset(test_tensor, test_labels)



train_loader = DataLoader(train_dataset, batch_size = batch_size)

val_loader   = DataLoader(val_dataset, batch_size = batch_size)

test_loader  = DataLoader(test_dataset, batch_size = batch_size)
import torch.nn as nn

import torch.nn.functional as F



class network(nn.Module):

    def __init__(self):

        super(network, self).__init__()

        self.preproc = nn.Sequential(

            nn.Embedding(10000,100)

        )

        self.hidden = nn.Sequential(

            nn.Conv1d(in_channels=100, out_channels=60, kernel_size=3), 

            nn.ReLU(),     

            nn.Conv1d(in_channels=60, out_channels=100, kernel_size=3), 

            nn.ReLU(), 

            nn.MaxPool1d(10))

        

        self.output = nn.Sequential(

            nn.Linear(1900,1),

            nn.Sigmoid()

        )

    def forward(self, x):

        batch = x.size(0)

        x = self.preproc(x)

        x = x.transpose(2,1)

        

        y = self.hidden(x).view(batch, -1)

        return  self.output(y)
from sklearn.metrics import accuracy_score

def fit(net,crit,train_loader,val_loader,optimizer, epochs):

    best=0

    #net.cuda()

    for i in range(epochs):

        tr_loss = 0

        val_loss = 0

        val_accuracy =0

        for xx,yy in train_loader:

            #xx, yy = xx.cuda(), yy.cuda()

            optimizer.zero_grad()

            y = net.forward(xx)

            loss = crit(y,yy.float().view(len(yy),-1))

            tr_loss += loss

            loss.backward()

            optimizer.step()

        tr_loss /= len(train_loader)

        with torch.no_grad():

            for xx,yy in val_loader:

                all_preds = []

                #xx, yy = xx.cuda(), yy.cuda()

                y = net.forward(xx)

                loss = crit(y,yy.float().view(len(yy),-1))

                val_loss += loss

                for index in y:

                    if index>0.5:

                        all_preds.append(1)

                    else:

                        all_preds.append(0)

                yy = yy.cpu().numpy()

                val_accuracy += accuracy_score(all_preds,yy)

            val_accuracy /= len(val_loader)

            if val_accuracy>best:

                best = val_accuracy

                torch.save(net.state_dict(), "../model.py")

        print(("epoch:%d, train loss:%f, validation accuracy:%f" % (i,tr_loss.item(),val_accuracy.item())))

    net.cpu()

    print("Train ended. Best accuracy is %f" % float(best))
model = network()

from torch.optim import Adam

criterion = nn.BCELoss()

optimizer = Adam(model.parameters(), lr=0.005)

fit(model,criterion,train_loader,val_loader,optimizer,10)
from sklearn import metrics

all_preds = []

correct_preds = []

with torch.no_grad():

    model.eval()

    for xx, yy in test_loader:

        #model.cuda()

        #xx = xx.cuda()

        output = model.forward(xx)

        for i in output:

            if i>0.5:

                all_preds.append(1)

            else:

                all_preds.append(0)

        correct_preds.extend(yy.tolist())



print(metrics.classification_report(correct_preds, all_preds))
from keras.preprocessing.text import Tokenizer

df = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")

df = df.drop(['Unnamed: 0','file'],axis=1)

df.columns = ['type',"review","sentiment"]

df.head()

print(df.head())

df = df[df.sentiment != 'unsup']

df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})



df_train = df[df.type == 'train']

df_test = df[df.type == 'test']

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {

        'ngram_range' : (1, 2),

        'dtype' : 'int32',

        'strip_accents' : 'unicode',

        'decode_error' : 'replace',

        'analyzer' : 'word',

        'min_df' : 2,

    }

    

    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    x_train = tfidf_vectorizer.fit_transform(train_texts)

    x_val = tfidf_vectorizer.transform(val_texts)

    

    selector = SelectKBest(f_classif, k=min(6000, x_train.shape[1]))

    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val



df_bag_train, df_bag_test = ngram_vectorize(df_test['review'], df_test['sentiment'], df_train['review'])

df_bag_train, df_bag_test = ngram_vectorize(df_test['review'], df_test['sentiment'], df_train['review'])

from sklearn import metrics



nb = MultinomialNB()

nb.fit(df_bag_train, df_train['sentiment'])

nb_pred = nb.predict(df_bag_test)

print(metrics.classification_report(df_test['sentiment'], nb_pred))

cm = metrics.confusion_matrix(nb_pred, df_test['sentiment'])

print('Accuracy ',metrics.accuracy_score(df_test['sentiment'], nb_pred))