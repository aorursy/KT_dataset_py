# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import os

import string

import re

import numpy as np

from collections import Counter, defaultdict

from pathlib import Path

from tqdm import tqdm_notebook as tqdm



from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



from nltk.corpus import stopwords

import nltk
from gensim.models import KeyedVectors

from nltk.corpus import stopwords

import nltk

# nltk.download('stopwords')
train_df = pd.read_csv("../input/quora-question-pairs/train.csv.zip")
PATH = Path('../input')

train_df = pd.read_csv(PATH/"quora-question-pairs/train.csv.zip")

test_df = pd.read_csv(PATH/"quora-question-pairs/test.csv")

word2vec = KeyedVectors.load_word2vec_format("../input/googleword2vec/GoogleNews-vectors-negative300.bin", binary=True)
stops = set(stopwords.words('english'))



def text_to_word_list(text):

    ''' Pre process and convert texts to a list of words '''

    text = str(text)

    text = text.lower()



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)



    text = text.split()



    return text



# Prepare embedding

vocab2index = {"<PAD>":0, "UNK":1}

words = ["<PAD>", "UNK"]  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
questions_cols = ['question1', 'question2']



# Iterate over the questions only of both training and test datasets

for dataset in [train_df, test_df]:

    for index, row in tqdm(dataset.iterrows()):



        # Iterate through the text of both questions of the row

        for question in questions_cols:



            q2n = []  # q2n -> question numbers representation

            for word in text_to_word_list(row[question]):

                # Check for unwanted words

                if word in stops and word not in word2vec.vocab:

                    continue



                if word not in vocab2index:

                    vocab2index[word] = len(words)

                    q2n.append(len(words))

                    words.append(word)

                else:

                    q2n.append(vocab2index[word])



            # Replace questions as word to question as number representation

            dataset.at[index, question] = q2n

          
embedding_dim = 300

embeddings_matrix = 1 * np.random.randn(len(vocab2index) + 1, embedding_dim)  # This will be the embedding matrix

embeddings_matrix[0] = 0  # So that the padding will be ignored



# Build the embedding matrix

for word, index in vocab2index.items():

    if word in word2vec.vocab:

        embeddings_matrix[index] = word2vec.word_vec(word)

del word2vec



V = len(embeddings_matrix)

print(V)
train_df.head()
test_df.head()
train, valid = train_test_split(train_df, test_size=0.2, random_state=42)
def encode_sentence(s, vocab2index=vocab2index, N=50, padding_start=True):

    '''helper function to add paddings to sentence'''

    enc = np.zeros(N, dtype=np.int32)

    enc1 = np.array(s)

    l = min(N, len(enc1))

    if not padding_start:

        enc[:l] = enc1[:l]

    else:

        enc[N-l:] = enc1[:l]

    return enc, l
class QuoraDataset(Dataset):

    def __init__(self, df, is_train=True):

        self.is_train = is_train

        self.X1 = [encode_sentence(train) for train in df.question1]

        self.X2 = [encode_sentence(train) for train in df.question2]

        if self.is_train:

            self.y = df.is_duplicate.values

    

    def __len__(self):

        return len(self.X1)

    

    def __getitem__(self, idx):

        if self.is_train:

            x1 = self.X1[idx]

            x2 = self.X2[idx]

            return x1, x2, self.y[idx]

        else:

            x1 = self.X1[idx]

            x2 = self.X2[idx]

            return x1, x2



train_ds = QuoraDataset(train, is_train=True)

valid_ds = QuoraDataset(valid, is_train=True)

test_ds = QuoraDataset(test_df, is_train=False)



batch_size = 3000

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
def exponent_neg_manhattan_distance(x1, x2):

    ''' Helper function for the similarity estimate of the LSTMs outputs '''

    return torch.exp((-torch.sum(torch.abs(x1 - x2), dim=1)))

    # the distance function here gives data in range [0, 1]

    # use binary_cross_entropy as loss fucntion
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pre_weights):

        super(LSTMModel,self).__init__()

        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pre_weights is not None:

            self.embeddings.weight.data.copy_(torch.from_numpy(pre_weights))

            self.embeddings.weight.requires_grad = False ## freeze embeddings

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.linear = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)



    def forward(self, x1, x2):

        x1 = self.embeddings(x1)

        x2 = self.embeddings(x2)

        x1 = self.dropout(x1)

        x2 = self.dropout(x2)

        x1_lstm, (h1, ct) = self.lstm(x1)

        x2_lstm, (h2, ct) = self.lstm(x2)

        return exponent_neg_manhattan_distance(h1[-1], h2[-1])
def train_epocs(model, optimizer, train_dl, valid_dl, epochs=10):

    for i in tqdm(range(epochs)):

        model.train()

        sum_loss = 0.0

        total = 0

        for x1, x2, y in train_dl:

            x1 = x1[0].long().cuda()

            x2 = x2[0].long().cuda()

            y_pred = model(x1, x2).cpu()

            optimizer.zero_grad()

            loss = F.binary_cross_entropy(y_pred, y.float())

            loss.backward()

            optimizer.step()

            sum_loss += loss.item()*y.shape[0]

            total += y.shape[0]

        val_loss, val_acc = val_metrics(model, valid_dl)

        if i % 5 == 1:

            print("train loss %.3f val loss %.3f and val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))





def val_metrics(model, valid_dl):

    model.eval()

    correct = 0

    total = 0

    sum_loss = 0.0

    for x1, x2, y in valid_dl:

        x1 = x1[0].long().cuda()

        x2 = x2[0].long().cuda()

        y_hat = model(x1, x2).cpu()

        loss = F.binary_cross_entropy(y_hat, y.float())

        y_pred = y_hat > 0.5

        correct += (y_pred.float() == y).float().sum()

        total += y.shape[0]

        sum_loss += loss.item()*y.shape[0]

    return sum_loss/total, correct/total





def update_optimizer(optimizer, lr):

    for i, param_group in enumerate(optimizer.param_groups):

        param_group["lr"] = lr
model = LSTMModel(V, 300, 50, pre_weights=embeddings_matrix).cuda()

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.01)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=30)
prediction = []

for i, (x1, x2) in tqdm(enumerate(test_dl)):

    x1 = x1[0].long().cuda()

    x2 = x2[0].long().cuda()

    pred = model(x1, x2).cpu().detach().numpy()

    prediction.extend(pred)



submission = pd.read_csv("../input/quora-question-pairs/sample_submission.csv.zip")

submission['is_duplicate'] = np.array(prediction)

submission.to_csv('submission_lstm.csv', index=False)