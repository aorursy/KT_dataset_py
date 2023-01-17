import numpy as np

import pandas as pd

import os

from multiprocessing import Pool
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
def parallelize_apply(df,func,colname,num_process,newcolnames):

    # takes as input a df and a function for one of the columns in df

    pool =Pool(processes=num_process)

    arraydata = pool.map(func,tqdm(df[colname].values))

    pool.close()

    

    newdf = pd.DataFrame(arraydata,columns = newcolnames)

    df = pd.concat([df,newdf],axis=1)

    return df



def parallelize_dataframe(df, func):

    df_split = np.array_split(df, 4)

    pool = Pool(4)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



# some fetures 

# ['','','india/n','','quora','','sex','','','country/countries','china','','','chinese','','']

def add_features(df):

    df['question_text'] = df['question_text'].apply(lambda x:str(x))

    # df = parallelize_apply(df,sentiment,'question_text',4,['sentiment','subjectivity']) 

    # df['sentiment'] = df['question_text'].progress_apply(lambda x:sentiment(x))

    df['total_length'] = df['question_text'].apply(len)

    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                axis=1)

    df['num_words'] = df.question_text.str.count('\S+')

    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] 

    return df



train_df = parallelize_dataframe(train_df, add_features)

test_df = parallelize_dataframe(test_df, add_features)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



max_features = 100000



tokenizer = Tokenizer(num_words=max_features, lower=True)

full_text = list(train_df['question_text'].values)

tokenizer.fit_on_texts(full_text)



train_tokenized = tokenizer.texts_to_sequences(train_df['question_text'].fillna('missing'))

test_tokenized = tokenizer.texts_to_sequences(test_df['question_text'].fillna('missing'))
train_meta = np.array(train_df.iloc[:,3:])

test_meta = np.array(test_df.iloc[:,2:])

print(train_meta.shape)

print(test_meta.shape)
max_len = 72

train_X = np.column_stack((pad_sequences(train_tokenized, maxlen=max_len), train_meta))

test_X = np.column_stack((pad_sequences(test_tokenized, maxlen=max_len), test_meta))

train_y = train_df.target.values
from gensim.models import KeyedVectors

from gensim.test.utils import datapath, get_tmpfile

from gensim.scripts.glove2word2vec import glove2word2vec



news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings = KeyedVectors.load_word2vec_format(news_path, binary=True)
mean = np.mean(embeddings.vectors)

std = np.std(embeddings.vectors)

_, size = embeddings.vectors.shape

num_words = min(len(tokenizer.word_index), max_features)

embedding_matrix = np.random.normal(mean, std, (num_words, size))
for word, index in tokenizer.word_index.items():

    if index >= max_features: continue

    try:

        word_vector = embeddings.get_vector(word)

        embedding_matrix[index] = word_vector

    except:

        pass

print(embedding_matrix.shape)
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import random

from tqdm import  tqdm

tqdm.pandas()



def seed_torch(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed = 0

seed_torch(seed)



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



class BiLSTM(nn.Module):

    def __init__(self, embedding_matrix = embedding_matrix, embed_size = 300, hidden_size = 128):

        super().__init__()

        self.embedding = nn.Embedding(max_features, size)

        if embedding_matrix is not None:

            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

            self.embedding.weight.requires_grad = False

        

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional = True, batch_first = True, num_layers = 2, dropout = 0.1)

        self.fc = nn.Linear(hidden_size * 2 + 6, 1)

        

    def forward(self, x):

        embedded = self.embedding(x[:,:72])

        f = torch.tensor(x[:,72:], dtype=torch.float32).cuda()

        h_lstm, (h,c) = self.lstm(embedded)

        h_fc = torch.cat((h[-1],h[-2],f), dim = 1)

        

        return self.fc(h_fc)
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader, TensorDataset

import time

import re

import random

from sklearn.metrics import f1_score



splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(train_X, train_y))

train_epochs = 5

batch_size = 64



print(type(train_X), train_X.shape)

print(type(train_y), train_y.shape)

train_X = torch.Tensor(train_X)

train_y = torch.Tensor(train_y)

train_meta = torch.Tensor(train_meta)

test_meta = torch.Tensor(test_meta)





train_preds = np.zeros((len(train_X)))

test_preds = np.zeros((len(test_X)))



x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



for i, (train_idx, valid_idx) in enumerate(splits):

    print('Fold:{0}'.format(i))

    x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()

    y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()

    x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()

    y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    

    train_data = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

    valid_data = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    

    model = BiLSTM()

    model.cuda()

    loss_function = nn.BCEWithLogitsLoss()

    optimiser = optim.Adam(model.parameters())

    

    for epoch in range(train_epochs):

        start_time = time.time()

        model.train()

        avg_loss = 0.0

        # train  model

        for x_batch, y_batch in tqdm(train_loader, disable=True):

            y_predicted = model(x_batch)

            loss = loss_function(y_predicted, y_batch)

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

            avg_loss += loss.item()/len(train_loader)



        # evaluate model

        model.eval()

        val_predicted_fold = np.zeros(x_val_fold.size(0))

        test_predicted_fold = np.zeros(len(test_X))

        avg_val_loss =0.0



        for i, (x_batch, y_batch) in enumerate(valid_loader):

            y_predicted = model(x_batch).detach()

            avg_val_loss += loss_function(y_predicted, y_batch).item()/len(valid_loader)

            val_predicted_fold[i*batch_size:(i+1)*batch_size] = sigmoid(y_predicted.cpu().numpy())[:,0]   



        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1,

                                                        train_epochs, avg_loss, avg_val_loss, elapsed_time))

    # predict testing dataset

    test_predicted_fold = np.zeros(len(test_X))

    for i, (x_batch,) in enumerate(test_loader):

        y_pred = model(x_batch).detach()

        test_predicted_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    

    train_preds[valid_idx] = val_predicted_fold

    test_preds += test_predicted_fold / len(splits) # take average
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)]):

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result
search_result = threshold_search(train_y, train_preds)

search_result
sub = pd.read_csv('../input/sample_submission.csv')

sub.prediction = test_preds > search_result['threshold']

sub.to_csv("submission.csv", index=False)