import warnings

warnings.filterwarnings('ignore')

import logging

import os

import re

import collections

import copy

import datetime

import random

import traceback



import matplotlib.pyplot as plt

%matplotlib inline

import scipy

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import numpy as np



import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer



import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import StepLR

from torch import nn

from torch.nn import functional as F



nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

stop_words_en = set(stopwords.words('english'))

stemmer_en = SnowballStemmer('english')
dir_data = '../input/nlp-getting-started/'

file_test = 'test.csv'

file_name = 'train.csv'

submission_name = 'submission.csv'

valid_size = .3

random_seed = 123



random.seed(random_seed)

np.random.seed(random_seed)

torch.manual_seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
df = pd.read_csv(os.path.join(dir_data, file_name))

df_test = pd.read_csv(os.path.join(dir_data, file_test))

df.shape, df_test.shape
df.target.value_counts()
x_train, x_valid, y_train, y_valid = train_test_split(

    df, 

    df['target'],

    test_size=valid_size,

    random_state=random_seed

)

x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
config = {

    'TextPreprocessor': {

        'del_orig_col': False,

        'mode_stemming': True,

        'mode_norm': True,

        'mode_remove_stops': True,

        'mode_drop_long_words': True,

        'mode_drop_short_words': True,

        'min_len_word': 3,

        'max_len_word': 15,

        'max_size_vocab': 2000,

        'max_doc_freq': 0.5,

        'min_count': 5,

        'pad_word': None,



        

    },

    'VectorizeTexts': {

        'mode_bin': False,

        'mode_idf': False,

        'mode_idf': False,

        'mode_tfidf': True,

        'mode_scale': True,

    },

    'TrainModel': {

        'lr': .01,

        'step_size_scheduler': 10,

        'gamma_scheduler': 0.9,

        'early_stopping_patience': 40,

        'batch_size': 64,

        'epoch_n': 500,

        'criterion': F.cross_entropy

    },

    'Predict': {

        'batch_size': 1,

    }

}
class TextPreprocessor(object):

    def __init__(self, config):

        """Preparing text features."""



        self._del_orig_col = config.get('del_orig_col', True)

        self._mode_stemming = config.get('mode_stemming', True)

        self._mode_norm = config.get('mode_norm', True)

        self._mode_remove_stops = config.get('mode_remove_stops', True)

        self._mode_drop_long_words = config.get('mode_drop_long_words', True)

        self._mode_drop_short_words = config.get('mode_drop_short_words', True)

        self._min_len_word = config.get('min_len_word', 3)

        self._max_len_word = config.get('max_len_word', 17)

        self._max_size_vocab = config.get('max_size_vocab', 100000)

        self._max_doc_freq = config.get('max_doc_freq', 0.8) 

        self._min_count = config.get('min_count', 5)

        self._pad_word = config.get('pad_word', None)



    def _clean_text(self, input_text):

        """Delete special symbols."""



        input_text = input_text.str.lower()

        input_text = input_text.str.replace(r'[^a-z ]+', ' ')

        input_text = input_text.str.replace(r' +', ' ')

        input_text = input_text.str.replace(r'^ ', '')

        input_text = input_text.str.replace(r' $', '')



        return input_text





    def _text_normalization_en(self, input_text):

        '''Normalization of english text'''



        return ' '.join([lemmatizer.lemmatize(item) for item in input_text.split(' ')])





    def _remove_stops_en(self, input_text):

        '''Delete english stop-words'''



        return ' '.join([w for w in input_text.split() if not w in stop_words_en])





    def _stemming_en(self, input_text):

        '''Stemming of english text'''



        return ' '.join([stemmer_en.stem(item) for item in input_text.split(' ')])





    def _drop_long_words(self, input_text):

        """Delete long words"""

        return ' '.join([item for item in input_text.split(' ') if len(item) < self._max_len_word])





    def _drop_short_words(self, input_text):

        """Delete short words"""



        return ' '.join([item for item in input_text.split(' ') if len(item) > self._min_len_word])

    

    

    def _build_vocabulary(self, tokenized_texts):

        """Build vocabulary"""

        

        word_counts = collections.defaultdict(int)

        doc_n = 0



        for txt in tokenized_texts:

            doc_n += 1

            unique_text_tokens = set(txt)

            for token in unique_text_tokens:

                word_counts[token] += 1

                

        word_counts = {word: cnt for word, cnt in word_counts.items()

                       if cnt >= self._min_count and cnt / doc_n <= self._max_doc_freq}

        

        sorted_word_counts = sorted(word_counts.items(),

                                    reverse=True,

                                    key=lambda pair: pair[1])

        

        if self._pad_word is not None:

            sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

            

        if len(word_counts) > self._max_size_vocab:

            sorted_word_counts = sorted_word_counts[:self._max_size_vocab]

            

        word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

        word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')



        return word2id, word2freq



    def transform(self, df):        

        

        columns_names = df.select_dtypes(include='object').columns

        df[columns_names] = df[columns_names].astype('str')

        

        for i in df.index:

            df.loc[i, 'union_text'] = ' '.join(df.loc[i, columns_names])

            

        if self._del_orig_col:

            df = df.drop(columns_names, 1)

            

        df['union_text'] = self._clean_text(df['union_text'])

        

        if self._mode_norm:

            df['union_text'] = df['union_text'].apply(self._text_normalization_en, 1)

            

        if self._mode_remove_stops:

            df['union_text'] = df['union_text'].apply(self._remove_stops_en, 1)

            

        if self._mode_stemming:

            df['union_text'] = df['union_text'].apply(self._stemming_en)

            

        if self._mode_drop_long_words:

            df['union_text'] = df['union_text'].apply(self._drop_long_words, 1)

            

        if self._mode_drop_short_words:

            df['union_text'] = df['union_text'].apply(self._drop_short_words, 1)

            

        df.loc[(df.union_text == ''), ('union_text')] = 'EMPT'

        

        tokenized_texts = [[word for word in text.split(' ')] for text in df.union_text]

        word2id, word2freq = self._build_vocabulary(tokenized_texts)



        return tokenized_texts, word2id, word2freq
class VectorizeTexts(object):

    def __init__(self, config):

        """Preparing text features."""

        

        self._mode_bin = config.get('mode_bin', True)

        self._mode_idf = config.get('mode_idf', True)

        self._mode_tf = config.get('mode_tf', True)

        self._mode_tfidf = config.get('mode_tfidf', True)

        self._mode_scale = config.get('mode_scale', True)

        

    def _get_bin(self, result):

        """Get binary vectors"""

        

        result = (result > 0).astype('float32')

        

        return result

    

    def _get_tf(self, result):

        """Get term frequency."""

        

        result = result.tocsr()

        result = result.multiply(1 / result.sum(1))

        

        return result

    

    def _get_idf(self, result):

        """Get inverse document frequency."""

        

        result = (result > 0).astype('float32').multiply(1 / word2freq)

        

        return result

    

    def _get_tfidf(self, result):

        """Get term frequency and inverse document frequency."""

        

        result = result.tocsr()

        result = result.multiply(1 / result.sum(1)) 

        result = result.multiply(1 / word2freq) 

        

        return result 

    

    def _get_scale(self, result):

        """Standardize Tfidf dataset."""

        

        result = result.tocsc()

        result -= result.min()

        result /= (result.max() + 1e-6)

        

        return result

    

    def transform(self, tokenized_texts, word2id, word2freq):

        

        result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')

        for text_i, text in enumerate(tokenized_texts):

            for token in text:

                if token in word2id:

                    result[text_i, word2id[token]] += 1

        

        if self._mode_bin:

            result = self._get_bin(result)

        

        if self._mode_idf:

            result = self._get_idf(result)

            

        if self._mode_tf:

            result = self._get_tf(result)

            

        if self._mode_tfidf:

            result = self._get_tfidf(result)

            

        if self._mode_scale:

            result = self._get_scale(result)

            

        return result.tocsr()
class SparseFitDataset(Dataset):

    def __init__(self, features, targets):

        '''Dataset for train. Return features and labels.'''

        

        self.features = features

        self.targets = targets



    def __len__(self):

        return self.features.shape[0]



    def __getitem__(self, idx):

        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()

        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()

        return cur_features, cur_label

    

class SparsePredDataset(Dataset):

    def __init__(self, features):

        '''Dataset for predict. Return features only.'''

        

        self.features = features



    def __len__(self):

        return self.features.shape[0]



    def __getitem__(self, idx):

        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()

        return cur_features
class TrainModel(object):

    def __init__(self, config):

        """Train a model."""

        

        self._lr = config.get('lr', 1.0e-3)

        self._step_size_scheduler = config.get('step_size_scheduler', 10)

        self._gamma_scheduler = config.get('gamma_scheduler', 0.1)

        self._early_stopping_patience = config.get('early_stopping_patience', 10)

        self._batch_size = config.get('batch_size', 32)

        self._epoch_n = config.get('epoch_n', 100)

        self._criterion = config.get('criterion', F.cross_entropy)



    def train(self, model, train_dataset, val_dataset):



        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        lr_scheduler = StepLR(optimizer, step_size=self._step_size_scheduler, gamma=self._gamma_scheduler)

        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)

        val_dataloader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(device)



        best_val_loss = float('inf')

        best_epoch_i = 0

        best_model = copy.deepcopy(model)



        for epoch_i in range(self._epoch_n):

            try:

                epoch_start = datetime.datetime.now()

                print('Epoch {}/{}'.format(epoch_i, self._epoch_n))

                print('lr_value: {}'.format(lr_scheduler.get_lr()[0]), flush=True)

                lr_scheduler.step()

                model.train()

                mean_train_loss = 0

                mean_train_acc = 0

                train_batches_n = 0

                

                for batch_x, batch_y in train_dataloader:

                    

                    batch_x = batch_x.to(device)

                    batch_y = batch_y.to(device)

                    

                    optimizer.zero_grad()

                    

                    pred = model(batch_x)

                    

                    loss = self._criterion(pred, batch_y)

                    acc = accuracy_score(pred.detach().cpu().numpy().argmax(-1), 

                                         batch_y.cpu().numpy())

                    loss.backward()

                    optimizer.step()

                    

                    mean_train_loss += float(loss)

                    mean_train_acc += float(acc)

                    train_batches_n += 1



                mean_train_loss /= train_batches_n

                mean_train_acc /= train_batches_n

                print('{:0.2f} s'.format((datetime.datetime.now() - epoch_start).total_seconds()))

                print('Train Loss', mean_train_loss)

                print('Train Acc', mean_train_acc)



                model.eval()

                mean_val_loss = 0

                mean_val_acc = 0

                val_batches_n = 0



                with torch.no_grad():

                    for batch_x, batch_y in val_dataloader:



                        batch_x = batch_x.to(device)

                        batch_y = batch_y.to(device)



                        pred = model(batch_x)

                        loss = self._criterion(pred, batch_y)

                        acc = accuracy_score(pred.detach().cpu().numpy().argmax(-1), 

                                             batch_y.cpu().numpy())



                        mean_val_loss += float(loss)

                        mean_val_acc += float(acc)

                        val_batches_n += 1



                mean_val_loss /= val_batches_n

                mean_val_acc /= val_batches_n

                print('Val Loss', mean_val_loss)

                print('Val Acc', mean_val_acc)



                if mean_val_loss < best_val_loss:

                    best_epoch_i = epoch_i

                    best_val_loss = mean_val_loss

                    best_model = copy.deepcopy(model)

                    print('New best model!')

                elif epoch_i - best_epoch_i > self._early_stopping_patience:

                    print('The model has not improved over the last {} epochs, stop learning.'.format(

                        self._early_stopping_patience))

                    break

                print()

            except KeyboardInterrupt:

                print('Stopped by user.')

                break

            except Exception as ex:

                print('Error: {}\n{}'.format(ex, traceback.format_exc()))

                break



        return best_val_loss, best_model
class Predict(object):

    def __init__(self, config):

        """Predict with trained model"""

        

        self._batch_size = config.get('batch_size', 32)



    def predict(self, model, dataset):

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        results_by_batch = []

        model.to(device)

        model.eval()



        dataloader = DataLoader(

            dataset, 

            batch_size=self._batch_size, 

            shuffle=False

        )

        labels = []

        with torch.no_grad():

            for batch_x in tqdm.tqdm(dataloader, total=len(dataset)/self._batch_size):

                batch_x = batch_x.to(device)

                batch_pred = model(batch_x)

                results_by_batch.append(batch_pred.detach().cpu().numpy())



        return np.concatenate(results_by_batch, 0)
%%time

train_tokens, word2id, word2freq = TextPreprocessor(config['TextPreprocessor']).transform(x_train)

valid_tokens, _, _ = TextPreprocessor(config['TextPreprocessor']).transform(x_valid)
%%time

train_vectors = VectorizeTexts(config['VectorizeTexts']).transform(train_tokens, word2id, word2freq)

valid_vectors = VectorizeTexts(config['VectorizeTexts']).transform(valid_tokens, word2id, word2freq)
train_dataset = SparseFitDataset(train_vectors, y_train.tolist())

val_dataset = SparseFitDataset(valid_vectors, y_valid.tolist())

best_val_loss, best_model = TrainModel(config['TrainModel']).train(

    model=nn.Linear(len(word2id), len(set(y_train))),

    train_dataset=train_dataset,

    val_dataset=val_dataset 

)
%%time

test_tokens, _, _ = TextPreprocessor(config['TextPreprocessor']).transform(df_test)

test_vectors = VectorizeTexts(config['VectorizeTexts']).transform(test_tokens, word2id, word2freq)
test_pred = Predict(config['Predict']).predict(best_model, SparsePredDataset(test_vectors))
submission = pd.DataFrame({'id': df_test.id, 'target': test_pred.argmax(-1)})
submission.target.value_counts()
submission.to_csv(submission_name, index=False)