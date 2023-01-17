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
dir_data = '../input/nlp-getting-started'

file_test = 'test.csv'

file_train = 'train.csv'



random_seed = 123

random.seed(random_seed)

np.random.seed(random_seed)

torch.manual_seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
config = {

    'TextPreprocessor': {

        'mode_stemming': True,

        'mode_norm': True,

        'mode_remove_stops': True,

        'mode_drop_long_words': True,

        'mode_drop_short_words': True,

        'min_len_word': 3,

        'max_len_word': 15,

        'max_size_vocab': 50000,

        'max_doc_freq': 0.9,

        'min_count': 5,

        'pad_word': '<PAD>',

        'text_column': 'text'        

    },

    'PaddedSequenceDataset': {

        'out_len': 100,

        'pad_value': 0

    },

    'SkipGramNegativeSamplingTrainer': {

        'emb_size': 100,

        'sentence_lent': 20,

        'radius': 5,

        'negative_samples_n': 50,

    },

    'train': {

        'lr': 1e-3,

        'epoch_n': 500,

        'batch_size': 64,

        'early_stopping_patience': 40,

        'max_batches_per_epoch_train': 2000

    },

    'Embeddings': {

        'topk': 10,

    }   

}
df_train = pd.read_csv(os.path.join(dir_data, file_train))

df_test = pd.read_csv(os.path.join(dir_data, file_test))

df_train.shape, df_test.shape
class TextPreprocessor(object):

    def __init__(self, config):

        """Preparing text features."""



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

        self._pad_word = config.get('pad_word', '<PAD>')

        self._text_column = config.get('text_column', 'union_text')



    def _clean_text(self, input_text):

        """Delete special symbols."""



        input_text = input_text.str.lower()

        input_text = input_text.str.replace(r'[^a-z ]+', ' ') 

        input_text = input_text.str.replace(r'http', ' ')

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

            sorted_word_counts = [(self._pad_word, 0)] + sorted_word_counts

            

        if len(word_counts) > self._max_size_vocab:

            sorted_word_counts = sorted_word_counts[:self._max_size_vocab]

            

        word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

        word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

        

        return word2id, word2freq

    

    def transform(self, df):       

        

        if self._text_column == 'union_text':

            columns_names = df.select_dtypes(include='object').columns

            df[columns_names] = df[columns_names].astype('str')

            for i in df.index:

                df.loc[i, self._text_column] = ' '.join(df.loc[i, columns_names])

            

        df[self._text_column] = self._clean_text(df[self._text_column])

        

        if self._mode_norm:

            df[self._text_column] = df[self._text_column].apply(self._text_normalization_en, 1)

            

        if self._mode_remove_stops:

            df[self._text_column] = df[self._text_column].apply(self._remove_stops_en, 1)

            

        if self._mode_stemming:

            df[self._text_column] = df[self._text_column].apply(self._stemming_en)

            

        if self._mode_drop_long_words:

            df[self._text_column] = df[self._text_column].apply(self._drop_long_words, 1)

            

        if self._mode_drop_short_words:

            df[self._text_column] = df[self._text_column].apply(self._drop_short_words, 1)

            

        df.loc[(df[self._text_column] == ''), (self._text_column)] = '<EMP>'

        

        tokenized_texts = [[word for word in text.split(' ')] for text in df[self._text_column]]

        word2id, word2freq = self._build_vocabulary(tokenized_texts)



        return tokenized_texts, word2id, word2freq

    

    def texts_to_token_ids(self, tokenized_texts, word2id):

        """Convert texts to Ids"""

        

        return [[word2id[token] for token in text if token in word2id]

                for text in tokenized_texts]



class PaddedSequenceDataset(Dataset):

    def __init__(self, config, texts, targets):

        

        self.texts = texts

        self.targets = targets

        self.out_len = config.get('out_len', 100)

        self.pad_value = config.get('pad_value', 0)

        

    def _ensure_length(self, txt, out_len, pad_value):

        if len(txt) < out_len:

            txt = list(txt) + [pad_value] * (out_len - len(txt))

        else:

            txt = txt[:out_len]

        return txt



    def __len__(self):

        return len(self.texts)



    def __getitem__(self, item):

        txt = self.texts[item]



        txt = self._ensure_length(txt, self.out_len, self.pad_value)

        txt = torch.tensor(txt, dtype=torch.long)



        target = torch.tensor(self.targets[item], dtype=torch.long)



        return txt, target



def add_fake_token(word2id, token='<PAD>'):

    word2id_new = {token: i + 1 for token, i in word2id.items()}

    word2id_new[token] = 0

    return word2id_new

    

def make_diag_mask(size, radius):

    """Square matrix Size x Size with two bars of width radius along the main diagonal."""

    idxs = torch.arange(size)

    abs_idx_diff = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs()

    mask = ((abs_idx_diff <= radius) & (abs_idx_diff > 0)).float()

    return mask



make_diag_mask(10, 3)





class SkipGramNegativeSamplingTrainer(nn.Module):

    def __init__(self, config, vocab_size):

        super().__init__()

        

        self.emb_size = config.get('emb_size', 100)

        self.sentence_len = config.get('sentence_len', 100)

        self.radius = config.get('radius', 5)

        self.negative_samples_n = config.get('negative_samples_n', 50)

        self.vocab_size = vocab_size



        self.center_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)

        self.center_emb.weight.data.uniform_(-1.0 / self.emb_size, 1.0 / self.emb_size)

        self.center_emb.weight.data[0] = 0



        self.context_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)        

        self.context_emb.weight.data.uniform_(-1.0 / self.emb_size, 1.0 / self.emb_size)

        self.context_emb.weight.data[0] = 0



        self.positive_sim_mask = make_diag_mask(self.sentence_len, self.radius)

    

    def forward(self, sentences):

        """sentences - Batch x MaxSentLength"""

        batch_size = sentences.shape[0]

        center_embeddings = self.center_emb(sentences)  # Batch x MaxSentLength x EmbSize



        # evaluate the similarity with real neighboring words

        positive_context_embs = self.context_emb(sentences).permute(0, 2, 1)  # Batch x EmbSize x MaxSentLength

        positive_sims = torch.bmm(center_embeddings, positive_context_embs)  # Batch x MaxSentLength x MaxSentLength

        positive_probs = torch.sigmoid(positive_sims)



        # increase the estimate of the probability of meeting these pairs of words together

        positive_mask = self.positive_sim_mask.to(positive_sims.device)

        positive_loss = F.binary_cross_entropy(positive_probs * positive_mask,

                                               positive_mask.expand_as(positive_probs))



        # choose random "negative" words

        negative_words = torch.randint(1, self.vocab_size,

                                       size=(batch_size, self.negative_samples_n),

                                       device=sentences.device)  # Batch x NegSamplesN

        negative_context_embs = self.context_emb(negative_words).permute(0, 2, 1)  # Batch x EmbSize x NegSamplesN

        negative_sims = torch.bmm(center_embeddings, negative_context_embs)  # Batch x MaxSentLength x NegSamplesN

        

        # decrease the score the probability of meeting these pairs of words together

        negative_loss = F.binary_cross_entropy_with_logits(negative_sims,

                                                           negative_sims.new_zeros(negative_sims.shape))



        return positive_loss + negative_loss





def no_loss(pred, target):

    """Fictitious loss function - when the model calculates the loss function itself"""

    return pred



def train_eval_loop(model, train_dataset, val_dataset, criterion,

                    lr=1e-4, epoch_n=10, batch_size=32,

                    early_stopping_patience=10, l2_reg_alpha=0,

                    max_batches_per_epoch_train=10000,

                    max_batches_per_epoch_val=1000,

                    data_loader_ctor=DataLoader,

                    optimizer_ctor=None,

                    lr_scheduler_ctor=None,

                    shuffle_train=True,

                    dataloader_workers_n=0):

    """

    Loop for training the model. After each epoch, the quality of the model is assessed by deferred sampling.

    

    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)



    if optimizer_ctor is None:

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)

    else:

        optimizer = optimizer_ctor(model.parameters(), lr=lr)



    if lr_scheduler_ctor is not None:

        lr_scheduler = lr_scheduler_ctor(optimizer)

    else:

        lr_scheduler = None



    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,

                                        num_workers=dataloader_workers_n)

    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,

                                      num_workers=dataloader_workers_n)



    best_val_loss = float('inf')

    best_epoch_i = 0

    best_model = copy.deepcopy(model)



    for epoch_i in range(epoch_n):

        try:

            epoch_start = datetime.datetime.now()

            print('Epoch {}'.format(epoch_i))



            model.train()

            mean_train_loss = 0

            train_batches_n = 0

            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):

                if batch_i > max_batches_per_epoch_train:

                    break



                batch_x = batch_x.to(device)

                batch_y = batch_y.to(device)



                pred = model(batch_x)

                loss = criterion(pred, batch_y)



                model.zero_grad()

                loss.backward()



                optimizer.step()



                mean_train_loss += float(loss)

                train_batches_n += 1



            mean_train_loss /= train_batches_n

            print('Epoch: {} iterations, {:0.2f} sec.'.format(train_batches_n,

                                                           (datetime.datetime.now() - epoch_start).total_seconds()))

            print('Mean train loss', mean_train_loss)







            model.eval()

            mean_val_loss = 0

            val_batches_n = 0



            with torch.no_grad():

                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):

                    if batch_i > max_batches_per_epoch_val:

                        break



                    batch_x = batch_x.to(device)

                    batch_y = batch_y.to(device)



                    pred = model(batch_x)

                    loss = criterion(pred, batch_y)



                    mean_val_loss += float(loss)

                    val_batches_n += 1



            mean_val_loss /= val_batches_n

            print('Mean valid loss', mean_val_loss)



            if mean_val_loss < best_val_loss:

                best_epoch_i = epoch_i

                best_val_loss = mean_val_loss

                best_model = copy.deepcopy(model)

                print('The new best model!')

            elif epoch_i - best_epoch_i > early_stopping_patience:

                print('The model has not improved over the last {} epochs, stop learning.'.format(

                    early_stopping_patience))

                break



            if lr_scheduler is not None:

                lr_scheduler.step(mean_val_loss)



            print()

        except KeyboardInterrupt:

            print('Stopped by user.')

            break

        except Exception as ex:

            print('Error: {}\n{}'.format(ex, traceback.format_exc()))

            break



    return best_val_loss, best_model





class Embeddings(object):

    def __init__(self, config, embeddings, word2id):

        

        self.topk = config.get('topk', 10)

        self.embeddings = embeddings

        self.embeddings /= (np.linalg.norm(self.embeddings, ord=2, axis=-1, keepdims=True) + 1e-4)

        self.word2id = word2id

        self.id2word = {i: w for w, i in word2id.items()}



    def most_similar(self, word):

        return self.most_similar_by_vector(self.get_vector(word))



    def analogy(self, a1, b1, a2):

        a1_v = self.get_vector(a1)

        b1_v = self.get_vector(b1)

        a2_v = self.get_vector(a2)

        query = b1_v - a1_v + a2_v

        return self.most_similar_by_vector(query)



    def most_similar_by_vector(self, query_vector):

        similarities = (self.embeddings * query_vector).sum(-1)

        best_indices = np.argpartition(-similarities, self.topk, axis=0)[:self.topk]

        result = [(self.id2word[i], similarities[i]) for i in best_indices]

        result.sort(key=lambda pair: -pair[1])

        return result



    def get_vector(self, word):

        if word not in self.word2id:

            raise ValueError('Uknown word "{}"'.format(word))

        return self.embeddings[self.word2id[word]]



    def get_vectors(self, *words):

        word_ids = [self.word2id[i] for i in words]

        vectors = np.stack([self.embeddings[i] for i in word_ids], axis=0)

        return vectors
train_tokens, word2id, word2freq = TextPreprocessor(config['TextPreprocessor']).transform(df_train)

test_tokens, _, _ = TextPreprocessor(config['TextPreprocessor']).transform(df_test)
train_token_ids = TextPreprocessor(config['TextPreprocessor']).texts_to_token_ids(test_tokens, word2id)

test_token_ids = TextPreprocessor(config['TextPreprocessor']).texts_to_token_ids(test_tokens, word2id)
train_dataset = PaddedSequenceDataset(

    config['PaddedSequenceDataset'], 

    train_token_ids, 

    np.zeros(len(train_token_ids))

)

test_dataset = PaddedSequenceDataset(

    config['PaddedSequenceDataset'],

    test_token_ids, 

    np.zeros(len(test_token_ids))

)
model = SkipGramNegativeSamplingTrainer(config['SkipGramNegativeSamplingTrainer'], len(word2id))
best_val_loss, best_model = train_eval_loop(

    model,

    train_dataset,

    test_dataset,

    no_loss,

    lr=config['train']['lr'],

    epoch_n=config['train']['epoch_n'],

    batch_size=config['train']['batch_size'],

    early_stopping_patience=config['train']['early_stopping_patience'],

    max_batches_per_epoch_train=config['train']['max_batches_per_epoch_train'],

    max_batches_per_epoch_val=len(test_dataset),

    lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)

)
embeddings = Embeddings(

    config['Embeddings'], 

    model.center_emb.weight.detach().cpu().numpy(), 

    word2id

)
train_tokens_clean = []

for i in train_tokens:

    i = [w if w in word2id else '<PAD>' for w in i]

    train_tokens_clean.append(i)
test_tokens_clean = []

for i in test_tokens:

    i = [w if w in word2id else '<PAD>' for w in i]

    test_tokens_clean.append(i)
embeddings.most_similar('california')
embeddings.most_similar('japan')
embeddings.most_similar('forest')