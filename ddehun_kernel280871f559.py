# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_name = os.path.join(dirname, filename)

        if 'train.csv' in full_name:

            train_raw_path = full_name

        elif 'test.csv' in full_name:

            test_raw_path = full_name
import csv

import matplotlib.pyplot as plt





'''

Read Dataset

'''

def read_dataset(train_fname, test_fname):    

    train_data, test_data = {}, {}



    positive_counter = 0

    length_counter = []

    all_toks = {}

    

    # Data reading

    for setname in [train_fname, test_fname]:

        column_title = None

        with open(setname, newline='') as csvfile:

            reader = csv.reader(csvfile)

            for row in reader:

                if column_title is None:  # First row

                    column_title = row

                    continue

                sent1, sent2 = [int(el) for el in row[1].split()], [int(el) for el in row[2].split()]



                id_ = int(row[0])

                if len(column_title) == 4:  # Train

                    label = int(row[3])

                    positive_counter += label

                    train_data[id_] = {'sent1':sent1, 'sent2':sent2, 'label':label}

                    

                    length_counter.extend([len(sent1), len(sent2)])

                    

                    for tok in sent1+sent2:

                        assert isinstance(tok, int)

                        if tok not in all_toks: all_toks[tok] = 0

                        all_toks[tok] += 1

                        

                else:  # Test

                    test_data[id_] = {'sent1':sent1, 'sent2':sent2}

    return train_data, test_data, positive_counter, length_counter, all_toks



train_data, test_data, positive_counter, length_counter, all_toks = read_dataset(train_raw_path, test_raw_path)





'''

General Statistic

'''

print('[Dataset size]')

print("Train data size: {}".format(len(train_data)))

print("Test data size: {}".format(len(test_data)))



print('\n[Label Distribution]')

print("Pos/Neg: {}/{}".format(positive_counter, len(train_data)-positive_counter))



print('\n[Sentence Length]')

print("Median: {}".format(sorted(length_counter)[len(length_counter)//2]))

print("Average: {}".format(sum(length_counter)/len(length_counter)))

print("Max, Min: {} {}".format(max(length_counter), min(length_counter)))



print('\n[Vocab Size]')

print("Unique words count: {}".format(len(list(all_toks.keys()))))

import operator

sorted_all_toks = sorted(all_toks.items(), key=operator.itemgetter(1), reverse=True)



plt.figure(figsize=(15,10))

plt.hist([el[1] for el in sorted_all_toks], label='train',  normed=True)

plt.title('Word frequency')

plt.xlabel('Number of word')

plt.xlabel('Frequency')

plt.legend()

plt.show()





dupl_counter = {}  # 'sentence': count

for key,item in train_data.items():

    sent1, sent2 = ' '.join([str(el) for el in item['sent1']]), ' '.join([str(el) for el in item['sent2']])

    if sent1 in dupl_counter: dupl_counter[sent1] += 1

    else: dupl_counter[sent1] = 1

    if sent2 in dupl_counter: dupl_counter[sent2] += 1

    else: dupl_counter[sent2] = 1

sentence_frequency = list(dupl_counter.values())



plt.figure(figsize=(12,5))

plt.hist(sentence_frequency,label='train',normed=True)

plt.title('Histogram of sentence frequency')

plt.legend()

plt.show()
class Config:

    def __init__(self):

        self.w2v_dim = 200

        self.vocab_size = 8200

config = Config()



class Vocab():

    def __init__(self):

        self.word2id, self.id2word = {}, {}

    

    def __len__(self):

        assert len(self.word2id) == len(self.id2word)

        return len(self.word2id)

    

    def build_vocab(self, all_toks):

        # Only with train data, without test data

        assert len(self.word2id) == 0

        for idx, word in enumerate(all_toks):

            self.word2id[word[0]] = idx

        assert len(self.word2id) == len(all_toks) == config.vocab_size-1

        

        self.unk_id = len(self.word2id)

        self.word2id['<unk>'] = self.unk_id

        

        for k,v in self.word2id.items():

            self.id2word[v] = k

        assert len(self.word2id) == len(self.id2word)

        print("Vocab size is: {}".format(len(self.word2id)))

        self.vocab_size = len(self.word2id)

        assert self.vocab_size == config.vocab_size

    

    def sent2ids(self, sent):

        assert all([isinstance(tok, int) for tok in sent]) and isinstance(sent, list)

        return [self.word2id[tok] if tok in self.word2id else self.unk_id for tok in sent]
import pickle



kaggle_path = '/kaggle/working/'

train_bin_fname = kaggle_path + 'train_bin.pck'

test_bin_fname = kaggle_path + 'test_bin.pck'

vocab_bin_fname = kaggle_path + 'vocab_bin.pck'



def preprocess(train_data, test_data, all_toks):

    '''

    Save the data with replacing the old vocab into own.

    '''

    # Build Vocab

    vocab = Vocab()

    vocab.build_vocab(all_toks[:config.vocab_size-1])

    with open(vocab_bin_fname, 'wb') as f:

        pickle.dump(vocab, f)

    

    my_train_data, my_test_data = {}, {}

    

    for id_, item in train_data.items():

        train_data[id_]['sent1'] = vocab.sent2ids(item['sent1'])

        train_data[id_]['sent2'] = vocab.sent2ids(item['sent2'])



    with open(train_bin_fname, 'wb') as f:

        pickle.dump(train_data, f)

    

    for id_, item in test_data.items():

        test_data[id_]['sent1'] = vocab.sent2ids(item['sent1'])

        test_data[id_]['sent2'] = vocab.sent2ids(item['sent2'])

    with open(test_bin_fname, 'wb') as f:

        pickle.dump(test_data, f)

        

    return train_data, test_data, vocab

    

    

train_data, test_data, vocab = preprocess(train_data, test_data, sorted_all_toks)
'''

Utility functions

'''

def get_ngram(sent, gram=1):

    '''

    Args:

        sent: A list of integers

    Return:

        result: set of n-grams for the given sent

    '''

    assert isinstance(sent, list) and all([isinstance(el, int) for el in sent])

    result = []

    for idx, val in enumerate(sent):

        if idx == len(sent)-gram+1: break

        result.append(' '.join([str(el) for el in sent[idx:idx+gram]]))

    return set(result)



# Test

a = [1,2,3,4,5]

print(get_ngram(a,1))

print(get_ngram(a,2))

print(get_ngram(a,3))
from gensim.models import Word2Vec



'''

Train and load the w2v model

'''

print(vocab.unk_id)

def get_w2v_model():

    kaggle_path = '/kaggle/working/'

    embedding_path = os.path.join(kaggle_path, 'w2v.bin')



    

    sentences = []

    all_tok = []

    for k,v in train_data.items():

        sentences.append([str(el) for el in v['sent1']] + [str(el) for el in v['sent2']])   

        all_tok.extend(sentences[-1])



    model = Word2Vec(sentences, size=config.w2v_dim, window=3, min_count=1)

    model.save(embedding_path)

    print('8199' in all_tok)

    return model



model = get_w2v_model()

print(model)
kaggle_path = '/kaggle/working/'

embedding_path = os.path.join(kaggle_path, 'w2v.bin')
'''

Build idf matrix

'''

#Build IDF Matrix

inverted_index = {tok:[] for tok in range(len((vocab.word2id.keys())))}

for idx, sample in train_data.items():

    for word_id in sample['sent1']:

        assert isinstance(word_id, int)

        inverted_index[word_id].append(2*idx)



    for word_id in sample['sent2']:

        assert isinstance(word_id, int)

        inverted_index[word_id].append(2*idx + 1)



for k, v in inverted_index.items():

    inverted_index[k] = set(v)



idf_values = {k:np.log(2*len(list(train_data.keys())) / (len(v)+1))  for k,v in inverted_index.items()}
'''

A class for traditional NLP feature engineering

'''

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from numpy import dot

from numpy.linalg import norm

from scipy.spatial.distance import cdist

from scipy.stats import spearmanr

from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel

from sklearn.preprocessing import minmax_scale



#from nltk.translate.nist_score import sentence_nist



class FeatureEngineer():

    def __init__(self, idf_values):

        self.bleu_smoother = SmoothingFunction()

        self.w2v = get_w2v_model()

        self.idf_values = idf_values

        

    def get_feature(self, sent1, sent2):

        pair_feature = self.get_sentence_pair_feature(sent1, sent2)

        single_feature = self.get_sentence_feature(sent1, sent2)

        feature = np.concatenate((pair_feature, single_feature), axis=0)

        return feature

    

    def get_sentence_pair_feature(self, sent1, sent2):

        ngram = self.get_ngram_overlap(sent1, sent2)

        mt = self.get_mt_feature(sent1, sent2)

        return np.concatenate((ngram, mt), axis=0)

    

    def get_sentence_feature(self, sent1, sent2):

        s1_bow, s2_bow = self.get_bow_feature(sent1), self.get_bow_feature(sent2)

        s1_emb, s2_emb = self.get_embedding_feature(sent1), self.get_embedding_feature(sent2)

        

        bow_feature = self.kernal_for_single_sents(s1_bow, s2_bow)

        emb_feature = self.kernal_for_single_sents(s1_emb, s2_emb)

        return np.concatenate((bow_feature, emb_feature), axis=0)

    

    

    def get_bow_feature(self, sent):

        bow = [0 for _ in range(vocab.vocab_size)]

        for wordid in sent:

            bow[wordid] += self.idf_values[wordid]  # Weighted by idf

        return np.array(bow)

    

    def get_embedding_feature(self, sent):

        '''

        For a given sentence, make w2v feature with min/max/avg pooling.

        '''

        embedding_stack = np.asarray([self.w2v.wv[str(tok)] for tok in sent])

        min_pool, max_pool, avg_pool = embedding_stack.min(0), embedding_stack.max(0), np.average(embedding_stack, 0)

        

        return np.concatenate((min_pool, max_pool, avg_pool))

    

    def get_mt_feature(self, sent1, sent2):

        bleu = sentence_bleu([sent1], sent2, smoothing_function=self.bleu_smoother.method3)

        #nist = sentence_nist([sent1], sent2)

        #return [float(bleu), float(nist)]

        return np.array([float(bleu)])

    

    def get_ngram_overlap(self, sent1, sent2):

        # Original Ref: https://www.aclweb.org/anthology/S12-1060.pdf

        overlaps = [0,0,0]

        for n in range(3):

            sent1_gram = get_ngram(sent1, n+1)

            sent2_gram = get_ngram(sent2, n+1)

            len_sum = max(1, len(sent1_gram) + len(sent2_gram))

            overlaps[n] = 2 / len_sum * len(sent1_gram & sent2_gram)



        return np.array(overlaps)

    

    def kernal_for_single_sents(self, feature1, feature2):

        # To reduce the dimension of features of two sentences.

        cosine = 1 - dot(feature1, feature2)/(norm(feature1)*norm(feature2))

        manhanttan = 1 - cdist([feature1], [feature2], metric='cityblock')[0][0]

        euclidean = np.linalg.norm(feature1 - feature2)

        spearman = spearmanr(feature1, feature2)[0]

        sigmoid = 1/(1+np.exp(-dot(feature1, feature2)))

        rbf = rbf_kernel(np.array([feature1])-np.array([feature2]), gamma=1)[0][0]

        polynomial = polynomial_kernel(np.array([feature1]), np.array([feature2]))[0][0]

        laplacian = laplacian_kernel(np.array([feature1]), np.array([feature2]))[0][0]

        

        return np.array([cosine, manhanttan, euclidean, spearman, sigmoid, rbf, polynomial, laplacian])

    

engineer = FeatureEngineer(idf_values)



# Test

sent1, sent2 = [1,2,3,4,5],[4,5,6,7,8]

print(engineer.get_feature(sent1, sent2))



def scaling_feature(X):

    '''

    Scaling each feature into 0-1.

    '''

    return minmax_scale(X, axis=0, copy=True)    
from sklearn.model_selection import train_test_split



save_path = os.path.join(kaggle_path, 'processed/')



if not os.path.exists(save_path):

    os.makedirs(save_path)

serialized_fname = os.path.join(save_path, 'processed_everything.pck')

    

if os.path.exists(serialized_fname):

    with open(serialized_fname, 'rb') as f:

        X_train, X_dev, y_train, y_dev, test_X = pickle.load(f)

else:

    X, Y = [], []

    for cid, el in train_data.items():

        if cid%1000==0: print(cid)

        Y.append(el['label'])

        X.append(engineer.get_feature(el['sent1'], el['sent2']))



    X_train, X_dev, y_train, y_dev = train_test_split(np.array(X), np.array(Y), test_size=0.1, random_state=1515)



    test_X = []

    for cid, el in test_data.items():

        test_X.append(engineer.get_feature(el['sent1'], el['sent2']))

    test_X = np.array(test_X)



    # Serialize

    with open(os.path.join(save_path, 'processed_everything.pck'), 'wb') as f:

        pickle.dump([X_train, X_dev, y_train, y_dev, test_X], f)

        

'''Normalize each feature'''



scaled_X_train, scaled_X_dev = scaling_feature(X_train), scaling_feature(X_dev)

scaled_X_test = scaling_feature(test_X)
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE





model = TSNE(learning_rate=100)

transformed = model.fit_transform(scaled_X_train)



xs = transformed[:,0]

ys = transformed[:,1]

plt.scatter(xs,ys,c=Y[:36000], s=3)



plt.show()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import svm

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier





class FeatureRegressor():

    def __init__(self):

        pass

    

    def predict(self,X,Y):

        print("## Classifier Prediction Begin ##")

        self.rf_pred, self.gb_pred, self.svm_pred, self.xgb_pred, self.sgd_pred = self.rf.predict(X), self.gb.predict(X), self.svm.predict(X), self.xgb.predict(X), self.sgd.predict(X)

        rf_acc, gb_acc, svm_acc, xgb_acc, sgd_acc = accuracy_score(Y, self.rf_pred), accuracy_score(Y, self.gb_pred), accuracy_score(Y, self.svm_pred), accuracy_score(Y, self.xgb_pred), accuracy_score(Y, self.sgd_pred)

        

        print("## Individual Classifier Accuracy ##")

        print('Random Forest: {}'.format(rf_acc))

        print('Gradient Boosting: {}'.format(gb_acc))

        print('SVM: {}'.format(svm_acc))

        print('XGBoost: {}'.format(xgb_acc))

        print('SGD: {}'.format(sgd_acc))

        print('#'*15)

        

    def train_classifiers(self,X,Y):

        print("## Classifier Train Begin ##")

        self.RandomForest(X,Y)

        self.GradientBoosting(X,Y)

        self.SVM(X,Y)

        self.XGBoost(X,Y)

        self.SGD(X,Y)

    

    def RandomForest(self, X_train, Y_train):

        self.rf = RandomForestClassifier(n_estimators=100, oob_score=True)

        self.rf.fit(X_train, Y_train)

    

    def GradientBoosting(self, X_train, Y_train):

        self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

        self.gb.fit(X_train, Y_train)

    

    def SVM(self, X_train, Y_train):

        self.svm = svm.SVC()

        self.svm.fit(X_train, Y_train)

    

    def XGBoost(self, X_train, Y_train):

        ## Best Performace 0.77

        self.xgb = XGBClassifier(n_estimators=100, gamma=3, min_child_weight=1,max_depth=5)

        self.xgb.fit(X_train, Y_train)

        

    def SGD(self, X_train, Y_train):

        self.sgd = SGDClassifier(max_iter=500)

        self.sgd.fit(X_train, Y_train)

        

regressor = FeatureRegressor()
# Grid Hyperparmeter search

from sklearn.model_selection import KFold, GridSearchCV

from xgboost import XGBClassifier



if False:  # Set as True for tuning

    model = XGBClassifier()

    parameter_grid = {'booster':['gbtree'],

                     'silent':[True],

                     'max_depth':[5,8,10],

                     'min_child_weight':[1,3,5],

                     'gamma':[0,1,2,3],

                     'n_estimators':[100]}



    cv = KFold(n_splits=5, random_state=1)



    gcv = GridSearchCV(model, param_grid=parameter_grid, cv=cv, scoring='f1', n_jobs=5)



    gcv.fit(np.concatenate((scaled_X_train, scaled_X_dev),0), np.concatenate((y_train, y_dev), 0))

    print('final_params: ', gcv.best_params_)

    print('best score', gcv.best_score_)
TRAIN_WITH_FULL_DATA = True  # True when the model is trained with full train data (True for submission)



if TRAIN_WITH_FULL_DATA:

    regressor.train_classifiers(np.concatenate((scaled_X_train, scaled_X_dev),0), np.concatenate((y_train, y_dev), 0))

else:

    regressor.train_classifiers(scaled_X_train, y_train)

    

regressor.predict(scaled_X_dev, y_dev)

regressor_prediction = [regressor.xgb.predict(scaled_X_test)]

#regressor_prediction = [regressor.rf.predict(scaled_X_test), regressor.gb.predict(scaled_X_test), regressor.svm.predict(scaled_X_test), regressor.xgb.predict(scaled_X_test), regressor.sgd.predict(scaled_X_test)]
import string

import torch

from collections import Counter

from torch.utils.data import Dataset





'''

Data loader class for model

'''

class STSDataset(Dataset):

    def __init__(self, data, max_len=None, pad_idx=vocab.vocab_size, is_train=True):

        self.sent1_len = [len(el['sent1']) for k, el in data.items()]

        self.sent2_len = [len(el['sent2']) for k, el in data.items()]

        self.max_sent1_len, self.max_sent2_len = max(self.sent1_len), max(self.sent2_len)

        self.is_train = is_train

        self.data_num = len(self.sent1_len)

        

        self.data = {

            'ids': [],

            'sent1': torch.ones((self.data_num, self.max_sent1_len), dtype=torch.long) * pad_idx,

            'sent2': torch.ones((self.data_num, self.max_sent2_len), dtype=torch.long) * pad_idx}

        if self.is_train:

            self.data['labels']= torch.tensor([el['label'] for k, el in data.items()], dtype=torch.long)

                

        for idx, item in enumerate(data.values()):

            self.data['ids'].append(idx)

            final_pivot = min(len(item['sent1']), self.max_sent1_len)

            self.data['sent1'][idx][:final_pivot] = torch.tensor(item['sent1'][:final_pivot])

            final_pivot = min(len(item['sent2']), self.max_sent1_len)

            self.data['sent2'][idx][:final_pivot] = torch.tensor(item['sent2'][:final_pivot])

                          

    def __len__(self):

        return self.data_num

    

    def __getitem__(self, index):

        if self.is_train:

            return {'ids':self.data['ids'][index],

                   'sent1':self.data['sent1'][index],

                    'sent2':self.data['sent2'][index],

                    'sent1_len': min(self.sent1_len[index], self.max_sent1_len),

                    'sent2_len': min(self.sent2_len[index], self.max_sent2_len),

                    'label': self.data['labels'][index]}

        else:

            return {'ids':self.data['ids'][index],

                   'sent1':self.data['sent1'][index],

                    'sent2':self.data['sent2'][index],

                    'sent1_len': min(self.sent1_len[index], self.max_sent1_len),

                    'sent2_len': min(self.sent2_len[index], self.max_sent2_len)}    
# Hyperparameters

class DLConfig:

    def __init__(self):

        self.batch_size = 32

        self.epoch = 15

        self.lr = 3e-4

        self.keep_rate = 0.5

        self.num_class = 2

        self.max_grad_norm = 8

        self.hidden_dim = 200

        self.embed_dim = 200

        

        self.model_path = 'model/{}/'



config = DLConfig()

import torch.nn as nn





def sort_by_seq_lens(batch, sequences_lengths, descending=True):

    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)

    

    sorted_batch = batch.index_select(0, sorting_index)



    idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))

    _, reverse_mapping = sorting_index.sort(0, descending=False)

    restoration_index = idx_range.index_select(0, reverse_mapping)



    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index



def get_mask(sequences_batch, sequences_lengths):

    batch_size = sequences_batch.size()[0]

    max_length = torch.max(sequences_lengths)

    mask = torch.ones(batch_size, max_length, dtype=torch.float)

    mask[sequences_batch[:, :max_length] == 0] = 0.0

    return mask





def masked_softmax(tensor, mask):

    tensor_shape = tensor.size()

    reshaped_tensor = tensor.view(-1, tensor_shape[-1])



    # Reshape the mask so it matches the size of the input tensor.

    while mask.dim() < tensor.dim():

        mask = mask.unsqueeze(1)

    mask = mask.expand_as(tensor).contiguous().float()

    reshaped_mask = mask.view(-1, mask.size()[-1])



    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)

    result = result * reshaped_mask

    # 1e-13 is added to avoid divisions by zero.

    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)



    return result.view(*tensor_shape)





def weighted_sum(tensor, weights, mask):

    weighted_sum = weights.bmm(tensor)



    while mask.dim() < weighted_sum.dim():

        mask = mask.unsqueeze(1)

    mask = mask.transpose(-1, -2)

    mask = mask.expand_as(weighted_sum).contiguous().float()



    return weighted_sum * mask





def replace_masked(tensor, mask, value):

    mask = mask.unsqueeze(1).transpose(2, 1)

    reverse_mask = 1.0 - mask

    values_to_add = value * reverse_mask

    return tensor * mask + values_to_add



def check_prediction(prob, target):

    _, out_classes = prob.max(dim=1)

    correct = (out_classes==target).sum()

    return correct.item()


class RNNDropout(nn.Dropout):

    def forward(self, batch):

        ones = batch.new_ones(batch.shape[0], batch.shape[-1])

        mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)

        return mask.unsqueeze(1) * batch

    

class SeqEncoder(nn.Module):

    def __init__(self, rnn_type, input_dim, hidden_dim, layer_num, dropout, bidirectional=True):

        super(SeqEncoder, self).__init__()

        

        self.rnn_type = rnn_type

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.layer_num = layer_num

        self.dropout = dropout

        self.bidirectional = bidirectional

        

        self._encoder = rnn_type(input_dim, hidden_dim, num_layers=layer_num,bias=True, batch_first=True,

                                dropout=dropout, bidirectional=bidirectional)

        

    def forward(self, batch, lens):

        sorted_batch, sorted_lens, _, restoration_idx = sort_by_seq_lens(batch, lens)

        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lens, batch_first=True)

        

        outputs, _ = self._encoder(packed_batch, None)

        

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        reordered_outputs = outputs.index_select(0, restoration_idx)

        

        return reordered_outputs

    

class Attention(nn.Module):

    def forward(self, sent1_batch, sent1_mask, sent2_batch, sent2_mask):

        matrix = sent1_batch.bmm(sent2_batch.transpose(2,1).contiguous())

        

        sent1_attn = masked_softmax(matrix, sent2_mask)

        sent2_attn = masked_softmax(matrix.transpose(1,2).contiguous(), sent1_mask)

        

        attned_sent1 = weighted_sum(sent2_batch, sent1_attn, sent1_mask)

        attned_sent2 = weighted_sum(sent1_batch, sent2_attn, sent2_mask)

        

        return attned_sent1, attned_sent2
class ESIM(nn.Module):

    def __init__(self, vocab_size=vocab.vocab_size+1, embedding_dim=config.embed_dim, 

                 hidden_dim=config.hidden_dim, embeddings=None, padding_idx=vocab.vocab_size, 

                 keep_rate=config.keep_rate, num_classes=2, device='cpu'):

        super(ESIM, self).__init__()

        

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim

        self.num_classes = num_classes

        self.dropout = 1-keep_rate

        self.device = device

        

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)

        

        if embeddings is not None:

            self.word_embedding.weight = nn.Parameter(embeddings)

        self.rnn_dropout = RNNDropout(p=self.dropout)

        

        

        self.encoding = SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_dim, 1, self.dropout)

        

        self.attention = Attention()

        

        self.projection = nn.Sequential(nn.Linear(8 * self.hidden_dim, self.hidden_dim), nn.ReLU())

        

        self.composition = SeqEncoder(nn.LSTM, self.hidden_dim, self.hidden_dim, 1, self.dropout)

        

        self.fcn = nn.Sequential(nn.Dropout(p=self.dropout), nn.Linear(8*self.hidden_dim, self.hidden_dim),

                                nn.Tanh(), nn.Dropout(p=self.dropout), nn.Linear(self.hidden_dim, self.num_classes))

        

        self.apply(_init_model_weights)

        

    def forward(self, sent1, sent1_lens, sent2, sent2_lens):

        sent1_mask = get_mask(sent1, sent1_lens).to(self.device)

        sent2_mask = get_mask(sent2, sent2_lens).to(self.device)

        

        emb_sent1 = self.word_embedding(sent1)

        emb_sent2 = self.word_embedding(sent2)

        

        emb_sent1 = self.rnn_dropout(emb_sent1)

        emb_sent2 = self.rnn_dropout(emb_sent2)

        

        enc_sent1 = self.encoding(emb_sent1, sent1_lens)

        enc_sent2 = self.encoding(emb_sent2, sent2_lens)

        

        attn_sent1, attn_sent2 = self.attention(enc_sent1, sent1_mask, enc_sent2, sent2_mask)

        

        rich_sent1 = torch.cat([enc_sent1, attn_sent1, enc_sent1-attn_sent1, enc_sent1*attn_sent1], dim=-1)

        rich_sent2 = torch.cat([enc_sent2, attn_sent2, enc_sent2-attn_sent2, enc_sent2*attn_sent2], dim=-1)

        

        projected_sent1 = self.rnn_dropout(self.projection(rich_sent1))

        projected_sent2 = self.rnn_dropout(self.projection(rich_sent2))

        

        var_sent1 = self.composition(projected_sent1, sent1_lens)

        var_sent2 = self.composition(projected_sent2, sent2_lens)

        

        var_sent1_avg = torch.sum(var_sent1 * sent1_mask.unsqueeze(1).transpose(2,1), dim=1) / torch.sum(sent1_mask, dim=1, keepdim=True)

        var_sent2_avg = torch.sum(var_sent2 * sent2_mask.unsqueeze(1).transpose(2,1), dim=1) / torch.sum(sent2_mask, dim=1, keepdim=True)

        

        var_sent1_max, _ = replace_masked(var_sent1, sent1_mask, -1e7).max(dim=1)

        var_sent2_max, _ = replace_masked(var_sent2, sent2_mask, -1e7).max(dim=1)

        

        v = torch.cat([var_sent1_avg, var_sent1_max, var_sent2_avg, var_sent2_max], dim=-1)

        

        logits = self.fcn(v)

        prob = nn.functional.softmax(logits, dim=-1)

        

        return logits, prob

    

def _init_model_weights(module):

    if isinstance(module, nn.Linear):

        nn.init.xavier_uniform_(module.weight.data)

        nn.init.constant_(module.bias.data, 0.0)



    elif isinstance(module, nn.LSTM):

        nn.init.xavier_uniform_(module.weight_ih_l0.data)

        nn.init.orthogonal_(module.weight_hh_l0.data)

        nn.init.constant_(module.bias_ih_l0.data, 0.0)

        nn.init.constant_(module.bias_hh_l0.data, 0.0)

        hidden_size = module.bias_hh_l0.data.shape[0] // 4

        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0



        if (module.bidirectional):

            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)

            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)

            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)

            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)

            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

        

        
import shutil

from random import shuffle



import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from tqdm import tqdm

from tensorboardX import SummaryWriter







def build_env(train_, test_, config):

        

    train_dataset = STSDataset(esim_train_data)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = config.batch_size)



    test_dataset = STSDataset(esim_test_data)

    test_loader = DataLoader(test_dataset, shuffle=True, batch_size = config.batch_size)



    evaluation_dataset = STSDataset(test_data, is_train=False)

    evaluation_loader = DataLoader(evaluation_dataset, shuffle=False, batch_size=1)



    config.model_path = 'model/{}/'.format(config.exp_name)

    config.save_path, config.log_path = os.path.join(config.model_path, 'save'), os.path.join(config.model_path, 'logdir')



    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    return train_dataset, train_loader ,test_dataset, test_loader, evaluation_dataset, evaluation_loader, config, device





'''

Test Script

'''

def test_script(model, evaluation_loader, config, restore_epoch=13):

    config.batch_size = 1

    assert os.path.exists(config.save_path)

    

    model.eval()

    checkpoint = torch.load(os.path.join(config.save_path, 'esim_{}.pth.tar'.format(restore_epoch)))

    model.load_state_dict(checkpoint['model'])

    esim_result = []

    with torch.no_grad():

        for batch in evaluation_loader:

            sent1, sent1_len, sent2, sent2_len = batch['sent1'].to(device), batch['sent1_len'].to(device), batch['sent2'].to(device), batch['sent2_len'].to(device)

            _, prob = model(sent1, sent1_len, sent2, sent2_len)

            esim_result.append(prob)

    return esim_result

    

    

'''

Train Script

'''

def train(model, train_loader, test_loader, config):

    # save path and summary writer

    if os.path.exists(config.model_path):

        shutil.rmtree(config.model_path)

        

    os.makedirs(config.save_path)

    os.makedirs(config.log_path)



    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)



    # Begin training

    print("\n", '#'*30, 'TRAINING BEGIN', '#'*30)

    step = 0

    train_epoch_loss, train_epoch_acc, valid_epoch_loss, valid_epoch_acc = [[] for _ in range(4)]

    

    for epoch in range(config.epoch):

        model.train()



        train_loss, valid_loss = 0,0

        train_accuracy, valid_accuracy = 0,0

        batch_iterator = train_loader



        # Training

        for batch_index, batch in enumerate(batch_iterator):

            sent1, sent1_len, sent2, sent2_len, label = batch['sent1'].to(device), batch['sent1_len'].to(device), batch['sent2'].to(device), batch['sent2_len'].to(device), batch['label'].to(device)

            optimizer.zero_grad()



            logit, prob = model(sent1, sent1_len, sent2, sent2_len)



            loss = criterion(logit, label)

            accuracy = check_prediction(prob, label)



            loss.backward()



            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            train_loss += loss.item()

            train_accuracy += accuracy

            

            step += 1

        

        print("-> {} epoch {} step Loss {:.4f} Train Accuracy{:.4f}%".format(epoch, step, loss.item()/config.batch_size, accuracy/config.batch_size))

        train_loss /= len(train_loader)

        train_accuracy /= len(train_loader)



        # Validation

        with torch.no_grad():

            for batch in test_loader:

                sent1, sent1_len, sent2, sent2_len, label = batch['sent1'].to(device), batch['sent1_len'].to(device), batch['sent2'].to(device), batch['sent2_len'].to(device), batch['label'].to(device)

                logit, prob = model(sent1, sent1_len, sent2, sent2_len)

                loss = criterion(logit, label)

                accuracy = check_prediction(prob, label)

                valid_loss += loss.item()

                valid_accuracy += accuracy



        valid_loss /= len(test_loader)

        valid_accuracy /= len(test_loader)



        # Save the model at every epoch.

        torch.save({'epoch':epoch,

                    'model': model.state_dict(),

                    'train_loss':train_loss,

                   'valid_loss':valid_loss,

                   'train_acc':train_accuracy,

                   'valid_acc':valid_accuracy,

                   },os.path.join(config.save_path, 'esim_{}.pth.tar'.format(epoch)))

        train_epoch_loss.append(train_loss)

        train_epoch_acc.append(train_accuracy/config.batch_size*100)

        valid_epoch_loss.append(valid_loss)

        valid_epoch_acc.append(valid_accuracy/config.batch_size*100)

    

    # Draw the plot

    plt.figure()

    epoch_list = [_ for _ in range(config.epoch)]

    plt.plot(epoch_list, train_epoch_loss, "-r")

    plt.plot(epoch_list, valid_epoch_loss, "-b")

    plt.xlabel("epoch")

    plt.ylabel("loss")

    plt.legend(["Training loss", "Validation loss"])

    plt.title("Cross entropy loss")

    plt.show()

    plt.savefig(os.path.join(config.model_path, 'loss.png'))

    

    # Draw the plot

    plt.figure()

    plt.plot(epoch_list, train_epoch_acc, "-r")

    plt.plot(epoch_list, valid_epoch_acc, "-b")

    plt.xlabel("epoch")

    plt.ylabel("Accuracy(%)")

    plt.legend(["Training loss", "Validation loss"])

    plt.title("Accuracy")

    plt.show()

    plt.savefig(os.path.join(config.model_path, 'accuracy.png'))

    print("SAVEPATH: {}".format(os.path.join(config.model_path, 'accuracy.png')))

    

    print([round(el,2) for el in valid_epoch_acc])



    

    print('#' * 30,'Validation Best Accuracy: {} at {} epoch'.format(max(valid_epoch_acc), np.argmax(valid_epoch_acc)))
from pprint import pprint



'''

MAIN SCRIPT

'''



# Please set his value as False for trainin and testing.

IS_COMMIT = True



key_list = [key for key,val in train_data.items()]

shuffle(key_list)

esim_aggregation = []

exp_name = 'esim_{}_10fold'





if not IS_COMMIT: 

    for fold_index in range(10):

        # Dataset split

        esim_train_data, esim_test_data = {}, {}

        for k,v in train_data.items():

            if str(k)[-1] == str(fold_index): esim_test_data[k] = v

            else: esim_train_data[k] = v

        print("#"*10, "{} of 10 fold cross-validation begin".format(fold_index))



        exp_name = 'esim_{}_10fold'.format(fold_index)

        config.exp_name = exp_name

        train_dataset, train_loader ,test_dataset, test_loader, evaluation_dataset, evaluation_loader, config, device = build_env(esim_train_data, esim_test_data, config)

        w2v_model = Word2Vec.load('/kaggle/working/w2v.bin')

        w2v_embedding = torch.tensor(np.array([w2v_model.wv[str(el)] for el in range(vocab.vocab_size)] + [np.array([0.0 for _ in range(config.embed_dim)])]), dtype=torch.float).to(device)

        print("Experiment name: {}".format(config.exp_name))

        print("Save path: {}".format(config.save_path))

        print("Model path: {}".format(config.model_path))

        print("Device is {}".format(device))





        print("TRAINING START")

        model = ESIM(embeddings=w2v_embedding, device=device).to(device)

        train(model, train_loader ,test_loader, config)



        print("EVAL START")

        model = ESIM(keep_rate=1.0,device=device).to(device)

        esim_aggregation.append(test_script(model, evaluation_loader, config))

        

        # Aggregating ESIM CV results

        AGGREGATION_BY_PROB = False

        prob_results = [[0.0, 0.0] for _ in range(10000)]

        if AGGREGATION_BY_PROB:

            for one_fold in esim_aggregation:

                for idx, el in enumerate(one_fold):

                    el = el.cpu().numpy()[0]

                    assert len(el) == 2 

                    prob_results[idx][0] += el[0]

                    prob_results[idx][1] += el[1]

        else:

            for one_fold in esim_aggregation:

                for idx, el in enumerate(one_fold):

                    el = el.cpu().numpy()[0]

                    assert len(el) == 2

                    if el[0] > el[1]:

                        prob_results[idx][0] += 1

                    else:

                        prob_results[idx][1] += 1

        esim_voting_result = [0 if el[0]>el[1] else 1 for el in prob_results]

else:

    print("Commit gogo")
'''

Final Voting

'''

if not IS_COMMIT:

    SELECTED_APPROACH = 'esim'  # One of ['full_ensemble', 'feature', 'esim', 'xgb']

    SUBMISSION_FNAME = 'esim_10fold'#'esim_only'#'ensemble_full_data_esim_no_tuning_xgb_gb_esim'



    def ensemble(pred):

        '''

        Args:

            pred(list): List of prediction result for each classifier.

                        shape of [n_classifier, np.array(n_test_size)]

        '''

        prediction = np.transpose(np.array(pred))  # Shape of [test_sample_num, classifier_num]

        result = []

        for preds in prediction:

            if sum(preds) >= len(preds)/2: # positive is majority case

                result.append(1)

            else:

                result.append(0)

        return np.array(result)



    if SELECTED_APPROACH == 'full_ensemble':

        regressor_prediction.extend(esim_aggregation)

        voting_result = ensemble(regressor_prediction)

    elif SELECTED_APPROACH == 'esim':

        voting_result = ensemble([esim_voting_result])

    elif SELECTED_APPROACH == 'feature':

        voting_result = ensemble(regressor_prediction)

    elif SELECTED_APPROACH == 'xgb':

        assert len(regressor_prediction) == 1

        voting_result = ensemble(regressor_prediction)





    def make_submission(result, name=''):

        submission_dir = os.path.join(kaggle_path, 'submission_dir')

        if not os.path.exists(submission_dir):

            os.makedirs(submission_dir)

        fname = os.path.join(submission_dir, 'submission_chaehun_{}.csv'.format(name))

        with open(fname, 'w', newline='') as f:

            writer = csv.writer(f)

            writer.writerow(['id', 'label'])

            for idx, item in enumerate(result):

                writer.writerow([40001 + idx, item])



    make_submission(voting_result, name=SUBMISSION_FNAME)



    # To get the label distribution of test set

    # make_submission([1 for _ in range(10000)], name='all_1')

    # make_submission([0 for _ in range(10000)], name='all_0')