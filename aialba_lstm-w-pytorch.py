import pandas as pd

import numpy as np



# Natural Language Processing

import nltk

import re

from keras.preprocessing import text, sequence



# Nueral Networks

from sklearn.model_selection import train_test_split

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F



# Misc Stuff

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import time

import random

import gc

import os

import multiprocessing as mp

import matplotlib.pyplot as plt
# disable progress bars when submitting

def is_interactive():

   return 'SHLVL' not in os.environ



if not is_interactive():

    def nop(it, *a, **k):

        return it



    tqdm = nop
# Code was taken from this kernel: https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version



def clean_special_chars(text):

    '''

    # Credit goes to: https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing



    Takes in text as input and returns the text with spaces around each item listed 

    in the variable punct. Special characters are additionally replaced in the outputted text.'''



    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    for p in punct:

        text = text.replace(p, f' {p} ')



    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

    for s in specials:

        text = text.replace(s, specials[s])



    return text



def clean_text(data):

    punct_pattern = '|'.join(["’", "‘", "´", "`"])

    possess_pattern = '|'.join(["'s", "s'"])



    # lowercasing the text

    data['comment_text'] = data['comment_text'].apply((lambda x: str.lower(x)))

    # replace incorrect/alternative forms of apostrophes to its correct form

    data['comment_text'] = data['comment_text'].str.replace(punct_pattern, "'")

    # expand contractions 

    data['comment_text'] = data['comment_text'].apply((lambda x: expand_contractions(x)))

    # remove endings for possessives

    data['comment_text'] = data['comment_text'].str.replace(possess_pattern, "") # embeddings for 9.072% training vocab after this step

    # separate text from symbols and punctuations

    data['comment_text'] = data['comment_text'].apply((lambda x: clean_special_chars(x))) # embeddings for 39.745% vocab

    # remove anything that is not characters or whitespace

    data['comment_text'] = data['comment_text'].apply((lambda x: re.sub('[^a-zA-z\s]', '', x)))

    # remove excess whitespace

    data['comment_text'] = data['comment_text'].astype(str).apply(lambda x: re.sub(' +', ' ',x))

    

    return data



# Credit goes to: https://github.com/kootenpv/contractions/blob/master/contractions/__init__.py

contractions_dict = {

    "ain't": "are not",

    "aren't": "are not",

    "can't": "cannot",

    "can't've": "cannot have",

    "'cause": "because",

    "c'mon": "come on",

    "could've": "could have",

    "couldn't": "could not",

    "couldn't've": "could not have",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hadn't've": "had not have",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'd've": "he would have",

    "he'll": "he will",

    "he'll've": "he will have",

    "he's": "he is",

    "how'd": "how did",

    "how're": "how are",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how is",

    "i'd": "i would",

    "i'd've": "i would have",

    "i'll": "i will",

    "i'll've": "i will have",

    "i'm": "i am",

    "i've": "i have",

    "isn't": "is not",

    "it'd": "it would",

    "it'd've": "it would have",

    "it'll": "it will",

    "it'll've": "it will have",

    "it's": "it is",

    "let's": "let us",

    "ma'am": "madam",

    "mayn't": "may not",

    "might've": "might have",

    "mightn't": "might not",

    "mightn't've": "might not have",

    "must've": "must have",

    "mustn't": "must not",

    "mustn't've": "must not have",

    "needn't": "need not",

    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",

    "oughtn't've": "ought not have",

    "shan't": "shall not",

    "sha'n't": "shall not",

    "shan't've": "shall not have",

    "she'd": "she would",

    "she'd've": "she would have",

    "she'll": "she will",

    "she'll've": "she will have",

    "she's": "she is",

    "should've": "should have",

    "shouldn't": "should not",

    "shouldn't've": "should not have",

    "so've": "so have",

    "so's": "so is",

    "that'd": "that would",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "there would",

    "there'd've": "there would have",

    "there's": "there is",

    "they'd": "they would",

    "they'd've": "they would have",

    "they'll": "they will",

    "they'll've": "they will have",

    "they're": "they are",

    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "we would",

    "we'd've": "we would have",

    "we'll": "we will",

    "we'll've": "we will have",

    "we're": "we are",

    "we've": "we have",

    "weren't": "were not",

    "what'll": "what will",

    "what'll've": "what will have",

    "what're": "what are",

    "what's": "what is",

    "what've": "what have",

    "when's": "when is",

    "when've": "when have",

    "where'd": "where did",

    "where's": "where is",

    "where've": "where have",

    "who'll": "who will",

    "who'll've": "who will have",

    "who's": "who is",

    "who've": "who have",

    "why's": "why is",

    "why've": "why have",

    "will've": "will have",

    "won't": "will not",

    "won't've": "will not have",

    "would've": "would have",

    "wouldn't": "would not",

    "wouldn't've": "would not have",

    "y'all": "you all",

    "y'all'd": "you all would",

    "y'all'd've": "you all would have",

    "y'all're": "you all are",

    "y'all've": "you all have",

    "you'd": "you would",

    "you'd've": "you would have",

    "you'll": "you will",

    "you'll've": "you shall have",

    "you're": "you are",

    "you've": "you have",

    "doin'": "doing",

    "goin'": "going",

    "nothin'": "nothing",

    "somethin'": "something",

}



contractions_re = re.compile('|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict = contractions_dict):

    def replace(match):

        v = match.group()

        if v in contractions_dict:

            return contractions_dict[v]

    return contractions_re.sub(replace, s)
def load_data(size=4):

    

    start = time.time()

    

    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

    

    train = clean_text(train)

    test = clean_text(test)

     

    x_train = train['comment_text']

    y_train = train['target']

    

    x_test = test['comment_text']

    

    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

    

    print('Data was successfully read in and preprocessed. Time taken: ',

      round((time.time() - start)/60, 2), ' minutes') 

    

    return  x_train, x_test, y_train, y_aux_train, test
def tokenize_text(x_train, x_test, pad_len):



    MAX_WORDS = 100_000

    

    start = time.time()

    

    tokenizer = text.Tokenizer(num_words = MAX_WORDS)

    tokenizer.fit_on_texts(list(x_train) + list(x_test))



    x_train = tokenizer.texts_to_sequences(x_train)

    x_train = sequence.pad_sequences(x_train, maxlen=pad_len)



    x_test = tokenizer.texts_to_sequences(x_test)

    x_test = sequence.pad_sequences(x_test, maxlen=pad_len)

        

    print('Text has been tokenized in', round((time.time() - start)/60, 2), ' minutes')

    

    return x_train, x_test, tokenizer

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path, encoding="utf8") as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))   



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words    



def make_embedder(tokenizer):

    

    GLOVE_PATH = '../input/crawl300d2m/crawl-300d-2M.vec'

    CRAWL_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'



    start = time.time()

    glove_matrix, unknown_words = build_matrix(tokenizer.word_index, GLOVE_PATH)

    crawl_matrix, unknown_words = build_matrix(tokenizer.word_index, CRAWL_PATH)



    embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis = -1)



    gc.collect()

    

    print('Embedding matrix was successfully made. Time taken:',

          round((time.time() - start)/60,2), ' minutes') 

    

    return embedding_matrix
def make_train_test(x_train, x_test, y_train, y_aux_train):

    

    # make test/train data for pytorch

    

    x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()

    x_test_torch =  torch.tensor(x_test, dtype=torch.long).cuda()

    y_train_torch = torch.tensor(

        np.hstack([y_train[:, np.newaxis], y_aux_train]),

        dtype=torch.float32).cuda()



    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)

    test_dataset = data.TensorDataset(x_test_torch)

    

    print('Pytorch data has been made.')

    

    return train_dataset, test_dataset, y_train_torch
class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x

    

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_aux_targets, LSTM_UNITS, max_features):

        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)

        

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

        

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = self.embedding_dropout(h_embedding)

        

        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)

        

        # global average pooling

        avg_pool = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool, _ = torch.max(h_lstm2, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))

        

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        

        result = self.linear_out(hidden)

        aux_result = self.linear_aux_out(hidden)

        out = torch.cat([result, aux_result], 1)

        

        return out
# Code was taken from this kernel: https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_model(model, train, test, loss_fn, output_dim,

                lr=0.001, batch_size=512, n_epochs=4, 

                enable_checkpoint_ensemble=True):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]

    optimizer = torch.optim.Adam(param_lrs, lr=lr)



    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    all_test_preds = []

    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    

#    model_stats = pd.DataFrame(columns = ['Epoch', 'Loss', 'Accuracy'])

    for epoch in range(n_epochs):

        start_time = time.time()

        

        scheduler.step()

        

        model.train()

        avg_loss = 0.

        

        for data in tqdm(train_loader, disable=False):

            x_batch = data[:-1]

            y_batch = data[-1]



            y_pred = model(*x_batch)            

            loss = loss_fn(y_pred, y_batch)



            optimizer.zero_grad()

            loss.backward()



            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

            

        model.eval()

        test_preds = np.zeros((len(test), output_dim))

    

        for i, x_batch in enumerate(test_loader):

            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())



            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

    

        all_test_preds.append(test_preds)

        elapsed_time = time.time() - start_time

        

        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(

              epoch + 1, n_epochs, avg_loss, elapsed_time))

        



            

    if enable_checkpoint_ensemble:

        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    

    else:

        test_preds = all_test_preds[-1]

        

    return test_preds
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

seed_everything()
# read in and preprocess data

x_train, x_test, y_train, y_aux_train, test = load_data(size=1)



# tokenize text

x_train, x_test, tokenizer = tokenize_text(x_train, x_test, pad_len = 205)



# embedding matrix

embedding_matrix = make_embedder(tokenizer)
# set variable units

lstm_units = 128

num_epochs = 5

pad_length = 250



# read in and preprocess data

x_train, x_test, y_train, y_aux_train, test = load_data(size=1)



# tokenize text

x_train, x_test, tokenizer = tokenize_text(x_train, x_test, pad_len = pad_length)



# embedding matrix

embedding_matrix = make_embedder(tokenizer)



# make train/test data for pytorch

train_dataset, test_dataset, y_train_torch = make_train_test(x_train, x_test, y_train, y_aux_train)



all_test_preds = []



start = time.time()



NUM_MODELS = 2

for model_idx in range(NUM_MODELS):

    print('Model ', model_idx)

    seed_everything(0 + model_idx)

    

    model = NeuralNet(

        embedding_matrix, y_aux_train.shape[-1],

        lstm_units, len(tokenizer.word_index) + 1)

    model.cuda()

    test_preds = train_model(

        model, train_dataset, test_dataset, 

        output_dim=y_train_torch.shape[-1], n_epochs = num_epochs,

        loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))

    

    all_test_preds.append(test_preds)

    print('Total time taken: ', round((time.time() - start)/60,2), 'minutes')

    print()
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': np.mean(all_test_preds, axis=0)[:, 0]

})



submission.to_csv('submission.csv', index=False)