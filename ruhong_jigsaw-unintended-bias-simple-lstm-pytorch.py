import numpy as np

import pandas as pd

import os

import time

import gc

import random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F
# disable progress bars when submitting

def is_interactive():

   return 'SHLVL' not in os.environ



if not is_interactive():

    def nop(it, *a, **k):

        return it



    tqdm = nop
# config

SEED = 1234

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

NUM_MODELS = 10

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 240

EPOCHS = 1

BATCH_SIZE = 512

LEARNING_RATE = 1e-3

DROPOUT_RATE = 0.3

CHARS_TO_REMOVE = '""“”’`\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
CONTRACTIONS = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", 

                "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",

                "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",

                "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",

                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",

                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",

                "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",

                "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",

                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",

                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",

                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",

                "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",

                "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "it's": "it is" }
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything(SEED)
def clean_contractions(str, mapping):

    """Expand contractions"""

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        str = str.replace(s, "'")

    res = ' '.join([mapping[t] if t in mapping else t for t in str.split(' ')])

    return res



def clean_special_chars(text, punct, replacement=' '):

    '''Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution'''

    for p in punct:

        text = text.replace(p, replacement)

    return text



def preprocess(data):

    data = data.astype(str)

    data = data.apply(lambda x: clean_contractions(x, mapping=CONTRACTIONS))

    data = data.apply(lambda x: clean_special_chars(x, punct=CHARS_TO_REMOVE))

    return data



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

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
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_model(model, train, test, loss_fn, output_dim, lr=LEARNING_RATE,

                batch_size=BATCH_SIZE, n_epochs=EPOCHS,

                enable_checkpoint_ensemble=True):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]

    optimizer = torch.optim.Adam(param_lrs, lr=lr)



    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    all_test_preds = []

    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    

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
class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x

    

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_aux_targets):

        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(DROPOUT_RATE)

        

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
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train.shape, test.shape
%%time

x_train = preprocess(train['comment_text'])

y_train = np.where(train['target'] >= 0.5, 1, 0)

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

x_test = preprocess(test['comment_text'])
%%time

tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
max_features = len(tokenizer.word_index) + 1

max_features
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape



del crawl_matrix

del glove_matrix

gc.collect()
x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()

x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()

y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)

test_dataset = data.TensorDataset(x_test_torch)



all_test_preds = []



for model_idx in range(NUM_MODELS):

    print('Model ', model_idx + 1)

    seed_everything(SEED + model_idx)

    model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])

    model.cuda()

    test_preds = train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], 

                             loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))

    all_test_preds.append(test_preds)

    print()
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': np.mean(all_test_preds, axis=0)[:, 0]

})

submission.head()
submission.shape
submission.to_csv('submission.csv', index=False)

print(os.listdir("."))