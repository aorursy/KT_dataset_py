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
import matplotlib.pyplot as plt

import re

import string

from tqdm import tqdm

import time

from collections import Counter



import nltk

import torch

import spacy

from sklearn.preprocessing import binarize



import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import random_split, Subset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)

print()
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')



train.dropna(inplace=True)
print(train.shape)

print(test.shape)
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    if (len(a) + len(b) - len(c)) == 0:

        return 0

    return float(len(c)) / (len(a) + len(b) - len(c))



def evaluate(true, pred):

    jac = 0

    for s1, s2 in zip(true, pred):

        jac += jaccard(s1, s2)

    jac /= len(true)

    return jac
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    # links

    text = re.sub('https?://\S+|www\.\S+', '', text)

    # multiple dots

    text = re.sub('<.*?>+', '', text)

    # punctuation

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # new lines

    text = re.sub('\n', '', text)

    # words containing numbers

    text = re.sub('\w*\d\w*', '', text)

    return text



# en = spacy.load('en') # en_core_web_sm



# def tokenize_en(sentence):

#     return [tok.text for tok in en.tokenizer(sentence)]



def preprocess_text(text):

    """

    Cleaning and parsing the text.

    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #tokenized_text = tokenize_en(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text



def preprocess_data(df):

    """

    Preprocess dataframe.

    text_clean and selected_text_clean will be used for training.

    """

    df['text_clean'] = df['text'].apply(str).apply(lambda x: preprocess_text(x))

    if 'selected_text' not in df.columns:

        df['selected_text_clean'] = ''

    else:

        df['selected_text_clean'] = df['selected_text'].apply(str).apply(lambda x: preprocess_text(x))

    # filter empty text after cleaning

    return df[df['text_clean'].map(len) > 0]

    
train = preprocess_data(train)

test = preprocess_data(test)

train.head(20)
train.iloc[6].text
text = '2am feedings for the baby are fun when he is all smiles and coos'.split()

selected_text = '2am feedings for the baby are fun when he is all smiles and coos'.split()



print(text)

print(selected_text)

1*np.isin(text, selected_text).astype(int)
test.head()
class Vocab:

    def __init__(self, texts):

        words = [w for sent in texts for w in sent.split()]

        self.counter = Counter(words)

        self.PAD_IND = 0

        self.TOK_IND = 1

        self.UNK_IND = 2

        self.word2index = {'PAD': self.PAD_IND, 'UNK': self.UNK_IND, 'TOK': self.TOK_IND}

        num_special = len(self.word2index)

        self.word2index.update({w: idx + num_special for idx, w in enumerate(set(words))})

        self.word2index.update({'positive': self.word2index.get('positive', len(self.word2index))})

        self.word2index.update({'negative': self.word2index.get('negative', len(self.word2index))})

        self.word2index.update({'neutral': self.word2index.get('neutral', len(self.word2index))})

        self.index2word = {ind: word for word, ind in self.word2index.items()}

    

    def __len__(self):

        return len(self.word2index)

    

    def __getitem__(self, key):

        return self.word2index.get(key, self.word2index['UNK'])
class Padder:

    def __init__(self, dim=0, pad_symbol=0, max_len=None):

        self.dim = dim

        self.pad_symbol = pad_symbol

        self.max_len = max_len

        

    def __call__(self, batch):

        def merge(sequences):

            lengths = [len(seq) for seq in sequences]

            max_len = self.max_len if self.max_len is not None else max(lengths)

            padded_seqs = torch.zeros(len(sequences), max_len).long()

            for i, seq in enumerate(sequences):

                end = lengths[i]

                padded_seqs[i, :end] = seq[:end]

            return padded_seqs, lengths

    

        sentiment, x, y = zip(*batch)



        sentiment = torch.cat(sentiment)

        x, x_lengths = merge(x)

        y, y_lengths = merge(y)



        return sentiment, x, y



class SentimentDataset:

    def __init__(self, text, sentiment, selected_text, vocab):

        self.text = text

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.vocab = vocab

    

    def __len__(self):

        return len(self.text)

    

    def _prepare_data(self, sentiment, text, selected_text):

        text_words = text.split()

        selected_text_words = selected_text.split()

        selected = np.isin(text_words, selected_text_words).astype(int)

                    

        x_seq = [self.vocab[w] for w in text_words]

        y_seq = self.vocab.TOK_IND*selected

        

        return [self.vocab[sentiment]], x_seq, y_seq



    def __getitem__(self, idx):

        orig_text = self.text[idx]

        orig_selected_text = self.selected_text[idx]

        orig_sentiment = self.sentiment[idx]

        sentiment, x, y = self._prepare_data(sentiment=orig_sentiment,

                                  text=orig_text,

                                  selected_text=orig_selected_text)

        sentiment = torch.tensor(sentiment, dtype=torch.long)

        x = torch.tensor(x, dtype=torch.long)

        y = torch.tensor(y, dtype=torch.long)

        return sentiment, x, y #, orig_text, orig_selected_text, orig_sentiment
MAX_LEN = 35

vocab = Vocab(train['text_clean'].values)

train_dataset = SentimentDataset(

    text=train['text_clean'].values,

    sentiment=train['sentiment'].values,

    selected_text=train['selected_text_clean'].values,

    vocab=vocab

)

test_dataset = SentimentDataset(

    text=test['text_clean'].values,

    sentiment=test['sentiment'].values,

    selected_text=train['selected_text_clean'].values,

    vocab=vocab

)
size = len(train_dataset)

print(f'dataset size={size}')

val_size = int(size*0.15)

print(f'val size={val_size}')

train_size = size - val_size

print(f'train size={train_size}')

print(f'test size={len(test_dataset)}')
trn, val = random_split(train_dataset, [train_size, val_size])

tst = test_dataset
print(len(val))

print(len(trn))
print(vocab['neutral'])

print(vocab['positive'])

print(vocab['negative'])
# train_data_loader = torch.utils.data.DataLoader(

#     train_dataset,

#     batch_size=3,

#     collate_fn=Padder(max_len=MAX_LEN)

# )

# for i, (sentiment, x, y) in enumerate(train_data_loader):

# #     print(f'text={orig_text}')

# #     print(f'selected_text={orig_selected_text}')    

# #     print(f'sentiment={orig_sentiment}')    

#     print(f'sentiment={sentiment}')   

#     print(f'x={x.shape}')   

#     print(f'y={y.shape}')    

#     print()
def load_glove():

    f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')

    

    embeddings = {}

    for i, line in enumerate(tqdm(f)):

        value = line.split(' ')

        word = value[0]

        vec = np.array(value[1:],dtype = 'float32')

        embeddings[word] = vec

    return embeddings



glove_embeddings = load_glove()

EMBEDDING_SIZE = list(glove_embeddings.values())[0].shape[0]
len(glove_embeddings)
def build_embedding_matrix(vocab: Vocab, embeddings: dict):

    num_words = len(vocab)

    embedding_size = list(embeddings.values())[0].shape[0]

    matrix = np.empty((num_words, embedding_size))

    default_emb = np.mean(list(embeddings.values()), axis=0)

    words_not_found = []

    for word, ind in vocab.word2index.items():

        if word not in embeddings:

            words_not_found.append(word)

            matrix[ind] = default_emb

        else:

            matrix[ind] = embeddings[word]

    print(f'Embedding not found for {len(words_not_found)} words out of {num_words}')

    return matrix, words_not_found
emb_matrix, not_found = build_embedding_matrix(vocab, glove_embeddings)

emb_matrix = torch.LongTensor(emb_matrix)

del glove_embeddings
emb_matrix = torch.LongTensor(emb_matrix)
class RNN(nn.Module):

    def __init__(self, hidden_dim, emb_dim, num_embeddings, emb_vectors=None, padding_idx=None, dropout=0):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings=num_embeddings,

                                embedding_dim=emb_dim,

                                padding_idx=padding_idx)

        if emb_vectors is not None:

            self.emb.weight.data.copy_(emb_vectors)

        self.rnn = nn.LSTM(input_size=emb_dim,

                          hidden_size=hidden_dim,

                          batch_first=True,

                          bidirectional=True

                        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.out_size = MAX_LEN

        self.fc = nn.Linear(in_features=hidden_dim*2,

                            out_features=self.out_size)



    def forward(self, sentiment, input, hidden=None):

        # input   :  (batch_size, seq_length)

        # add sentiment before input sequence

        input = torch.cat([sentiment.unsqueeze(dim=1).t(), input.t()]).t()

        batch_size = input.shape[0]

        emb = self.emb(input)                   # (batch_size, seq_length, emb_dim)

        #print(f'emb shape= {emb.shape}')

        rnn_out, hidden = self.rnn(emb, hidden) # (batch_size, seq_length, hidden_dim*2)

        out = self.relu(rnn_out)                # (batch_size, seq_length, hidden_dim*2)

        out = self.dropout(out)                 # (batch_size, seq_length, hidden_dim*2)

        out = self.fc(out)                      # (batch_size, seq_length, output_size)

        #print(f'out1 shape= {out.shape}')

        out = out.view(batch_size, -1)          # (batch_size, seq_length*output_size)

        #print(f'out2 shape= {out.shape}')

        # get last batch of labels

        preds = out[:, -self.out_size:]

        #print(f'preds shape= {preds.shape}')

        return preds
HIDDEN_DIM = 300

EMBEDDING_SIZE = 300

model = RNN(hidden_dim=HIDDEN_DIM, emb_dim=EMBEDDING_SIZE, num_embeddings=len(vocab), emb_vectors=emb_matrix, padding_idx=vocab.PAD_IND, dropout=0.1)

print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_func = nn.BCEWithLogitsLoss()

N_EPOCHS = 12

BATCH_SIZE = 32
def train(model, train, val, optimizer, loss_func, batch_size=BATCH_SIZE, epochs=N_EPOCHS):

    losses = []

    val_losses = []

    times = []

    if torch.cuda.is_available():

        model.cuda()

    for epoch in range(epochs):

        start = time.time()

        total_loss = 0

        batcher = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=Padder(max_len=MAX_LEN))

        t = 0

        model.train()

        for sentiment, x, y in batcher:

            if torch.cuda.is_available():

                sentiment = sentiment.cuda()

                x = x.cuda()

                y = y.cuda()

            y = y.float()



            preds = model(sentiment, x)   

            loss = loss_func(preds, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()



            t += 1

        total_loss /= len(batcher)

        losses.append(total_loss)



        # validation loss

        test_batcher = torch.utils.data.DataLoader(dataset=val, batch_size=len(val), shuffle=False, collate_fn=Padder(max_len=MAX_LEN))

        sentiment, test_x, test_y = next(iter(test_batcher))



        if torch.cuda.is_available():

            sentiment = sentiment.cuda()

            test_x = test_x.cuda()

            test_y = test_y.cuda()

        test_y = test_y.float()

        optimizer.zero_grad()

        model.eval()

        preds = model(sentiment, test_x)   

        val_loss = loss_func(preds, test_y)

        val_loss = val_loss.item()

        val_losses.append(val_loss)



        end = time.time()

        times.append(float(end - start)/60)

        print(f'Epoch {epoch}')

        print(f'\tTraining loss = {total_loss:.4}')

        print(f'\tValidation loss = {val_loss:.5}')

        print('\tEpoch took %.2f minutes' % (float(end - start)/60))

    return losses, val_losses
def plot_losses(losses, val_losses):

    fig, ax = plt.subplots(1,1, figsize=(16,4))

    ax.set(xlabel='epoch', ylabel='total loss',

        title='loss per epoch')

    ax.grid()

    losses = np.array(losses)

    ax.plot(losses, color='b', label='Train loss')

    ax.plot(val_losses, color='r', label='Validation loss')

    ax.legend()
losses, val_losses = train(model, trn, val, optimizer, loss_func)

plot_losses(losses, val_losses)
def decode_prediction(test_x, preds, threshold=0.5):

    test_x = test_x.to('cpu').detach().numpy()

    preds = preds.to('cpu').detach().numpy()

        

    binary_preds = binarize(preds, threshold=threshold)

    selected_words = []

    for test_seq, pred_seq in zip(test_x, binary_preds):

        test_inds = test_seq

        selected_seq = [vocab.index2word[ind] for ind, select in zip(test_inds, pred_seq) if select == vocab.TOK_IND]

        selected_words.append(selected_seq)

    return selected_words
def validate(model, dataset, threshold):

    # get prediction

    test_batcher = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, collate_fn=Padder(max_len=MAX_LEN))

    sentiment, test_x, test_y = next(iter(test_batcher))

    if torch.cuda.is_available():

        sentiment = sentiment.cuda()

        test_x = test_x.cuda()

    optimizer.zero_grad()

    model.eval()

    preds = model(sentiment, test_x)   

    probability_preds = torch.sigmoid(preds)



    y_pred = decode_prediction(test_x, probability_preds, threshold=threshold)

    y_true = decode_prediction(test_x, test_y)

    y_pred_text = [' '.join(words) for words in y_pred]

    y_true_text = [' '.join(words) for words in y_true]

    score = evaluate(y_true_text, y_pred_text)

    print(f'Jaccard test score={score:.3f}')

    return score
validate(model, val, threshold=0.4)
def split_by_sentiment(dataset):

    sentiment_inds = {

        'neutral': [],

        'positive': [],

        'negative': [],

    }

    for i, (sentiment_tok, x, y) in enumerate(dataset):

        sentiment = vocab.index2word[sentiment_tok.numpy()[0]]

        sentiment_inds[sentiment].append(i)

    ds_neutral = Subset(dataset, sentiment_inds['neutral'])

    ds_positive = Subset(dataset, sentiment_inds['positive'])

    ds_negative = Subset(dataset, sentiment_inds['negative'])

    return {

        'neutral': ds_neutral,

        'positive': ds_positive,

        'negative': ds_negative,

    }
val_by_sentiment = split_by_sentiment(val)

val_neutral = val_by_sentiment['neutral']

val_positive = val_by_sentiment['positive']

val_negative = val_by_sentiment['negative']



validate(model, val_neutral, threshold=0.4)

validate(model, val_positive, threshold=0.4)

validate(model, val_negative, threshold=0.4)
# get prediction

test_batcher = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=Padder(max_len=MAX_LEN))

sentiment, test_x, test_y = next(iter(test_batcher))

if torch.cuda.is_available():

    sentiment = sentiment.cuda()

    test_x = test_x.cuda()

model.eval()

preds = model(sentiment, test_x)   

probability_preds = torch.sigmoid(preds)



y_pred = decode_prediction(test_x, probability_preds, threshold=0.4)

y_true = decode_prediction(test_x, test_y)

y_pred_text = [' '.join(words) for words in y_pred]

y_true_text = [' '.join(words) for words in y_true]

score = evaluate(y_true_text, y_pred_text)
assert(len(y_pred_text) == len(test))

y_pred_text = [txt.replace(' PAD', '') for txt in y_pred_text]

test['selected_text']=y_pred_text

# if neutral leave whole text

test.loc[test['sentiment'] == 'neutral', 'selected_text'] = test.loc[test['sentiment'] == 'neutral', 'text']
ss.loc[:, 'selected_text']=test['selected_text']

ss[['textID','selected_text']].to_csv('submission.csv', index=False)
checkpoint = {'model': model,

              'state_dict': model.state_dict(),

              'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'base_rnn_checkpoint')