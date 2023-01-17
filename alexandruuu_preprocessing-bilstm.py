import operator

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegressionCV

import string

import numpy as np

import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

import os

print(os.listdir("../input"))

from sklearn.model_selection import train_test_split

import re

import time

import gc

import random

import os

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import torch.nn as nn

import torch.utils.data

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn
data = pd.read_csv("../input/train.csv",header=None,low_memory=False)

data_test = pd.read_csv("../input/test.csv",low_memory=False)



sentences = data[1][1:]

labels = data[2][1:]

sentences_test = []

labels.value_counts()
sentences.apply(lambda x: len(x.split())).plot(kind='hist');

plt.yscale('log')

plt.title('Distribution of question text length in words')

print('Average word length of questions in train is {0:.0f}.'.format(np.mean(sentences.apply(lambda x: len(x.split())))))

print('Average word length of questions in test is {0:.0f}.'.format(np.mean(sentences.apply(lambda x: len(x.split())))))
vec = CountVectorizer().fit(sentences)

bag_of_words = vec.transform(sentences)

sum_words = bag_of_words.sum(axis=0)



words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

words_freq_dict = dict(words_freq)

word_numbers = list(words_freq_dict.values())
print(np.sum(np.array(word_numbers)))

print(len(word_numbers))
sorted(word_numbers, reverse=True)[:10]
plt.hist(np.log10(word_numbers), bins=6)

plt.title("Log-log distribution of word frequencies")

plt.yscale('log')

plt.show()
words_freq[:10]
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
sentences = sentences.apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:5]})
def check_vocab_glove(corpus_vocab):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, 1

    

    embedding_vocab = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    oov = {}

    inv = {}

    

    for word, val in tqdm(corpus_vocab.items()):

        if word in embedding_vocab:

            inv[word] = val

        else:

            oov[word] = val

    return oov, inv
oov, inv = check_vocab_glove(vocab)
oov_words = list(oov.values())

inv_words = list(inv.values())

unique_inv = len(inv)

unique_oov = len(oov)

total_inv = np.sum(np.array(inv_words))

total_oov = np.sum(np.array(oov_words))
print(unique_inv/(unique_inv+unique_oov))

print(total_inv/(total_inv+total_oov))
np.max(np.array(oov_words))
#oov_words_freq =sorted(oov_, key = lambda x: x[1], reverse=True)

oov_words_freq = sorted(oov.items(), key=operator.itemgetter(1),reverse=True)
oov_words_freq[:10]
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x





def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    """Replace commonly misspelt words or contractions (e.g. can't -> cannot)"""

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
def load_and_prec():

    train_df = pd.read_csv("../input/train.csv")

    test_df = pd.read_csv("../input/test.csv")

    print("Train shape : ",train_df.shape)

    print("Test shape : ",test_df.shape)

    

    # lower

    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())

    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    

    # Clean the text

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))

    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    

    # Clean numbers

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))

    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

    

    # Clean speelings

    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    

    ## fill up the missing values

    train_X = train_df["question_text"].fillna("_##_").values

    test_X = test_df["question_text"].fillna("_##_").values

    

    

    ## Tokenize the sentences

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    test_X = tokenizer.texts_to_sequences(test_X)



    ## Pad the sentences 

    train_X = pad_sequences(train_X, maxlen=maxlen)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    ## Get the target values

    train_y = train_df['target'].values

    

    #shuffling the data

    trn_idx = np.random.permutation(len(train_X))



    train_X = train_X[trn_idx]

    train_y = train_y[trn_idx]

    

    return train_X, test_X, train_y, tokenizer.word_index, train_df
def load_glove(word_index):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 
maxlen = 72 # max number of words in a question to use

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)



start_time = time.time()



train_X, test_X, train_y, word_index, train_df = load_and_prec()

embedding_matrix = load_glove(word_index)



total_time = (time.time() - start_time) / 60

print("Took {:.2f} minutes".format(total_time))



print(np.shape(embedding_matrix))
finalword_index = word_index.items()
i = 0

def vocab_after_preprocess(train_df):



    inv = {}

    oov = {}

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, 1

    i = 0

    embedding_vocab = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    for sentence in train_df["question_text"]:

        i += 1

        words = sentence.split()

        for word in words:

            if word in embedding_vocab:

                if word in inv:

                    inv[word] +=1

                else:

                    inv[word]=1

            else:

                if word in oov:

                    oov[word] += 1

                else:

                    oov[word] = 1

    return inv, oov
inv_after, oov_after = vocab_after_preprocess(train_df)
print(len(inv_after)/(len(inv_after)+len(oov_after)))

print(sum(inv_after.values())/(sum(inv_after.values())+sum(oov_after.values())))
sorted_dict = sorted(oov_after.items(), key=lambda x: x[1], reverse=True)

sorted_dict[:10]
y = data[2][1:]

y = y.values

vectorizer = CountVectorizer(min_df=1)

sentences = data[1][1:]

X = vectorizer.fit_transform(list(sentences))



X[0].nonzero()



LR_model = LogisticRegressionCV(Cs=[0.1,0.05,0.01,0.005,0.003,0.001,0.0001],cv=5,random_state=0, solver='lbfgs').fit(X, y)



preds = LR_model.predict(X)



preds = [int(x) for x in preds]

y_true = [int(y_) for y_ in y]



f1_score(preds,y_true)
text_df
from scipy.sparse import coo_matrix

from scipy.sparse import csr_matrix

import tqdm

col = []

row = []



for ind, elem in tqdm.tqdm(enumerate(train_X)):

    col_ = [x for x in elem[elem>0]]

    row_ = np.ones(len(col_),dtype=np.int32)

    row_ = row_ * ind

    row.extend(row_)

    col.extend(col_)
data = np.ones(len(row))
lr_train_data = csr_matrix((data, (row, col)))
LR_model = LogisticRegressionCV(Cs=[0.1,0.05,0.01,0.005,0.003,0.001,0.0001],cv=5,random_state=0, solver='lbfgs').fit(lr_train_data,train_y)



preds = LR_model.predict(lr_train_data)



preds = [int(x) for x in preds]

y_true = [int(y_) for y_ in y]



f1_score(preds,y_true)
embed_size = 300 # how big is each word vector



batch_size = 512

train_epochs = 6

epochs=5
class BiLSTM(nn.Module):

    def __init__(self, embedding_matrix, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(p=dropout)

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        

        if static:

            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,

                            hidden_size=hidden_dim,

                            num_layers=lstm_layer, 

                            dropout = dropout,

                            bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim*lstm_layer*2, 1)

    

    def forward(self, sents):

        x = self.embedding(sents)

        x = torch.transpose(x, dim0=1, dim1=0)  # Swap batch and sentence dimensions

        lstm_out, (h_n, c_n) = self.lstm(x)

        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))

        return y
X_train, X_eval, y_train, y_eval = train_test_split(train_X,train_y,test_size=0.05)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
def train_BiLSTM(epochs, X_train, X_eval, y_train, y_eval):



    model = BiLSTM(embedding_matrix)

    model.cuda()

    

    x_train = torch.tensor(X_train, dtype=torch.long).cuda()

    y_train = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()

    x_val = torch.tensor(X_eval, dtype=torch.long).cuda()

    y_val = torch.tensor(y_eval[:, np.newaxis], dtype=torch.float32).cuda()

    

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters())

    

    train = torch.utils.data.TensorDataset(x_train, y_train)

    valid = torch.utils.data.TensorDataset(x_val, y_val)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    

    for epoch in range(epochs):

        start_time = time.time()

        model.train()

        

        avg_loss = 0

        for x_batch, y_batch in train_loader:

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

            

        model.eval()

        valid_preds = np.zeros((x_val.size(0)))

        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):

            y_pred = model(x_batch).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

            valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

            epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    return model
model = train_BiLSTM(train_epochs, X_train, X_eval, y_train, y_eval)
x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

y_pred_test = []

for i, (x_batch,) in enumerate(test_loader):

    y_pred = model(x_batch).detach()

    y_pred_test.extend(y_pred.cpu().numpy())

submission = data_test[['qid']].copy()

threshold = 0.5

submission['prediction'] = np.array(y_pred_test) > threshold

submission['prediction'] = submission['prediction'].astype(int)

submission.to_csv("submission.csv", index=False)