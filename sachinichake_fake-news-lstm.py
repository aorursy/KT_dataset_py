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
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from sklearn import metrics
from torch.nn import functional as F
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()  
import string
MAX_LEN = 512
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
EPOCHS = 100
MODEL_PATH = "../input/fake-news-dataset/modelv2.pt"
TRAINING_FILE = "../input/fake-news-dataset/train.csv"
TESTING_FILE = "../input/fake-news-dataset/test.csv"
GLOVE = "../input/glove6b300dtxt/glove.6B.300d.txt"
SUBMIT_FILE ="../input/fake-news-dataset/submit.csv"

max_features = None

# CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = GLOVE
NUM_MODELS = 1
LSTM_UNITS = 512
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path,encoding='utf-8') as f:
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

def train_model(model, train, test, loss_fn, lr=0.001,
                batch_size=512, n_epochs=EPOCHS,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    best_avg_loss = 0.0
    best_accuracy=0

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
        test_preds = np.zeros(len(test))

        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i + 1) * batch_size] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, elapsed_time))
        
        # if enable_checkpoint_ensemble:
        #   test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        # else:
        test_preds = all_test_preds[-1]

        outputs = np.array(test_preds) >= 0.5
        outputs= outputs*1
        accuracy = metrics.accuracy_score(df_submit.label.values,outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy
        # if avg_loss < best_avg_loss:
        #     torch.save(model.state_dict(), MODEL_PATH)
        #     best_avg_loss = accuracy
          


    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
    else:
        test_preds = all_test_preds[-1]

    return test_preds
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        

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
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        

        hidden = h_conc + h_conc_linear1

        result = self.linear_out(hidden).squeeze()
        

        return result

# def preprocess(data):
#     '''
#     Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
#     '''
#     punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
#     def clean_special_chars(text, punct):
#         for p in punct:
#             text = text.replace(p, ' ')
#         return text

#     data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
#     return data
import re 
def preprocess(data):
    by_article_list=[]
    for text in (data):
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = text.lower().split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        by_article_list.append(text)
    return by_article_list



# def preprocess(data):
#     by_article_list=[]
#     for article in (data):
#         words = word_tokenize(article)
#         words = [word.lower() for word in words if word.isalpha()] #lowercase
#         words = [word for word in words if word not in string.punctuation and word not in stop_words] #punctuation, stopwords
#         words = [lemmatizer.lemmatize(word) for word in words] #convert word to root form
#         by_article_list.append(' '.join(words))
#     return by_article_list
train = pd.read_csv(TRAINING_FILE).fillna("none").reset_index(drop=True)
test = pd.read_csv(TESTING_FILE).fillna("none").reset_index(drop=True)
df_submit= pd.read_csv(SUBMIT_FILE)

train.dropna(inplace=True)

x_train = preprocess(train['text'])
y_train = train['label'] 
x_test = preprocess(test['text'])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


max_features = max_features or len(tokenizer.word_index) + 1


glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('n unknown words (glove): ', len(unknown_words_glove))

embedding_matrix = glove_matrix
embedding_matrix.shape

# del crawl_matrix
del glove_matrix
gc.collect()


x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
y_train_torch = torch.tensor(y_train, dtype=torch.float32).cuda()

train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = data.TensorDataset(x_test_torch)

# all_test_preds = []

model = NeuralNet(embedding_matrix)
model.cuda()

test_preds = train_model(model, train_dataset, test_dataset, 
                        loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))


outputs = np.array(test_preds) >= 0.5
outputs= outputs*1
outputs_series = pd.Series(outputs)
pred_series = pd.Series(test_preds) 

submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction':pred_series,
    'outputs':outputs_series
})
submission.to_csv('/content/drive/My Drive/datasets/data/submission.csv', index=False)


# all_test_preds.append(test_preds)
# print()
