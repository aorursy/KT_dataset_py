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
filepath = '/kaggle/input/amazon-fine-food-reviews/Reviews.csv'
data = pd.read_csv(filepath)
data.head()
data = data[['Text', 'Score']]
data.head()
data.shape
import re
from nltk.corpus import stopwords

def decontract(sentence):
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub("", sentence)
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)
data['Text'] = data['Text'].str.lower()
data['Text'] = data['Text'].apply(decontract)
data['Text'] = data['Text'].apply(cleanPunc)
data['Text'] = data['Text'].apply(keepAlpha)
data['Text'] = data['Text'].apply(removeStopWords)
#removes characters repeated 
data['Text'] = data['Text'].apply(lambda x: re.sub(r'(\w)(\1{2,})', r'\1',x)) 
data['token_size'] = data['Text'].apply(lambda x: len(x.split(' ')))
data.describe()
# Avoid too much padding by filtering too long texts
data = data.loc[data['token_size'] < 60]
data.head()
data = data.sample(n= 50000)
# Construct a vocabulary
class ConstructVocab():
    
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for sent in self.sentences:
            self.vocab.update(sent.split(' '))
        
        #sort vacabulary
        self.vocab = sorted(self.vocab)
        
        #add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        #word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # 0 is the pad
            
        #index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
inputs = ConstructVocab(data['Text'].values.tolist())
inputs.vocab[:10]
input_tensor = [[inputs.word2idx[s] for s in es.split(' ')] for es in data['Text']]
input_tensor[:2]
def max_length(tensor):
    return max(len(t) for t in tensor)
max_length_input = max_length(input_tensor)
max_length_input
def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
        
    return padded
input_tensor = [pad_sequences(x, max_length_input) for x in input_tensor]
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import itertools
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

rates = list(set(data.Score.unique()))
num_rates = len(rates)

mlb = preprocessing.MultiLabelBinarizer()
data_labels = [set(rat) & set(rates) for rat in data[['Score']].values]
bin_rates = mlb.fit_transform(data_labels)
target_tensor = np.array(bin_rates.tolist())
target_tensor[:2]
data[:2]
get_rating =  lambda x: np.argmax(x)+1
get_rating(target_tensor[0])
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(input_tensor, target_tensor, 
                                                    test_size=0.2, random_state=1000)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, 
                                                    test_size=0.5, random_state=1000)

TRAIN_BUFFER_SIZE = len(X_train)
VAL_BUFFER_SIZE = len(X_val)
TEST_BUFFER_SIZE = len(X_test)
BATCH_SIZE = 64

TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
# Use Dataset class to represent the dataset object
class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [np.sum(1 - np.equal(x,0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        
        return x, y, x_len
    
    def __len__(self):
        return len(self.data)
import torch
from torch.autograd import Variable

train_dataset = MyData(X_train, y_train)
val_dataset = MyData(X_val, y_val)
test_dataset = MyData(X_test, y_test)

train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE,
                          drop_last=True, shuffle=True)
val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE,
                          drop_last=True, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE,
                          drop_last=True, shuffle=True)
train_dataset.dataset.target[:2]
embedding_dim = 256
units = 1024
vocab_inp_size = len(inputs.word2idx)
target_size = len(target_tensor[0])
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
def create_model(X_train):

    model = Sequential()
    model.add(Embedding(vocab_inp_size, embedding_dim, input_length=max_length_input))
    model.add(Dropout(0.5))
    model.add(GRU(units))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model
from keras.layers import Dense, Embedding, Dropout, GRU
from keras.models import Sequential
from keras import layers
from keras.utils.vis_utils import plot_model

model = create_model(pd.DataFrame(X_train))

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()
import matplotlib.pyplot as plt

timetaken = timecallback()
history = model.fit(pd.DataFrame(X_train), y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(pd.DataFrame(X_val), y_val),
                    batch_size=64,
                    callbacks = [timetaken])
time_epoch = []
time_epoch.append(timetaken.times[0][1])
for i in range(len(timetaken.times)-1):
    time_epoch.append(timetaken.times[i+1][1] - timetaken.times[i][1])
time_epoch
metrics = model.evaluate(pd.DataFrame(X_test), y_test)
print("Accuracy: {}".format(metrics[1]))
from sklearn import metrics
import seaborn as sns

predictions = model.predict_classes(pd.DataFrame(X_test)) 
labels = y_test.argmax(axis=1)
conf_matrix = metrics.confusion_matrix(predictions, labels)
conf_matrix = conf_matrix.astype(float)

for i in range(len(conf_matrix)):
    conf_matrix[i] = (conf_matrix[i]*100)/conf_matrix[i].sum()

df_cm = pd.DataFrame(conf_matrix, index = [i for i in "12345"],
                     columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)
sns.heatmap(df_cm, annot=True, cmap=cmap)
plt.title("Confusion matrix of categories (%)", fontsize=14)
plt.show()
#pip install pytorch-nlp
#pip install torchviz
import torch.nn as nn

class RateGRU(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(RateGRU, self).__init__()
        self.batch = batch_sz
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.output_size = output_size
        
        #layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units)
        self.fc = nn.Linear(self.hidden_units, self.output_size)
        
    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch, self.hidden_units)).to(device)
    
    def forward(self, x, lens, device):
        x = self.embedding(x)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(x, self.hidden)
        out = output[-1, :, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out, self.hidden
def sort_batch(X, y, lengths):
    "sort the batches by length"
    
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    
    return X.transpose(0, 1), y, lengths
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RateGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
model.to(device)

it = iter(train_dataset)
x, y, x_len = next(it)

xs, ys, lens = sort_batch(x, y, x_len)

print('Input size: ', xs.size())

output, _ = model(xs.to(device), lens, device)
print(output.size())
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RateGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
model.to(device)

#loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def loss_function(y, prediction):
    target = torch.max(y, 1)[1] 
    loss = criterion(prediction, target)
    
    return loss

def accuracy(target, logit):
    target = torch.max(target, 1)[1]
    corrects = (torch.max(logit, 1)[1].data == target).sum()
    accuracy = 100. * corrects / len(logit)
    
    return accuracy
from torchvision import models
print(model)
EPOCHS = 10

for epoch in range(EPOCHS):
    
    start = time.time()
    total_loss = 0
    train_accuracy, val_accuracy = 0, 0
    
    for (batch, (inp, targ, lens)) in enumerate(train_dataset):
        loss = 0
        predictions, _ = model(inp.permute(1, 0).to(device), lens, device)
        
        loss += loss_function(targ.to(device), predictions)
        batch_loss = (loss / int(targ.shape[1]))
        total_loss += batch_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_accuracy = accuracy(targ.to(device), predictions)
        train_accuracy += batch_accuracy
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Val Loss {:.4f}'.format(epoch + 1,
                                                              batch, 
                                                              batch_loss.cpu().detach().numpy()))
            
    for (batch, (inp, targ, lens)) in enumerate(val_dataset):
        
        predictions, _ = model(inp.permute(1, 0).to(device), lens, device)
        batch_accuracy = accuracy(targ.to(device), predictions)
        val_accuracy += batch_accuracy
        
    print('Epoch {} Loss {:.4f} -- Train Acc. {:.4f} -- Val Acc. {:.4f}'.format(epoch + 1,
                                                                                total_loss / TRAIN_N_BATCH,
                                                                                train_accuracy / TRAIN_N_BATCH,
                                                                                val_accuracy / TEST_N_BATCH))
        
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
x_raw = []
y_raw = []
all_predictions = []
test_accuracy = 0

for (batch, (inp, targ, lens)) in enumerate(test_dataset):
        
    predictions, _ = model(inp.permute(1, 0).to(device), lens, device)
    batch_accuracy = accuracy(targ.to(device), predictions)
    test_accuracy += batch_accuracy
    
    all_predictions = all_predictions + [i.item() for i in torch.max(predictions, 1)[1]]
    y_raw = y_raw + [y.item() for y in torch.max(targ, 1)[1]]
        
print('Test Accuracy {:.4f}'.format(test_accuracy.cpu().detach().numpy() / TEST_N_BATCH))
conf_matrix = metrics.confusion_matrix(all_predictions, y_raw)
conf_matrix = conf_matrix.astype(float)

for i in range(len(conf_matrix)):
    conf_matrix[i] = np.true_divide(conf_matrix[i],conf_matrix[i].sum())*100

df_cm = pd.DataFrame(conf_matrix, index = [i for i in "12345"],
                     columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)
sns.heatmap(df_cm, annot=True, cmap=cmap)
plt.title("Confusion matrix of categories (%)", fontsize=14)
plt.show()
283/conf_matrix[i].sum()