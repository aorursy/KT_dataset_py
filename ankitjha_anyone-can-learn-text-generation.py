import numpy as np

import pandas as pd

from tqdm import tqdm

from collections import Counter

!pip install pyspellchecker

import re 

from wordcloud import WordCloud, STOPWORDS

from spellchecker import SpellChecker

import os

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from collections import Counter

%matplotlib inline

import torch.utils.data

from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import os
df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')

df.dropna(inplace = True)

df.head()
# Function to check presence of a digit in a string

def contains_digit(string):

    return bool(re.search(r'\d', string))
class TextPreprocessing:

    def __init__(self, train):

        self.text = ''

        self.cleaned_text = []

        self.input_text = ''

        self.train = train

        

    def gettext(self, df, colname, num_cols):

        if self.train == True:

            #concatenating all the column values to form a single string

            for line in df[colname].values[:num_cols]:

                self.text += line

        else:

            for line in df[colname].values[num_cols:]:

                self.text += line

    

    # Cleaning the text

    def clean_text(self):

        self.text = re.sub("([\(\[]).*?([\)\]])", "", self.text)

        fullstops = ['...', '..', '!!!', '!!', '!']

        for s in fullstops:

            self.text = self.text.replace(s, ".")

    

    # Removing sentences with digits

    def remove_numbers(self):

        for sentence in self.text.split('.'):

            if(contains_digit(sentence) == False):

                self.cleaned_text.append(sentence.rstrip())

        self.input_text = '.'.join(sentence for sentence in self.cleaned_text)

    

    # Spell correction of misspelled words

    def spell_correction(self):

        _spell = SpellChecker()

        incorrect_words = []

        correct_words = []

        # Finding all words in the input text

        res = re.findall(r'\w+', self.input_text) 

        for word in tqdm(set(res)):

            correct = _spell.correction(word)

            if(word != correct):

                incorrect_words.append(word)

                correct_words.append(correct)

        for i, word in enumerate(incorrect_words):

            self.input_text = self.input_text.replace(word, correct_words[i])

        return self.input_text.lower()
text_data = TextPreprocessing(train = True)

text_data.gettext(df, 'Review Text', 100)

text_data.clean_text()

text_data.remove_numbers()

input_text = text_data.spell_correction()

# Here, input text contains the preprocessed text.

train_text = input_text.split()



# Fetching data for validation

valid_text_data = TextPreprocessing(False)

valid_text_data.gettext(df, 'Review Text', -100)

valid_text_data.clean_text()

valid_text_data.remove_numbers()

valid_input_text = valid_text_data.spell_correction()

valid_text = valid_input_text.split()
def wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

wordcloud(train_text, title="Word Cloud of Text")
words_set = set(train_text).union(set(valid_text))

key_val = {}

val_key = {}

for i, key in enumerate(words_set):

    key_val[key] = i

    val_key[i] = key
{k: key_val[k] for k in list(key_val)[:5]}
{k: val_key[k] for k in list(val_key)[:5]}
batch_size = 16

num_timesteps = 32

train_arr = []

valid_arr = []

train_loss = []

valid_loss = []

num_epochs = 10

num_batches = int(len(train_text) / (batch_size * num_timesteps))

valid_num_batches = int(len(valid_text) / (batch_size * num_timesteps))



# Size of embedding

embedding_size = 64



# Number of units in LSTM Layer

num_lstm_units = 64



# Using cuda if present else cpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for word in train_text:

    train_arr.append(key_val[word])

    

# Reshaping data to pass into model

train_data = np.reshape(train_arr[:num_batches * batch_size * num_timesteps], (num_batches * batch_size, -1)) 

target = np.reshape(np.append(train_data[1:], train_data[0]), (num_batches * batch_size, -1))

for word in valid_text:

    valid_arr.append(key_val[word])

validation_features = np.reshape(valid_arr[:valid_num_batches * batch_size * num_timesteps], (valid_num_batches * batch_size, -1))

validation_target = np.reshape(np.append(validation_features[1:], validation_features[0]), (valid_num_batches * batch_size, -1))
# train_data -> shape: num_entries * num_timesteps

train_data.shape
print(train_data[:5])
print(target[:5])
#converting numpy array into torch tensor

train = torch.from_numpy(train_data)

targets = torch.from_numpy(target) 



validation = torch.from_numpy(validation_features)

validation_target = torch.from_numpy(validation_target) 



train_set = torch.utils.data.TensorDataset(train,targets)

valid_set = torch.utils.data.TensorDataset(validation, validation_target)





#Loading data into Dataloader

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 4)

validation_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 4)
#################################################

#             Defining the model                #

#################################################

class Model(nn.Module):

    def __init__(self, num_words, num_timesteps, num_lstm_units, embedding_size):

        super(Model, self).__init__()

        self.num_timesteps = num_timesteps

        self.num_lstm_units = num_lstm_units

        self.embedding = nn.Embedding(num_words, embedding_size)

        self.lstm = nn.LSTM(embedding_size, num_lstm_units, num_layers = 1, batch_first=True)

        self.dense = nn.Linear(num_lstm_units, num_words)

        

    def forward(self, x, hidden_unit):

        embeddings = self.embedding(x)

        prediction, state = self.lstm(embeddings, hidden_unit)

        logits = self.dense(prediction)

        return logits, state
model = Model(len(words_set), num_timesteps,

              embedding_size, num_lstm_units)

model = model.to(device)



# Defining the Loss Function

criterion = nn.CrossEntropyLoss()



# Defining the optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# Function to train the model



def train(epoch):

    #####################

    #  Train the model  #

    #####################

    model.train()

    tr_loss = 0

    # initializing hidden state and cell state with zeros

    h_t, c_t = torch.zeros(1, batch_size, num_lstm_units), torch.zeros(1, batch_size, num_lstm_units)

    h_t.zero_()

    c_t.zero_()

    h_t, c_t = h_t.to(device), c_t.to(device)

    for batch_idx, (X_train, y) in enumerate(train_loader):

        X_train, y = Variable(X_train), Variable(y)

        if torch.cuda.is_available():

            X_train = X_train.to(device)

            y = y.to(device)

            

        # clearning the gradients of all optimized variables

        optimizer.zero_grad()

        

        # forward propogation: computing the outputs predicted by the model

        # The output include the target word values(logits), and the hidden state(h_t) and cell state(c_t).

        logits, (h_t, c_t) = model(X_train, (h_t, c_t))

        # calculating the loss

        loss = criterion(logits.transpose(1, 2), y)

        h_t = h_t.detach()

        c_t = c_t.detach()

        # backpropogation: computing loss with respect to model parameters

        loss.backward()

        # performing a single step of optimization

        optimizer.step()

        tr_loss += loss.item()

    train_loss.append(tr_loss / len(train_loader))

    print('Train Epoch: {} \tTrain Loss: {:.6f}'.format(epoch, (tr_loss / len(train_loader))))
# Function to validate the model



def evaluate(data_loader):

    ################################

    #    Evaluating the Model      #

    ################################

    model.eval()

    h_t, c_t = torch.zeros(1, batch_size, num_lstm_units), torch.zeros(1, batch_size, num_lstm_units)

    h_t.zero_()

    c_t.zero_()

    h_t, c_t = h_t.to(device), c_t.to(device)

    loss = 0

    for data, target in data_loader:

        data, target = Variable(data, volatile=True), Variable(target)

        if torch.cuda.is_available():

            data = data.to(device)

            target = target.to(device)

        logits, (h_t, c_t) = model(data, (h_t, c_t))

        loss += criterion(logits.transpose(1, 2), target).item()

        

    loss /= len(data_loader.dataset)

    valid_loss.append(loss)    

    print('Train Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, loss / len(data_loader)))
# Initializing number of epochs

n_epochs = 10

for epoch in range(n_epochs):

    #train the model

    train(epoch)

    #evaluate the model

    evaluate(validation_loader)
# Function for plotting train and validation loss for each epoch

def plot_graphs(train_loss, valid_loss, epochs):

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20,4))

    ax = fig.add_subplot(1, 2, 1)

    plt.title("Train Loss")

    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('train_loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')

    ax = fig.add_subplot(1, 2, 2)

    plt.title("Validation Loss")

    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='test')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('vaidation _loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')
# plotting train and validation loss

plot_graphs(train_loss, valid_loss, n_epochs)
model.eval()

h_t, c_t = torch.zeros(1, batch_size, num_lstm_units), torch.zeros(1, batch_size, num_lstm_units)

h_t.zero_()

c_t.zero_()

h_t, c_t = h_t.to(device), c_t.to(device) 

words = []

final_predictions = []

#count = 0

##################################################

# Generating predictions for next timestep  #

##################################################

for i, (word, target_variable) in enumerate(validation_loader):

 #   if(count == 3):

 #       break

    word = word.to(device)

    numpy_word = word.cpu().numpy()

    output, (h_t, c_t) = model(word, (h_t, c_t))

    words = numpy_word.ravel()

    predictions = torch.topk(output[0], k=3)[1].tolist()

    print('**********************************************************************************')

    input_string = []

    for word in words[-25:-15]:

        input_string.append(val_key[word])

    s = ' '.join(input_string)

    print('The input string is:\t {}'.format(s))

    string = []

    string.append(val_key[predictions[0][0]])

    string.append(val_key[predictions[0][1]])

    string.append(val_key[predictions[0][2]])

    print('\nTop 3 predictions Timestep i + 1 for the input string are are {}\n'.format(string))

    print('**********************************************************************************')