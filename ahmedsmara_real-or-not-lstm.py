from nltk.corpus import stopwords

from nltk import wordpunct_tokenize

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.naive_bayes import GaussianNB

from torch.autograd import Variable

from string import punctuation

from gensim.models import Word2Vec



import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import collections

import nltk

import torch.nn.functional as F

import matplotlib.pyplot as plt

import re

import string



nltk.download('stopwords')

nltk.download('wordnet')

%matplotlib inline
def clean(text):

    text=text.lower()

    stp=set(stopwords.words("english"))

    placesp = re.compile('[/(){}\[\]\|@,;]')

    removech= re.compile('[^0-9a-z #+_]')

    st=WordNetLemmatizer()

    text=re.sub(placesp,' ',text)

    text=re.sub(removech,' ',text)

    text=text.split()

    text=[w for w in text if not w in stp]

    text=[st.lemmatize(w) for w in text]

    text=" ".join(text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
words=[]

for i in range(train.shape[0]):

    train.at[i,'text']=clean(train.loc[i,'text']).split(' ')

    for j in range(len(train.loc[i,'text'])):

        words.append(train.loc[i,'text'][j])

        

for i in range(test.shape[0]):

    test.at[i,'text']=clean(test.loc[i,'text']).split(' ')

    for j in range(len(test.loc[i,'text'])):

        words.append(test.loc[i,'text'][j])
from collections import Counter



## Build a dictionary that maps words to integers

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



## use the dict to tokenize each review in reviews_split

## store the tokenized reviews in reviews_ints



text_ints=[]

test_x=[]

for review in train.text:

    text_ints.append([vocab_to_int[word] for word in review])

    

for review in test.text:

    test_x.append([vocab_to_int[word] for word in review])

review_lens = Counter([len(x) for x in text_ints])

maxlen=max(review_lens)
def pad_features(reviews_ints, seq_length):

    ''' Return features of review_ints, where each review is padded with 0's 

        or truncated to the input seq_length.

    '''

    

    # getting the correct rows x cols shape

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)



    # for each review, I grab that review and 

    for i, row in enumerate(reviews_ints):

        features[i, -len(row):] = np.array(row)[:seq_length]

    

    return features
seq_length = 30

features = pad_features(text_ints, seq_length=seq_length)

test_x = pad_features(test_x, seq_length=seq_length)
encoded_labels=list(train.target)

split_frac = 0.8



split_idx = int(len(features)*split_frac)

train_x, val_x = features[:split_idx], features[split_idx:]

train_y, val_y = encoded_labels[:split_idx], encoded_labels[split_idx:]





## print out the shapes of your resultant feature data

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(train_x.shape), 

      "\nValidation set: \t{}".format(val_x.shape))
fk=np.zeros(test_x.shape[0])
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_x),  torch.tensor(train_y))

valid_data = TensorDataset(torch.from_numpy(val_x),  torch.tensor(val_y))

test_data = TensorDataset(torch.from_numpy(test_x),  torch.tensor(list(fk)))



# dataloaders

batch_size = 20



# make sure the SHUFFLE your training data

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=1)
train_on_gpu=True
class SentimentRNN(nn.Module):

    """

    The RNN model that will be used to perform Sentiment analysis.

    """



    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        """

        Initialize the model by setting up the layers.

        """

        super(SentimentRNN, self).__init__()



        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        # embedding and LSTM layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 

                            dropout=drop_prob, batch_first=True)

        

        # dropout layer

        self.dropout = nn.Dropout(0.3)

        

        # linear and sigmoid layers

        self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

        



    def forward(self, x, hidden):

        """

        Perform a forward pass of our model on some input and hidden state.

        """

        batch_size = x.size(0)



        # embeddings and lstm_out

        x = x.long()

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

    

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        

        # dropout and fully-connected layer

        out = self.dropout(lstm_out)

        out = self.fc(out)

        # sigmoid function

        sig_out = self.sig(out)

        

        # reshape to be batch_size first

        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        

        # return last sigmoid output and hidden state

        return sig_out, hidden

    

    

    def init_hidden(self, batch_size):

        ''' Initializes hidden state '''

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,

        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden
# Instantiate the model w/ hyperparams

vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens

output_size = 1

embedding_dim = 400

hidden_dim = 256

n_layers = 2



net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)



print(net)


# loss and optimization functions

lr=0.001



criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
mnloss=np.Inf

epochs = 20 # 3-4 is approx where I noticed the validation loss stop decreasing



counter = 0

print_every = 100

clip=5 # gradient clipping



# move model to GPU, if available

if(train_on_gpu):

    net.cuda()



net.train()

# train for some number of epochs

for e in range(epochs):

    # initialize hidden state

    h = net.init_hidden(batch_size)



    # batch loop

    for inputs, labels in train_loader:

        if len(inputs)<batch_size:

            continue

        counter += 1



        if(train_on_gpu):

            inputs, labels = inputs.cuda(), labels.cuda()



        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])



        # zero accumulated gradients

        net.zero_grad()



        # get the output from the model

        output, h = net(inputs, h)



        # calculate the loss and perform backprop

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()



        # loss stats

        if counter % print_every == 0:

            # Get validation loss

            val_h = net.init_hidden(batch_size)

            val_losses = []

            net.eval()

            for inputs, labels in valid_loader:

                if len(inputs)<batch_size:

                    continue

                # Creating new variables for the hidden state, otherwise

                # we'd backprop through the entire training history

                val_h = tuple([each.data for each in val_h])



                if(train_on_gpu):

                    inputs, labels = inputs.cuda(), labels.cuda()



                output, val_h = net(inputs, val_h)

                val_loss = criterion(output.squeeze(), labels.float())



                val_losses.append(val_loss.item())



            net.train()

            if mnloss<np.mean(val_losses):

                mnloss=np.mean(val_losses)

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
final=[]
h = net.init_hidden(1)



net.eval()

for inputs, labels in test_loader:



    h = tuple([each.data for each in h])



    if(train_on_gpu):

        inputs, labels = inputs.cuda(), labels.cuda()

    

    output, h = net(inputs, h)

    pred = torch.round(output.squeeze())

    final.append(pred.item())
pred=[]

for i in range(len(final)):

    pred.append(int (final[i]))
sample_submission['target']=pred

sample_submission.to_csv("submissionfinal.csv", index=False)