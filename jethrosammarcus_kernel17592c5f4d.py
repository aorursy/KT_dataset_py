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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data_sub=pd.read_csv('../input/test.csv')

train_data=pd.read_csv('../input/train.csv')

reviews=train_data['text'].get_values()

labels=train_data['user'].get_values()

input_test=test_data_sub['text'].get_values()

y_test=list()
from string import punctuation



# get rid of punctuation

reviews = reviews.lower() # lowercase, standardize

all_text = ''.join([c for c in reviews if c not in punctuation])



# split by new lines and spaces

reviews_split = all_text.split('\n')

all_text = ' '.join(reviews_split)



# create a list of words

words = all_text.split()
from collections import Counter



## Build a dictionary that maps words to integers

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



## use the dict to tokenize each review in reviews_split

## store the tokenized reviews in reviews_ints

reviews_ints = []

for review in reviews_split:

    reviews_ints.append([vocab_to_int[word] for word in review.split()])
# 1=positive, 0=negative label conversion

labels_split = labels.split('\n')

encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])



# outlier review stats

review_lens = Counter([len(x) for x in reviews_ints])

print("Zero-length reviews: {}".format(review_lens[0]))

print("Maximum review length: {}".format(max(review_lens)))
print('Number of reviews before removing outliers: ', len(reviews_ints))



## remove any reviews/labels with zero length from the reviews_ints list.



# get indices of any reviews with length 0

non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]



# remove 0-length reviews and their labels

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]

encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])



print('Number of reviews after removing outliers: ', len(reviews_ints))
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
split_frac = 0.8



## split data into training, validation, and test data (features and labels, x and y)



split_idx = int(len(features)*split_frac)

train_x, remaining_x = features[:split_idx], features[split_idx:]

train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]



test_idx = int(len(remaining_x)*0.5)

val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]

val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]



## print out the shapes of your resultant feature data

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(train_x.shape), 

      "\nValidation set: \t{}".format(val_x.shape),

      "\nTest set: \t\t{}".format(test_x.shape))
import torch

from torch.utils.data import TensorDataset, DataLoader



# create Tensor datasets

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))



# dataloaders

batch_size = 50



# make sure the SHUFFLE your training data

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
# obtain one batch of training data

dataiter = iter(train_loader)

sample_x, sample_y = dataiter.next()



print('Sample input size: ', sample_x.size()) # batch_size, seq_length

print('Sample input: \n', sample_x)

print()

print('Sample label size: ', sample_y.size()) # batch_size

print('Sample label: \n', sample_y)
# First checking if GPU is available

train_on_gpu=torch.cuda.is_available()



if(train_on_gpu):

    print('Training on GPU.')

else:

    print('No GPU available, training on CPU.')
import torch.nn as nn



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
# training params



epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing



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



                # Creating new variables for the hidden state, otherwise

                # we'd backprop through the entire training history

                val_h = tuple([each.data for each in val_h])



                if(train_on_gpu):

                    inputs, labels = inputs.cuda(), labels.cuda()



                output, val_h = net(inputs, val_h)

                val_loss = criterion(output.squeeze(), labels.float())



                val_losses.append(val_loss.item())



            net.train()

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))