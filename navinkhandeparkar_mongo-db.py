# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import torch

import nltk

from torch.utils import data as data2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data, test_data = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')

train_data_len = train_data.shape[0]

print('train data length: {}'.format(train_data_len)) # 1306122

train_data.head()
print(f'Number of training examples: {len(train_data)}')

print(f'Number of testing examples: {len(test_data)}')

# statistis of target 0 and 1

t0, t1 = len(train_data[train_data.sentiment == 0]), len(train_data[train_data.sentiment == 1])

t0_pct, t1_pct = t0 / train_data_len * 100, t1 / train_data_len * 100

print('target 0 vs 1 = {} vs {}, {:.2f}% vs {:.2f}%'.format(t0, t1, t0_pct, t1_pct))
from collections import Counter



def process_text(type_data, seq_len, phase):

    all_text2 = ' '.join(type_data.review)

    # create a list of words

    words = all_text2.split()

    # Count all the words using Counter Method

    count_words = Counter(words)



    total_words = len(words)

    sorted_words = count_words.most_common(total_words)

    

    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    reviews_int = []

    for review in type_data.review:

        r = [vocab_to_int[w] for w in review.split()]

        reviews_int.append(r)

    reviews_len = [len(x) for x in reviews_int]

    if phase == 'train':

        encoded_labels = np.array(type_data.sentiment)

        encoded_labels = np.array([ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ])

    else:

        encoded_labels = np.array([ i for i in type_data.Id])

    reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]

    

    features = pad_features(reviews_int, seq_len)

    return reviews_int, encoded_labels, features, vocab_to_int

    
# import pandas as pd

# import matplotlib.pyplot as plt

# %matplotlib inline



# pd.Series(reviews_len).hist()

# plt.show()

# pd.Series(reviews_len).describe()
def pad_features(reviews_int, seq_length):

    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.

    '''

    features = np.zeros((len(reviews_int), seq_length), dtype = int)

    

    for i, review in enumerate(reviews_int):

        review_len = len(review)

        

        if review_len <= seq_length:

            zeroes = list(np.zeros(seq_length-review_len))

            new = zeroes+review

        elif review_len > seq_length:

            new = review[0:seq_length]

        

        features[i,:] = np.array(new)

    

    return features
import torch

from torch.utils.data import DataLoader, TensorDataset



def train_data_split(features, encoded_labels, phase, qids=None):

    if phase == 'train':

        split_frac = 0.8

        len_feat = len(features)

        train_x = features[0:int(split_frac*len_feat)]

        train_y = encoded_labels[0:int(split_frac*len_feat)]

        remaining_x = features[int(split_frac*len_feat):]

        remaining_y = encoded_labels[int(split_frac*len_feat):]

        valid_x = remaining_x[0:int(len(remaining_x)*0.5)]

        valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

        test_x = remaining_x[int(len(remaining_x)*0.5):]

        test_y = remaining_y[int(len(remaining_y)*0.5):]



        # create Tensor datasets

        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

        valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # dataloaders

        batch_size = 50

        # make sure to SHUFFLE your data

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    else:

        split_frac = 1

        len_feat = len(features)

        train_x = features[0:int(split_frac*len_feat)]

        qids = np.array([i for i in qids])

        # create Tensor datasets

        test_data = TensorDataset(torch.from_numpy(qids), torch.from_numpy(train_x) )

        # dataloaders

        batch_size = 50

        # make sure to SHUFFLE your data

        train_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        valid_loader = None

        test_loader = None

    

    return train_loader, valid_loader, test_loader


import torch.nn as nn



class SentimentLSTM(nn.Module):

    """

    The RNN model that will be used to perform Sentiment analysis.

    """



    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        """

        Initialize the model by setting up the layers.

        """

        super().__init__()



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

        

        if (True):

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden
def training(train_loader, valid_loader, test_loader, net):

    # loss and optimization functions

    lr=0.001

    batch_size = 50

    train_on_gpu =True



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

            inputs = inputs.type(torch.LongTensor)

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

                    inputs = inputs.type(torch.LongTensor)



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

                

    # Get test data loss and accuracy



    test_losses = [] # track loss

    num_correct = 0



    # init hidden state

    h = net.init_hidden(batch_size)



    net.eval()

    # iterate over test data

    for inputs, labels in test_loader:



        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])

        inputs = inputs.type(torch.LongTensor)



        if(train_on_gpu):

            inputs, labels = inputs.cuda(), labels.cuda()



        # get predicted outputs

        output, h = net(inputs, h)



        # calculate loss

        test_loss = criterion(output.squeeze(), labels.float())

        test_losses.append(test_loss.item())



        # convert output probabilities to predicted class (0 or 1)

        pred = torch.round(output.squeeze())  # rounds to the nearest integer



        # compare predictions to true label

        correct_tensor = pred.eq(labels.float().view_as(pred))

        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        num_correct += np.sum(correct)





    # -- stats! -- ##

    # avg test loss

    print("Test loss: {:.3f}".format(np.mean(test_losses)))



    # accuracy over all test data

    test_acc = num_correct/len(test_loader.dataset)

    print("Test accuracy: {:.3f}".format(test_acc))
def test(net, test_loader, batch_size=50):

    test_l_h = net.init_hidden(batch_size)

    ret_qid = []

    ret_pred = []

    test_len = len(test_loader)

    counter = 0

    with torch.no_grad():

        for qids, inputs in test_loader:

            inputs = inputs.type(torch.LongTensor)

            counter += 1

            inputs = inputs.cuda()

            

            # for LSTM

            test_l_h = tuple([each.data for each in test_l_h])

            # for GRU

#             test_g_h = test_g_h.data



            outputs, test_l_h = net(inputs, test_l_h)

            

            ret_qid.append(qids)

            ret_pred.append(torch.round(outputs.squeeze()).cpu().numpy().astype(int))

            

            if counter % 300 == 0:

                print('{}/{} done'.format(counter, test_len))



    return ret_qid, ret_pred

    
reviews_int, encoded_labels, features, vocab_to_int = process_text(train_data, 200, 'train');

# Instantiate the model w/ hyperparams

vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding

output_size = 1

embedding_dim = 400

hidden_dim = 256

n_layers = 2

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
train_loader, valid_loader, test_loader = train_data_split(features, encoded_labels, 'train')
training(train_loader, valid_loader, test_loader, net)
reviews_int, encoded_labels, features, a = process_text(test_data, 200, 'test');

test_loader, valid_loader, a = train_data_split(features, encoded_labels, 'test', test_data.Id)
ret_qid, ret_pred = test(net, test_loader, batch_size=50)
ret_qid, ret_pred = np.concatenate(ret_qid), np.concatenate(ret_pred)

submit_df = pd.DataFrame({"Id": ret_qid, "sentiment": ret_pred})
submit_df.to_csv('submission.csv', index=False)