# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.autograd import Variable

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import torch.utils.data

import matplotlib.ticker as ticker

from random import randint

import tensorflow as tf 
def load_data(path):

    df = pd.read_csv(path, header=None)

    X = df[0].values

    y = df[1].values

    x_tok = Tokenizer(char_level=True, filters='')

    x_tok.fit_on_texts(X)

    y_tok = Tokenizer(char_level=True, filters='')

    y_tok.fit_on_texts(y)

    

    X = x_tok.texts_to_sequences(X)

    y = y_tok.texts_to_sequences(y)

    

    X = pad_sequences(X)

    y = np.asarray(y)

    

    return X, y, x_tok.word_index, y_tok.word_index



X, y, x_wid, y_wid= load_data('../input/data.csv')

x_id2w = dict(zip(x_wid.values(), x_wid.keys()))

y_id2w = dict(zip(y_wid.values(), y_wid.keys()))

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('train size: {} - test size: {}'.format(len(X_train), len(X_test)))
# hidden size for one LSTM model

hidden_size = 128

learning_rate = 0.001

decoder_learning_ratio = 0.1

# Vocabulary Set of input sentences ( + 1 because you need padding characters)

input_size = len(x_wid) + 1



input_size = len(x_wid) + 1



# +2  because you need start character and end character

output_size = len(y_wid) + 2

# 2 characters located at the end

sos_idx = len(y_wid) 

eos_idx = len(y_wid) + 1



max_length = y.shape[1]

print("input vocab: {} - output vocab: {} - length of target: {}".format(input_size, output_size, max_length))
def decoder_sentence(idxs, vocab):

    text = ''.join([vocab[w] for w in idxs if (w > 0) and (w in vocab)])

    return text
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        # embedding vector of word

        self.embedding = nn.Embedding(input_size, hidden_size)

        # GRU Model variant of RNN learns vector representation of sentence

        self.gru = nn.GRU(hidden_size, hidden_size)

        

    def forward(self, input):

        # Input: S x B

        embedded = self.embedding(input)

        output, hidden = self.gru(embedded)

        return output, hidden # S x B x H, 1 x B x H
class Attention(nn.Module):

    def __init__(self, hidden_size):

        super(Attention, self).__init__()

    

    def forward(self, hidden, encoder_outputs):

        """ The model receives the current hidden state of the decoder model and hidden states of Encoder Model

        Encoder_Output: T x B x H

        Hidden: S x B x H """

        # Tranpose about the correct shape to receive the matrix

        encoder_outputs = torch.transpose(encoder_outputs, 0, 1) # B x T x H

        hidden = torch.transpose(torch.transpose(hidden, 0, 1), 1, 2) # B x H x S

        # Calculating e, is the hidden interaction and the hidden state of the encoder model

        energies = torch.bmm(encoder_outputs, hidden) # B x Tx S

        energies = torch.transpose(energies, 1, 2) # B x S x T

        # Calculating alpha, is the weight of the weighted average should be calculated by the softmax function

        attn_weights = F.softmax(energies, dim = -1) # B x S x T

        

        # Calculating context vector by the weighted average

        output = torch.bmm(attn_weights, encoder_outputs) # B x S x H

        # Returns the necessary dimension

        output = torch.transpose(output, 0, 1) # S x B x H

        attn_weights = torch.transpose(attn_weights, 0, 1) # S x B x T

        # Return context vector and alpha weights for demonstration purposes Attention

        return output, attn_weights
class Decoder(nn.Module):

    def __init__(self, output_size, hidden_size, dropout):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.output_size = output_size

        

        # vector representation for words of output

        self.embedding = nn.Embedding(output_size, hidden_size)

        # define attention model

        self.attn = Attention(hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Decoder: GRU Model

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        # Predicting the words at each moment, we join the two hidden and context together

        self.concat = nn.Linear(self.hidden_size * 2, hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    

    def forward(self, input, hidden, encoder_outputs):

        # input: S x B

        # encoder_outputs: B x S x H

        # hidden: 1 x B x H

        embedded = self.embedding(input) # 1 x B x H

        embedded = self.dropout(embedded)

        # Represention of Sentence

        rnn_output, hidden = self.gru(embedded, hidden) # S x B x H, 1 x B x H

        # Calculating context vector base on hidden states

        context, attn_weights = self.attn(rnn_output, encoder_outputs) # S x B x H

        # Concat hidden state of decoder model current and context vector for predict

        concat_input = torch.cat((rnn_output, context), -1)

        concat_output = torch.tanh(self.concat(concat_input))  #SxBxH

        

        output = self.out(concat_output) # S x B x output_size

        return output, hidden, attn_weights
encoder = Encoder(input_size, hidden_size)

decoder = Decoder(output_size, hidden_size, 0.1)



# Initialize optimizers and criterion

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)

decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate * decoder_learning_ratio)

criterion = nn.CrossEntropyLoss()





input_encoder = torch.randint(1, input_size, (34, 6), dtype = torch.long)

encoder_outputs, hidden = encoder(input_encoder)

input_decoder = torch.randint(1, output_size, (10, 6), dtype = torch.long)

output, hidden, attn_weights = decoder(input_decoder, hidden, encoder_outputs)
def forward_and_compute_loss(inputs, targets, encoder, decoder, criterion):

    batch_size = inputs.size()[1]

    

    # define two start and end character

    sos = Variable(torch.ones((1, batch_size), dtype=torch.long)*sos_idx)

    eos = Variable(torch.ones((1, batch_size), dtype=torch.long)*eos_idx)

    

    # input of decoder model need add start character

    decoder_inputs = torch.cat((sos, targets), dim=0)

    # output predict of decoder model need add end character

    decoder_targets = torch.cat((targets, eos), dim=0)

    

    # forward calculating hidden states of sentence

    encoder_outputs, encoder_hidden = encoder(inputs)

    # Calculating output of decoder model

    output, hidden, attn_weights = decoder(decoder_inputs, encoder_hidden, encoder_outputs)

    

    output = torch.transpose(torch.transpose(output, 0, 1), 1, 2) # BxCxS

    decoder_targets = torch.transpose(decoder_targets, 0, 1)

    # Calculating loss 

    loss = criterion(output, decoder_targets)

    

    return loss, output



def train(inputs, targets,  encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    

    encoder.train()

    decoder.train()

    

    # zero gradient, must do every time update gradient

    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()

    

    # Predict of each time the position has the largest prob

    train_loss, output = forward_and_compute_loss(inputs, targets,encoder, decoder,criterion)    

    

    train_loss.backward()

    # Updating one step

    encoder_optimizer.step()

    decoder_optimizer.step()

    

    # Return Loss for Print

    return train_loss.item()



def evaluate(inputs, targets, encoder, decoder, criterion):

    

    encoder.eval()

    decoder.eval()

    # Calculating loss

    eval_loss, output = forward_and_compute_loss(inputs, targets, encoder, decoder,criterion)

    output = torch.transpose(output, 1, 2)

    

    # Predict of each time the position has the largest prob

    pred_idx = torch.argmax(output, dim=-1).squeeze(-1)

    pred_idx = pred_idx.data.cpu().numpy()

    

    # Return Loss and prediction result

    return eval_loss.item(), pred_idx



def predict(inputs, encoder, decoder, target_length=max_length):

    # When predicting we need to calculate the results immediately at each time

    # then stop the word predicted to be calculating the next word     

    batch_size = inputs.size()[1]

    

    # The first input of the decoder model is the starting character, we predict the next character, 

    # then use this character to predict the next word.

    decoder_inputs = Variable(torch.ones((1, batch_size), dtype=torch.long)*sos_idx)

    

    # Calculating hidden state of encoder model, 

    # As the vector of words, we need to calculate the vector context based on these hidden states

    encoder_outputs, encoder_hidden = encoder(inputs)

    hidden = encoder_hidden

    

    preds = []

    attn_weights = []

    

    # Calculating Every word at every moment

    for i in range(target_length):

        # Predict the first word 

        output, hidden, attn_weight = decoder(decoder_inputs, hidden, encoder_outputs)

        output = output.squeeze(dim=0)

        pred_idx = torch.argmax(output, dim=-1)

        

        # Change the next input with the word that has been predicted

        decoder_inputs = Variable(torch.ones((1, batch_size), dtype=torch.long)*pred_idx)

        preds.append(decoder_inputs)

        attn_weights.append(attn_weight.detach())

    

    preds = torch.cat(preds, dim=0)

    preds = torch.transpose(preds, 0, 1)

    attn_weights = torch.cat(attn_weights, dim=0)

    attn_weights = torch.transpose(attn_weights, 0, 1)

    return preds, attn_weights
epochs = 20

batch_size = 64



encoder = Encoder(input_size, hidden_size)

decoder = Decoder(output_size, hidden_size, 0.1)



# Initialize optimizers and criterion

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

criterion = nn.CrossEntropyLoss()



X_val = torch.tensor(X_test, dtype=torch.long)

y_val = torch.tensor(y_test, dtype=torch.long)

X_val = torch.transpose(X_val, 0, 1)

y_val = torch.transpose(y_val, 0, 1)



for epoch in range(epochs):

    for idx in range(len(X_train)//batch_size):

        # input đầu vào của chúng ta là timestep first nhé. 

        X_train_batch = torch.tensor(X_train[batch_size*idx:batch_size*(idx+1)], dtype=torch.long)

        y_train_batch = torch.tensor(y_train[batch_size*idx:batch_size*(idx+1)], dtype=torch.long)

        

        X_train_batch = torch.transpose(X_train_batch, 0, 1)

        y_train_batch = torch.transpose(y_train_batch, 0, 1)

        train_loss= train(X_train_batch, y_train_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    eval_loss, preds = evaluate(X_val, y_val, encoder, decoder, criterion)

    

    print('Epoch {} - train loss: {:.3f} - eval loss: {:.3f}'.format(epoch, train_loss, eval_loss))

    print_idx = np.random.randint(0, len(preds), 3)

    for i in print_idx:

        x_val = decoder_sentence(X_val[:,i].numpy(), x_id2w)

        y_pred = decoder_sentence(preds[i], y_id2w)

        print(" {:<35s}\t{:>10}".format(x_val, y_pred))
preds, attn_weights = predict(X_val ,encoder, decoder, target_length = 10)
def show_attention(input_sentence, output_words, attentions):

    # Set up figure with colorbar

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions.numpy(), cmap='bone')

    fig.colorbar(cax)



    # Set up axes

    ax.set_xticks(np.arange(len(input_sentence)))

    ax.set_xticklabels(list(input_sentence), rotation=90)

    ax.set_yticks(np.arange(len(output_words)))

    ax.set_yticklabels(list(output_words))

    ax.grid()

    ax.set_xlabel('Input Sequence')

    ax.set_ylabel('Output Sequence')

    plt.show()
show_idx = randint(0, len(preds))

text_x = decoder_sentence(X_val[:,show_idx].numpy(), x_id2w)

text_y = decoder_sentence(preds[show_idx].numpy(), y_id2w)

attn_weight = attn_weights[show_idx, :, -len(text_x):]

show_attention(text_x, text_y, attn_weight)