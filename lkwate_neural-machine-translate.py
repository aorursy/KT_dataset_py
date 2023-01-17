from __future__ import unicode_literals, print_function, division

from io import open

import unicodedata

import string

import re

import random



import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

## load device with cuda if it's available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
SOS_token = 0

EOS_token = 1 



class Lang:

    def __init__(self, name):

        self.name = name

        self.word2index = {}

        self.word2count = {}

        self.index2word = {0:"SOS", 1:"EOS"}

        self.n_words = 2 # count SOS and EOS

    

    def addSentence(self, sentence):

        for word in sentence.split(' '): 

            self.addWord(word)

    

    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.n_words

            self.word2count[word] = 1

            self.index2word[self.n_words] = word

            self.n_words += 1

        else:

            self.word2count[word] += 1
#Turn a unicode string to plain ASCII

def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])", r" \1", s)

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s
def readLangs(lang1, lang2, reverse=False):

    print("Reading lines...")

    

    #Read the file and split into lines

    lines = open('/kaggle/input/dataenfr/data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    

    #Split every line into pairs and normalize

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    

    if reverse:

        pairs = [list(reversed(p)) for p in pairs]

        input_lang = Lang(lang2)

        output_lang = Lang(lang1)

    else:

        input_lang = Lang(lang1)

        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
MAX_LENGTH = 10



eng_prefixes = (

    "i am", "i m",

    "he is", "he s", 

    "she is", "she s", 

    "you are", "you re", 

    "we are", "we re", 

    "they are", "they re"

)



def filterPair(p):

    return (len(p[0].split(' '))) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)



def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]
def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    print("Read %s sentence pairs" % len(pairs))

    #filter pairs

    pairs = filterPairs(pairs)

    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")

    for pair in pairs:

        input_lang.addSentence(pair[0])

        output_lang.addSentence(pair[1])

    print("Counted words: ")

    print(input_lang.name, input_lang.n_words)

    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs



input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

print(random.choice(pairs))
class EncoderRNN(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(vocab_size, input_size)

        self.BiLTSM_layer = nn.LSTM(input_size, hidden_size, bidirectional=True)

        self.linearWc = nn.Linear(2 * hidden_size, hidden_size)

        self.linearWh = nn.Linear(2 * hidden_size, hidden_size)

    

    def forward(self, sequence):

        '''

            this function return a tuple of (output, (h_n, c_n))

            output : all the output of recurrent model shape (seq_length, batch, num_layer * num_direction)

            h_n : last hidden state shape (num_layer * num_direction, batch, hidden_size)

            c_n : last cell state, it has the same shape of h_n

        '''

        # we assume that sequence is list of integers which are indices of words in vocabulary

        

        embedded = self.embedding_layer(sequence)

        h_0 = torch.zeros((2, 1, self.hidden_size)).to(device)

        c_0 = torch.zeros((2, 1, self.hidden_size)).to(device)

        output, (h_n, c_n) = self.BiLTSM_layer(embedded, (h_0, c_0))

        output = output.squeeze(1).to(device) # shape (seq_length, 2 * hidden_size)

        h_n = self.linearWh(output[output.shape[0] - 1]).reshape(1, 1, h_n.shape[2]).to(device)

        c_n = self.linearWc(torch.cat((c_n[1], c_n[0]), 1)).reshape(1, 1, c_n.shape[2]).to(device)

        return output, (h_n, c_n)

    

    def init_weights(self):

        return torch.zeros(2, 1, self.hiddent_size)
class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, vocab_size, dropout_p = 0.1):

        super(AttnDecoderRNN, self).__init__()

        '''

            hidden_size : h size of features of embedded

            vocab_size : size of vocabulary

        '''

        # parameters

        self.hidden_size = hidden_size 

        self.vocab_size = vocab_size

        self.dropout_p = dropout_p

        

        # layers 

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.lstm = nn.LSTM(2 * self.hidden_size, self.hidden_size)

        self.linearWeightAttn = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.linearU = nn.Linear(self.hidden_size * 3, self.hidden_size)

        self.linearVocab = nn.Linear(self.hidden_size, self.vocab_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.softmaxAtt = nn.Softmax(dim=0)

        self.softmaxVocab = nn.LogSoftmax(dim=0)

        

    def forward(self, y_t, hidden, cell, o_t, encoder_outputs):

        '''

            assume that : 

                y_t : torch.LongTensor([index_word])

                o_t : torch.Tensor() shape : (hiddens_size)

                encoder_outputs : shape (seq_length, hidden_size * 2)

                return  : p_t, o_t, hidden, cell

        '''

        embedded = self.embedding(y_t).squeeze().to(device)

        o_t = o_t.squeeze().to(device)

        y_hat_t = torch.cat((embedded, o_t), 0).to(device)

        y_hat_t = y_hat_t.reshape(1, 1, y_hat_t.shape[0]).to(device)

        _, (hidden, cell) = self.lstm(y_hat_t, (hidden, cell))

        e_score = self.linearWeightAttn(encoder_outputs) # shape of alpha : (seq_length, hidden_size)

        hidden_t = hidden.reshape(self.hidden_size, 1)

        e_score = torch.mm(e_score, hidden_t) # shape of alpha : (seq_length, 1)

        e_score = e_score.squeeze(1) # shape :(seq_length)

        alpha = self.softmaxAtt(e_score)

        a_t = torch.mm(encoder_outputs.t(), alpha.unsqueeze(1)) # shape : (2 * hidden_size, 1)

        u_t = torch.cat((a_t, hidden_t), 0) # shape : (3 * hidden_size, 1)

        v_t = self.linearU(u_t.t()) # shape : (1, hidden_size)

        o_t = self.dropout(v_t.tanh()) # shape : (1, hidden_size)

        p_t = self.softmaxVocab(self.linearVocab(o_t).squeeze(0)).unsqueeze(0)

        

        return p_t, o_t, hidden, cell
def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] for word in sentence.split(' ')]



def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype = torch.long, device = device).view(-1, 1)



def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(input_lang, pair[0])

    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)
hidden_size = 256
def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):



    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()



    input_length = len(input)

    target_length = len(target)



    loss = 0

    

    ## pass through encoder

    encoder_outputs, (hn, cn) = encoder(input)

    

    ## compute h_dec, c_dec

    

    decoder_hidden = hn

    decoder_cell = cn

    decoder_input = torch.tensor([[SOS_token]], device=device)

    output_combined = torch.zeros(hidden_size).to(device)



    # use teacher forcing method   

    for di in range(target_length):

        prob_softmax, output_combined, decoder_hidden, decoder_cell = decoder(

                decoder_input, decoder_hidden, decoder_cell, output_combined, encoder_outputs)

        decoder_input = target[di]

        loss += criterion(prob_softmax, target[di])

        if decoder_input.item() == EOS_token:

            break



    loss.backward()



    encoder_optimizer.step()

    decoder_optimizer.step()



    return loss.item() / target_length
import time

import math



def asMinutes(s):

    m = math.floor(s / 60)

    s -= m * 60 

    return '%dm %ds' % (m, s)



def timeSince(since, percent):

    now = time.time()

    s = now - since

    es = s / (percent)

    rs = es - s

    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def trainIters(encoder, decoder, encoder_optimzer, decoder_optimizer, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()

    plot_losses = [] 

    print_loss_total = 0

    plot_loss_total = 0



    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    

    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]

        input_tensor = training_pair[0]

        target_tensor = training_pair[1]

        

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimzer, decoder_optimizer, criterion)

        print_loss_total += loss

        plot_loss_total += loss

        

        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every

            print_loss_total = 0 

            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        

        if iter % plot_every == 0: 

            plot_loss_avg = plot_loss_total / plot_every

            plot_losses.append(plot_loss_avg)

            plot_loss_total = 0

    showPlot(plot_losses)
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import matplotlib.ticker as ticker

import numpy as np





def showPlot(points):

    plt.figure()

    fig, ax = plt.subplots()

    # this locator puts ticks at regular intervals

    loc = ticker.MultipleLocator(base=0.2)

    ax.yaxis.set_major_locator(loc)

    plt.plot
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    with torch.no_grad():

        input_tensor = tensorFromSentence(input_lang, sentence)

        input_length = input_tensor.size()[0]



        encoder_outputs, (decoder_hidden, decoder_cell) = encoder(input_tensor)

        

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS



        ## compute h_dec, c_dec

        decoder_input = torch.tensor([[SOS_token]], device=device)

        output_combined = torch.zeros(hidden_size)





        decoded_words = []

        

        for di in range(max_length):

            prob_softmax, output_combined, decoder_hidden, decoder_cell = decoder(

                decoder_input, decoder_hidden, decoder_cell, output_combined, encoder_outputs)

            topi = torch.argmax(prob_softmax)

            if topi.item() == EOS_token:

                decoded_words.append('<EOS>')

                break

            else:

                decoded_words.append(output_lang.index2word[topi.item()])



            decoder_input = topi.squeeze().detach()



        return decoded_words
def evaluateRandomly(encoder, decoder, n=10):

    for i in range(n):

        pair = random.choice(pairs)

        print('>', pair[0])

        print('=', pair[1])

        output_words = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words)

        print('<', output_sentence)

        print('')
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, hidden_size).to(device)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)



# optimizer

learning_rate = 0.01

encoder_optimzer = optim.SGD(encoder1.parameters(), lr = learning_rate)

decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr = learning_rate)



# train and evaluate

trainIters(encoder1, attn_decoder1, encoder_optimzer, decoder_optimizer, 100000, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1)
pathEncoder = 'checkpointEncoderRNN.pth'

pathDecoder = 'checkpointDecoderRNN.pth'

checkpointEncoder = {

    'model' : encoder1, 

    'state_dict' : encoder1.state_dict()

}

checkpointDecoder = {

    'model' : attn_decoder1, 

    'state_dict' : attn_decoder1.state_dict()

}



#save

torch.save(checkpointEncoder, pathEncoder)

torch.save(checkpointDecoder, pathDecoder)