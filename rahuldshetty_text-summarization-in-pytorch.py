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
data_path = '/kaggle/input/news-summary/news_summary_more.csv'

glove_path = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'
data = pd.read_csv(data_path,encoding='utf-8')

data.head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",



                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",



                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",



                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",



                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",



                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",



                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",



                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",



                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",



                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",



                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",



                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",



                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",



                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",



                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",



                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",



                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",



                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",



                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",



                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",



                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",



                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",



                           "you're": "you are", "you've": "you have"}
import re

from nltk.corpus import stopwords



stop_words = stopwords.words('english')



def preprocess(text):

    text = text.lower() # lowercase

    text = text.split() # convert have'nt -> have not

    for i in range(len(text)):

        word = text[i]

        if word in contraction_mapping:

            text[i] = contraction_mapping[word]

    text = " ".join(text)

    text = text.split()

    newtext = []

    for word in text:

        if word not in stop_words:

            newtext.append(word)

    text = " ".join(newtext)

    text = text.replace("'s",'') # convert your's -> your

    text = re.sub(r'\(.*\)','',text) # remove (words)

    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations

    text = re.sub(r'\.',' . ',text)

    return text



sample = "(hello) hi there .man tiger caller who's that isn't it ? WALL-E"

print(preprocess(sample))
data['headlines'] = data['headlines'].apply(lambda x:preprocess(x))

data['text'] = data['text'].apply(lambda x:preprocess(x))

print(data['headlines'][20],data['text'][20])
x = data['text']

y = data['headlines']

print(x[50],y[50],sep='\n')
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0

EOS_token = 1





class Lang:

    def __init__(self, name):

        self.name = name

        self.word2index = {}

        self.word2count = {}

        self.index2word = {0: "SOS", 1: "EOS"}

        self.n_words = 2  # Count SOS and EOS



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
def readLangs(text, summary, reverse=False):

    print("Reading lines...")

    

    # Split every line into pairs and normalize

    pairs = [[text[i],summary[i]] for i in range(len(text))]



    # Reverse pairs, make Lang instances

    if reverse:

        pairs = [list(reversed(p)) for p in pairs]

        input_lang = Lang(summary)

        output_lang = Lang(text)

    else:

        input_lang = Lang(text)

        output_lang = Lang(summary)



    return input_lang, output_lang, pairs
def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")

    for pair in pairs:

        input_lang.addSentence(pair[0])

        output_lang.addSentence(pair[1])

    print("Counted words:")

    print(input_lang.name, input_lang.n_words)

    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs
input_lang, output_lang, pairs = prepareData( x, y , False)

print(random.choice(pairs))
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)



    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden



    def initHidden(self):

        return torch.zeros(1, 1, self.hidden_size, device=device)
class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden



    def initHidden(self):

        return torch.zeros(1, 1, self.hidden_size, device=device)
MAX_LENGTH = 90
class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):

        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.dropout_p = dropout_p

        self.max_length = max_length



        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)



    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)

        embedded = self.dropout(embedded)



        attn_weights = F.softmax(

            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),

                                 encoder_outputs.unsqueeze(0))



        output = torch.cat((embedded[0], attn_applied[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)



        output = F.relu(output)

        output, hidden = self.gru(output, hidden)



        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights



    def initHidden(self):

        return torch.zeros(1, 1, self.hidden_size, device=device)
def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] for word in sentence.split(' ')]





def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)





def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(input_lang, pair[0])

    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()



    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()



    input_length = input_tensor.size(0)

    target_length = target_tensor.size(0)



    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)



    loss = 0



    for ei in range(input_length):

        encoder_output, encoder_hidden = encoder(

            input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]



    decoder_input = torch.tensor([[SOS_token]], device=device)



    decoder_hidden = encoder_hidden



    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False



    if use_teacher_forcing:

        # Teacher forcing: Feed the target as the next input

        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(

                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing



    else:

        # Without teacher forcing: use its own predictions as the next input

        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(

                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()  # detach from history as input



            loss += criterion(decoder_output, target_tensor[di])

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

    plt.plot(points)
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    print("Training....")

    start = time.time()

    plot_losses = []

    print_loss_total = 0  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every



    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs))

                      for i in range(n_iters)]

    criterion = nn.NLLLoss()



    for iter in range(1, n_iters + 1):

        if iter% 1000 == 0:

            print(iter,"/",n_iters + 1)

        training_pair = training_pairs[iter - 1]

        input_tensor = training_pair[0]

        target_tensor = training_pair[1]



        loss = train(input_tensor, target_tensor, encoder,

                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

        plot_loss_total += loss



        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every

            print_loss_total = 0

            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),

                                         iter, iter / n_iters * 100, print_loss_avg))



        if iter % plot_every == 0:

            plot_loss_avg = plot_loss_total / plot_every

            plot_losses.append(plot_loss_avg)

            plot_loss_total = 0



    showPlot(plot_losses)
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    with torch.no_grad():

        input_tensor = tensorFromSentence(input_lang, sentence)

        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.initHidden()



        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)



        for ei in range(input_length):

            encoder_output, encoder_hidden = encoder(input_tensor[ei],

                                                     encoder_hidden)

            encoder_outputs[ei] += encoder_output[0, 0]



        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS



        decoder_hidden = encoder_hidden



        decoded_words = []

        decoder_attentions = torch.zeros(max_length, max_length)



        for di in range(max_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(

                decoder_input, decoder_hidden, encoder_outputs)

            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:

                decoded_words.append('<EOS>')

                break

            else:

                decoded_words.append(output_lang.index2word[topi.item()])



            decoder_input = topi.squeeze().detach()



        return decoded_words, decoder_attentions[:di + 1]
def evaluateRandomly(encoder, decoder, n=10):

    for i in range(n):

        pair = random.choice(pairs)

        print('>', pair[0])

        print('=', pair[1])

        output_words, attentions = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words)

        print('<', output_sentence)

        print('')
hidden_size = 300

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)



trainIters(encoder1, attn_decoder1, 175000, print_every=5000)
torch.save(encoder1.state_dict(), './enc.w')

torch.save(attn_decoder1.state_dict(), './att.w')
evaluateRandomly(encoder1, attn_decoder1)