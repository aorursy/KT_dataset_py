%matplotlib inline

from __future__ import unicode_literals, print_function, division

from io import open

import unicodedata

import string

import re

import numpy as np

import pandas as pd

import random

from collections import Counter

from decimal import Decimal

from fastai.text import *



import torch

import torch.nn as nn

from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



pd.set_option('display.max_colwidth', -1)
# Turn a Unicode string to plain ASCII, thanks to

# http://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

    )



# Lowercase, trim





def normalizeString(s,remove_brackets=False):

    #s = unicodeToAscii(s.lower(). v())

    s = unicodeToAscii(s.strip())

    s = re.sub(r"([\.!\?,])", r" \1", s)

    s = re.sub(r"\-", r" ", s)

    if remove_brackets:

        s = re.sub(r"[\]]"," . ",s)

        s = re.sub(r"[\[]"," ",s)

    s = re.sub(r"[^a-zA-Z.!?0-9_\[\]]+", r" ", s)

    s = re.sub(r"  +", r" ", s)

    s = re.sub(r"^ +",r"",s)

    s = re.sub(r" +$",r"",s)

    s = s.strip()

    return s
path = Path('../input')
df_train = pd.read_csv(path/"trainset.csv")

df_test = pd.read_csv(path/"testset_w_refs.csv")

df_dev = pd.read_csv(path/"devset.csv")

print(df_train.shape)

print(df_dev.shape)

print(df_test.shape)

df_train.head()
import unicodedata

def strip_accents(s):

   return ''.join(c for c in unicodedata.normalize('NFD', s)

                  if unicodedata.category(c) != 'Mn')
def delexicalize(attribute,value,new_value,new_row,row):

    new_row["ref"] = re.sub(value,new_value,new_row["ref"])

    new_row["ref"] = re.sub(value.lower(),new_value.lower(),new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value.lower()),new_value.lower(),new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value),new_value,new_row["ref"])

    value0=value[0]+value[1:].lower()

    new_row["ref"] = re.sub(value0,new_value,new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value0),new_value,new_row["ref"])

    value0=value[0].lower()+value[1:]

    new_row["ref"] = re.sub(value0,new_value,new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value0),new_value,new_row["ref"])

    return new_row
def process_features(df):

    rows = []

    for i,row in df.iterrows():

        row0 = row.to_dict()

        row0["original_ref"]=row0["ref"]

        row0["original_mr"]=row0["mr"]

        row0["ref"] = re.sub("  +"," ",row0["ref"])

        row0["mr"] = re.sub("  +"," ",row0["mr"])

        name = re.sub(r"^.*name\[([^\]]+)\].*$",r"\1",row0["mr"].strip())

        near = re.sub(r"^.*near\[([^\]]+)\].*$",r"\1",row0["mr"].strip())

        name = re.sub("  +"," ",name)

        near = re.sub("  +"," ",near)

        row0 = delexicalize("name",name,"Xxx",row0,row)

        row0 = delexicalize("near",near,"Yyy",row0,row)

        row0["mr"] = re.sub(r"name\[[^\]]+\](, *| *$)",r"name[Xxx]\1",row0["mr"].strip())

        row0["mr"] = re.sub(r"near\[[^\]]+\](, *| *$)",r"near[Yyy]\1",row0["mr"].strip())

        row0["mr"] = re.sub(r", *$","",row0["mr"].strip())

        row0["mr"] = re.sub(r" *, *",",",row0["mr"].strip())

        row0["mr"] = row0["mr"].strip()

        if row["ref"]==row0["ref"]:

            continue

        row0["mr"] = re.sub(",",", ",row0["mr"])

        row0["mr"] = re.sub("  +"," ",row0["mr"])

        row0["mr"] = re.sub(" +$","",row0["mr"])

        rows.append(row0)

    return pd.DataFrame(rows)
df_train=process_features(df_train)

df_dev=process_features(df_dev)

df_test=process_features(df_test)

df_train.shape,df_dev.shape,df_test.shape
df_train.sample(10)
rows=[]

for i,row in df_train.iterrows():

    row0=row

    values = row["mr"].split(", ")

    row0["num_mrs"]=len(values)

    rows.append(row0)



df_train = pd.DataFrame(rows)
df_train = df_train.sort_values(by=["num_mrs","original_mr"],ascending=True)

df_train.head(5)
SOS_token = 0

EOS_token = 1





class Rep:

    def __init__(self, name):

        self.name = name

        self.word2index = {}

        self.word2count = {}

        self.index2word = {0: "SOS", 1: "EOS"}

        self.n_words = 2  # Count SOS and EOS



    def addSentence(self, sentence,sep):

        for word in sentence.split(sep):

            self.addWord(word.strip())



    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.n_words

            self.word2count[word] = 1

            self.index2word[self.n_words] = word

            self.n_words += 1

        else:

            self.word2count[word] += 1
def readTexts(df,col_input, col_output):

    print("Reading lines...")

    input_texts = list(df[col_input])

    output_texts = TextDataBunch.from_df(text_cols=col_output,path=".",train_df=df,valid_df=df)

    pairs = []



    it2 = iter(output_texts.train_dl.x)

    for i in range(len(output_texts.train_dl.x)):

        input_text = input_texts[i].strip()

        output_text = ((next(it2)).text).strip()

        pairs.append([input_text,output_text])





    input_rep = Rep(col_input)

    output_rep = Rep(col_output)



    return input_rep, output_rep, pairs
def prepareData(df,col_input, col_output):

    input_text, output_text, pairs = readTexts(df,col_input, col_output)

    print("Read %s text pairs" % len(pairs))

    print("Counting words...")

    for pair in pairs:

        input_text.addSentence(pair[0],",")

        output_text.addSentence(pair[1]," ")

    print("Counted words:")

    print(input_text.name, input_text.n_words)

    print(output_text.name, output_text.n_words)

    return input_text, output_text, pairs





input_rep, output_rep, pairs = prepareData(df_train,'mr', 'ref')
input_rep.index2word
sorted(output_rep.word2count.items(), key=operator.itemgetter(1,0), reverse=True)
for i in range(3):

    p = pairs[i]

    print(p[0])

    print(p[1])

    print("\n")
max_length=0

for r in pairs:

    l = len(r[1].split(" "))

    if l>max_length:

        max_length=l

MAX_LENGTH=max_length+1

MAX_LENGTH
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)    

        self.bidirectional = True

        self.num_layers = 2

        self.gru = nn.GRU(hidden_size, hidden_size,self.num_layers,bidirectional=self.bidirectional)



    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded

        output, hidden = self.gru(output, hidden)

        output = output[0,:,self.hidden_size:]+output[-1,:,:self.hidden_size] # we sum

        return output, hidden



    def initHidden(self):

        return torch.zeros(self.num_layers*(2 if self.bidirectional else 1), 1, self.hidden_size, device=device)
class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size,dropout_p=0.1,max_length=MAX_LENGTH):

        super(AttnDecoderRNN, self).__init__()

        

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.dropout_p = dropout_p

        self.max_length = max_length

        self.bidirectional = True

        self.num_layers = 2

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)      

        

        # Layer for attentional weights

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        

        self.dropout = nn.Dropout(self.dropout_p)

        

        # gru for decoder

        self.gru = nn.GRU(self.hidden_size, self.hidden_size,self.num_layers,bidirectional=self.bidirectional)

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

        output = output[0,:,self.hidden_size:]+output[-1, :,:self.hidden_size] # we sum

        output = F.log_softmax(self.out(output), dim=1)

        return output, hidden, attn_weights



    def initHidden(self):

        return torch.zeros(self.num_layers*(2 if self.bidirectional else 1), 1, self.hidden_size, device=device)
def indexesFromSentence(lang, sentence,sep):

    return [lang.word2index[word.strip()] for word in sentence.split(sep)]





def tensorFromSentence(lang, sentence,sep):

    indexes = indexesFromSentence(lang, sentence,sep)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)





def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(input_rep, pair[0],",")

    target_tensor = tensorFromSentence(output_rep, pair[1]," ")

    return (input_tensor, target_tensor)
teacher_forcing_ratio = 0.7





def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()



    optimizer.zero_grad()



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



    optimizer.step()



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
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01,momentum=0.):

    start = time.time()

    plot_losses = []

    print_loss_total = 0  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every



    params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)

    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=1,verbose=True)

    

    training_pairs = [tensorsFromPair(random.choice(pairs))

                      for i in range(n_iters)]

  

    criterion = nn.NLLLoss()



    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]

        input_tensor = training_pair[0]

        target_tensor = training_pair[1]



        loss = train(input_tensor, target_tensor, encoder,

                     decoder, optimizer, criterion)



        

        print_loss_total += loss

        plot_loss_total += loss



        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every

            scheduler.step(print_loss_avg)

            print_loss_total = 0

            print('%s (%d %d%%) %.4f %.2E' % (timeSince(start, iter / n_iters),

                                         iter, iter / n_iters * 100, print_loss_avg, Decimal(optimizer.param_groups[0]['lr'])))



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

    plt.plot(points)

    plt.show()
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    with torch.no_grad():

        input_tensor = tensorFromSentence(input_rep, sentence,",")

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

                decoded_words.append(output_rep.index2word[topi.item()])



            decoder_input = topi.squeeze().detach()



        return decoded_words, decoder_attentions[:di + 1]
def evaluateRandomly(encoder, decoder, n=10,is_random=True):

    for i in range(n):

        if is_random:

            pair = random.choice(pairs)

        else:

            pair = pairs[i]

        print('>', pair[0])

        print('=', pair[1])

        output_words, attentions = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words)

        print('<', output_sentence)

        print('')
fig = None

gc.collect()
def showAttention(input_sentence, output_words, attentions):

    fig = plt.figure(figsize=(18, 16))

    ax = fig.add_subplot(111)

    input_len = len(input_sentence.split(","))

    ax.matshow(attentions[:len(output_words), :input_len])

    ax.set_xticks(np.arange(0,input_len, step=1))

    ax.set_yticks(np.arange(0,len(output_words)))

    ax.set_xticklabels(input_sentence.split(","), rotation=90)

    ax.set_yticklabels(output_words)

    plt.show()





def evaluateAndShowAttention(input_sentence):

    output_words, attentions = evaluate(

        encoder1, attn_decoder1, input_sentence)

    print('input =', input_sentence)

    print('output =', ' '.join(output_words))

    showAttention(input_sentence, output_words, attentions)

hidden_size = 512

print(str(input_rep.n_words)+" "+str(hidden_size)+" "+str(output_rep.n_words))

encoder1 = EncoderRNN(input_rep.n_words,hidden_size).to(device)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_rep.n_words,dropout_p=0.).to(device)



trainIters(encoder1, attn_decoder1, len(pairs), print_every=1000,learning_rate=1e-2)
evaluateRandomly(encoder1, attn_decoder1,10,is_random=True)
evaluateAndShowAttention("name[Xxx], priceRange[less than £20], area[riverside]")
torch.save([encoder1,attn_decoder1],'encoder_decoder.pkl')
