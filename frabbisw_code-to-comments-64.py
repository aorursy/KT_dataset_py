import json

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import matplotlib.pyplot as plt
from __future__ import unicode_literals, print_function, division

from io import open

import unicodedata

import string

import re

import random

import json

import pickle
import numpy as np

import nltk

import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH=500

MAX_COMMENT_LENGTH=30



hidden_size = 64

learning_rate=.01
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
# Turn a Unicode string to plain ASCII, thanks to

# https://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

    )



def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    #s = re.sub(r"([.!?])", r" \1", s)

    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s
def readLangs(lang1, lang2, reverse=False):

    print("Reading lines...")



    # Read the file and split into lines

    lines = open('/kaggle/input/code-comment-ast/code_comments_data/code_comments_data/out_train.json', encoding='utf-8').read().strip().split('\n')



    # Split every line into pairs and normalize

    

    pairs=[]

    for l in lines:

        tmp1=normalizeString(json.loads(l)["ast"])

        tmp2=normalizeString(json.loads(l)["nl"])

        if len(tmp1) <= MAX_LENGTH and len(tmp2) <= MAX_COMMENT_LENGTH:

            pairs.append([tmp1,tmp2])



    # Reverse pairs, make Lang instances

    if reverse:

        pairs = [list(reversed(p)) for p in pairs]

        input_lang = Lang(lang2)

        output_lang = Lang(lang1)

    else:

        input_lang = Lang(lang1)

        output_lang = Lang(lang2)



    return input_lang, output_lang, pairs
#lines = open('/kaggle/input/code-comment-ast/code_comments_data/code_comments_data/out_tmp.json', encoding='utf-8').read().strip().split('\n')

#json.loads(lines[10])["ast"]
# input_lang, output_lang, pairs = readLangs("ast", "comment", reverse=False)

# print(input_lang.name)

# print(pairs[0][0])
def load_data(folderpath):

    with open(folderpath+'/input_lang.pkl', 'rb') as f:

        input_lang = pickle.load(f)

    with open(folderpath+'/output_lang.pkl', 'rb') as f:

        output_lang = pickle.load(f)

    with open(folderpath+'/pairs.pkl', 'rb') as f:

        pairs = pickle.load(f) 

    with open(folderpath+'/pairs2.pkl', 'rb') as f:

        pairs2 = pickle.load(f)     

    return input_lang, output_lang, pairs, pairs2
input_lang, output_lang, pairs, pairs2 = load_data("/kaggle/input/codecommentpkl/pkl-data")

print(random.choice(pairs))
def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs("ast", "comment", reverse)

    print("Read %s sentence pairs" % len(pairs))

    #pairs = filterPairs(pairs)

    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")

    for pair in pairs:

        input_lang.addSentence(pair[0])

        output_lang.addSentence(pair[1])

    print("Counted words:")

    print(input_lang.name, input_lang.n_words)

    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs



#prepareData('code', 'comments', True)



#input_lang, output_lang, pairs = prepareData('ast', 'comment', False)

#print(random.choice(pairs))
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

    indexes=[]

    for word in sentence.split(' '):

        if word in lang.word2index:

            indexes.append(lang.word2index[word])

    return indexes

    #return [lang.word2index[word] for word in sentence.split(' ')]



def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)



def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(input_lang, pair[0])

    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)
#indexesFromSentence(input_lang,"( methoddeclaration ( modifier_public ) modifier_public")
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

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

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
def adjust_learning_rate(optimizer, epoch, decay, lr):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lr * (0.9 ** (epoch // decay))

    print(lr)

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr
def trainIters(encoder, decoder,encoder_optimizer,decoder_optimizer, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, decay=5000):

    start = time.time()

    plot_losses = []

    print_loss_total = 0  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every



    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    

    for iter in range(1, n_iters + 1):

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



        if iter % decay == 0:

            adjust_learning_rate(encoder_optimizer,iter,decay,learning_rate)

            adjust_learning_rate(decoder_optimizer,iter,decay,learning_rate)
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
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    

    #encoder = checkpoint['encoder']12348

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)

    #encoder = EncoderRNN(12348, hidden_size).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    encoder_optimizer = checkpoint['encoder_optimizer']

    

    #decoder = checkpoint['decoder']

    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.3).to(device)

    #decoder = AttnDecoderRNN(hidden_size, 5639, dropout_p=0.5).to(device)

    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    decoder_optimizer = checkpoint['decoder_optimizer']

    

    #for parameter in model.parameters():

    #    parameter.requires_grad = False

    

    #model.eval()

    

    return encoder,encoder_optimizer,decoder,decoder_optimizer
# only for the 1st time to run the notebook

learning_rate=.01



# encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)

# decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.3).to(device)



# encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

# decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#After succesful run of 1st commit 

encoder,encoder_optimizer,decoder,decoder_optimizer = load_checkpoint("/kaggle/input/code-to-comments/checkpoint.pth")
trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, 300000, print_every=5000, learning_rate=learning_rate, decay=10000)

#trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, 10, print_every=2, learning_rate=learning_rate, decay=2)
checkpoint = {'encoder_optimizer' : encoder_optimizer, 'decoder_optimizer' : decoder_optimizer, 'encoder_state_dict': encoder.state_dict(),  'decoder_state_dict': decoder.state_dict()}

#torch.save(checkpoint, 'checkpoint.csv')

torch.save(checkpoint, 'checkpoint.pth')
text_file = open("Output.csv", "w")

text_file.write("Purchase Amount: " "TotalAmount")

text_file.close()
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
def getResultSamples(encoder, decoder, n=10, pairs=pairs):

    ret_list = []

    for i in range(n):

        pair = random.choice(pairs)

        output_words, attentions = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words[:-1])

        ret_list.append((pair[1],output_sentence))

    

    return ret_list
def evaluateRandomly(encoder, decoder, n=10, pairs=pairs):

    for i in range(n):

        pair = random.choice(pairs)

        #print('>', pair[0])

        print('Real:\t\t', pair[1])

        output_words, attentions = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words[:-1])

        print('Generated:\t', output_sentence)

        print('')
# lines = open('/kaggle/input/code-comment-ast/code_comments_data/code_comments_data/out_test.json', encoding='utf-8').read().strip().split('\n')

# pairs2=[]

# for l in lines:

#     tmp1=normalizeString(json.loads(l)["ast"])

#     tmp2=normalizeString(json.loads(l)["nl"])

#     if len(tmp1) <= MAX_LENGTH and len(tmp2) <= MAX_COMMENT_LENGTH:

#         pairs2.append([tmp1,tmp2])
evaluateRandomly(encoder, decoder,10,pairs=pairs2)
total_sam = 1000

outputs = getResultSamples(encoder, decoder,total_sam, pairs=pairs)

BLEUscore = 0

for out in outputs:

    BLEUscore += nltk.translate.bleu_score.sentence_bleu([out[1]], out[0])



print("####Training Score####")

print(BLEUscore/total_sam)

print("\n")
total_sam = 1000

outputs = getResultSamples(encoder, decoder,total_sam, pairs=pairs2)

BLEUscore = 0

for out in outputs:

    BLEUscore += nltk.translate.bleu_score.sentence_bleu([out[1]], out[0])



print("####Testing Score####")

print(BLEUscore/total_sam)

print("\n")
#input_lang, output_lang, pairs = readLangs("ast", "comment", reverse=False)

#print(len(pairs))
#sam_ast = "( MethodDeclaration ( Modifier_public ) Modifier_public ( Modifier_static ) Modifier_static ( SimpleType ( SimpleName_String ) SimpleName_String ) SimpleType ( SimpleName_unEscapeString ) SimpleName_unEscapeString ( SingleVariableDeclaration ( SimpleType ( SimpleName_String ) SimpleName_String ) SimpleType ( SimpleName_str ) SimpleName_str ) SingleVariableDeclaration ( SingleVariableDeclaration ( PrimitiveType ) PrimitiveType ( SimpleName_escapeChar ) SimpleName_escapeChar ) SingleVariableDeclaration ( SingleVariableDeclaration ( PrimitiveType ) PrimitiveType ( SimpleName_charToEscape ) SimpleName_charToEscape ) SingleVariableDeclaration ( Block ( ReturnStatement ( MethodInvocation ( SimpleName_unEscapeString ) SimpleName_unEscapeString ( SimpleName_str ) SimpleName_str ( SimpleName_escapeChar ) SimpleName_escapeChar ( ArrayCreation ( ArrayType ( PrimitiveType ) PrimitiveType ) ArrayType ( ArrayInitializer ( SimpleName_charToEscape ) SimpleName_charToEscape ) ArrayInitializer ) ArrayCreation ) MethodInvocation ) ReturnStatement ) Block ) MethodDeclaration"

#sam_ast = normalizeString(sam_ast)
#pairs[10][0]
#sam_ast
# output_words, attentions = evaluate(encoder,decoder, pairs[10][0])

# plt.matshow(attentions.numpy())
# from tqdm import tqdm as tq

# mx=0

# dd=0

# for i in tq(range(len(lines))):

#     line = lines[i]

#     if mx < len(json.loads(line)["ast"].split(" ")):

#         mx = len(json.loads(line)["ast"].split(" "))

#         dd = i

# print(mx,dd)

# len(json.loads(lines[dd])["ast"].split(" "))