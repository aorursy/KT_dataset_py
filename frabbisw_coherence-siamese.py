import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
seq_type = 'simple'

data_dir = '/kaggle/input/coherencepkl/'

pkls_dir = data_dir + ''

check_dir = '/kaggle/input/coherence-siamese/'
from matplotlib import pyplot as plt

%matplotlib inline
from __future__ import unicode_literals, print_function, division

from io import open

import unicodedata

import string

import re

import random

import pickle

import json



import numpy as np

import nltk

from nltk.translate.bleu_score import SmoothingFunction

import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=20)
print(device)
hidden_size = 12
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
def load_data():

    with open(pkls_dir+seq_type+'_lang.pkl', 'rb') as f:

        seq_lang = pickle.load(f)

    with open(pkls_dir+'comment_lang.pkl', 'rb') as f:

        comment_lang = pickle.load(f)

    with open(pkls_dir+'train_pairs.pkl', 'rb') as f:

        train_pairs = pickle.load(f)

    with open(pkls_dir+'test_pairs.pkl', 'rb') as f:

        test_pairs = pickle.load(f)

        

    return seq_lang, comment_lang, train_pairs, test_pairs
seq_lang, comment_lang, train_pairs, test_pairs = load_data()
# random.shuffle(train_pairs)

# test_pairs = train_pairs[-300:]

# train_pairs = train_pairs[:-300]
# with open('train_pairs.pkl', 'wb') as f:

#     pickle.dump(train_pairs, f)

# with open('test_pairs.pkl', 'wb') as f:

#     pickle.dump(test_pairs, f)
print(len(test_pairs), len(train_pairs))

test_pairs[0]
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(input_size, hidden_size)

        #self.gru = nn.GRU(hidden_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size)



    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded

        #output, hidden = self.gru(output, hidden)

        output, hidden = self.lstm(output, hidden)

        return output, hidden



    def initHidden(self):

        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))
def indexesFromSentence(lang, sentence):

    indexes=[]

    for word in sentence.split(' '):

        if word in lang.word2index:

            indexes.append(lang.word2index[word])

    return indexes



def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)



def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(seq_lang, pair[0])

    target_tensor = tensorFromSentence(comment_lang, pair[1])

    score = pair[2]

    return (input_tensor, target_tensor, score)
def train(ast_tensor, comment_tensor, score, encoder_ast, encoder_comment, ast_optimizer, comment_optimizer):

    ast_hidden = encoder_ast.initHidden()

    comment_hidden = encoder_comment.initHidden()



    ast_optimizer.zero_grad()

    comment_optimizer.zero_grad()



    ast_length = ast_tensor.size(0)

    comment_length = comment_tensor.size(0)



    for ei in range(ast_length):

        ast_output, ast_hidden = encoder_ast(ast_tensor[ei], ast_hidden)

        

    for ei in range(comment_length):

        comment_output, comment_hidden = encoder_comment(comment_tensor[ei], comment_hidden)

    

    distance = F.pairwise_distance(ast_hidden[0], comment_hidden[0]).sum()

    similarity = torch.exp(-distance)

    # similarity = torch.round(similarity)

    loss = (score - similarity)**2



    loss.backward()



    ast_optimizer.step()

    comment_optimizer.step()

        

    return loss.item(),similarity.item(),distance.item(),score
import time

import math



def asMinutes(s):

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



def timeSince(since):

    now = time.time()

    s = now - since

    return '%s' % (asMinutes(s))
def adjust_learning_rate(optimizer, epoch, decay, lr):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lr * (0.97 ** (epoch // decay))

    print(lr)

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr
def trainIters(encoder_ast, encoder_comment, ast_optimizer, comment_optimizer, n_iters, itr_start=0, print_every=1000, save_every=1000, learning_rate=0.01, decay=5000):

    start = time.time()

    print_loss_total_score_0 = 0

    print_loss_total_score_1 = 0

    print_similarity_total_score_0 = 0

    print_similarity_total_score_1 = 0

    cnt1=1

    cnt0=1



    random.shuffle(train_pairs)

    training_pairs = [tensorsFromPair(p) for p in train_pairs]

    

    print(len(training_pairs))

    

    criterion = nn.NLLLoss()



    total = len(training_pairs)

    for iter in range(itr_start+1, itr_start+n_iters + 1):

        choice = random.randrange(total)

        

        training_pair = training_pairs[choice]

        

        ast_tensor = training_pair[0]

        comment_tensor = training_pair[1]

        score = training_pair[2]

        

        loss,similarity,distance,score = train(ast_tensor, comment_tensor, score, encoder_ast, encoder_comment, ast_optimizer, comment_optimizer)

        

        if score == 1.0:

            print_loss_total_score_1 += loss

            print_similarity_total_score_1 += similarity

            cnt1+=1

        else:

            print_loss_total_score_0 += loss

            print_similarity_total_score_0 += similarity

            cnt0+=1

        

        if iter % print_every == 0:

            prt="iter: "+str(iter)+"\tcell time: "+timeSince(start)

            prt+="\n"

            prt+=("SCORE: "+"0"+"\tSIMILARITY: "+str(print_similarity_total_score_0/cnt0)+"\tLOSS: "+str(print_loss_total_score_0/cnt0))

            prt+="\n"

            prt+=("SCORE: "+"1"+"\tSIMILARITY: "+str(print_similarity_total_score_1/cnt1)+"\tLOSS: "+str(print_loss_total_score_1/cnt1))

            prt+="\n"

      

            print(prt)

    

            print_loss_total_score_0 = 0

            print_loss_total_score_1 = 0

            print_similarity_total_score_0 = 0

            print_similarity_total_score_1 = 0

            start = time.time()

            cnt1=1

            cnt0=1

            

        if iter % save_every == 0:

            save_checkpoint('',encoder_ast,ast_optimizer,encoder_comment,comment_optimizer,prt)

            

        if iter % decay == 0:

            adjust_learning_rate(ast_optimizer,iter,decay,learning_rate)

            adjust_learning_rate(comment_optimizer,iter,decay,learning_rate)
learning_rate=0.1

#total_pairs = len(pairs)
# only for the 1st time to run the notebook

encoder_ast = EncoderRNN(seq_lang.n_words, hidden_size).to(device)

encoder_comment = EncoderRNN(comment_lang.n_words, hidden_size).to(device)



ast_optimizer = optim.SGD(encoder_ast.parameters(), lr=learning_rate)

comment_optimizer = optim.SGD(encoder_comment.parameters(), lr=learning_rate)
def save_checkpoint(folderpath,encoder_ast,ast_optimizer,encoder_comment,comment_optimizer,prt):

    checkpoint = {'ast_optimizer' : ast_optimizer, 'comment_optimizer' : comment_optimizer, 'encoder_ast_state_dict': encoder_ast.state_dict(),  'encoder_comment_state_dict': encoder_comment.state_dict()}

    torch.save(checkpoint, "checkpoint_LSTM_"+str(hidden_size)+".pth")



    text_file = open("hudai.txt", "w")

    text_file.write('hehe')

    text_file.close()
def load_checkpoint(folderpath):

    checkpoint = torch.load(folderpath+"checkpoint_LSTM_"+str(hidden_size)+".pth")

    

    encoder_ast = EncoderRNN(seq_lang.n_words, hidden_size).to(device)

    encoder_ast.load_state_dict(checkpoint['encoder_ast_state_dict'])

    ast_optimizer = checkpoint['ast_optimizer']

    

    encoder_comment = EncoderRNN(comment_lang.n_words, hidden_size).to(device)

    encoder_comment.load_state_dict(checkpoint['encoder_comment_state_dict'])

    comment_optimizer = checkpoint['comment_optimizer']

    

    return encoder_ast,ast_optimizer,encoder_comment,comment_optimizer
# encoder_ast,ast_optimizer,encoder_comment,comment_optimizer = load_checkpoint(check_dir)
# trainIters(encoder_ast, encoder_comment, ast_optimizer, comment_optimizer, 10, itr_start=0, print_every=5, save_every=5, learning_rate=learning_rate, decay=5)

trainIters(encoder_ast, encoder_comment, ast_optimizer, comment_optimizer, 600000, itr_start=0, print_every=10000, save_every=10000, learning_rate=learning_rate, decay=5000)
def calculate_similarity(ast_tensor, comment_tensor, score, encoder_ast, encoder_comment):

    ast_hidden = encoder_ast.initHidden()

    comment_hidden = encoder_comment.initHidden()



    ast_length = ast_tensor.size(0)

    comment_length = comment_tensor.size(0)



    for ei in range(ast_length):

        ast_output, ast_hidden = encoder_ast(ast_tensor[ei], ast_hidden)

        

    for ei in range(comment_length):

        comment_output, comment_hidden = encoder_comment(comment_tensor[ei], comment_hidden)

    

    distance = F.pairwise_distance(ast_hidden[0], comment_hidden[0]).sum()

    similarity = torch.exp(-distance)

    similarity = torch.round(similarity)

    

    return similarity
def get_result(encoder_ast, encoder_comment, pairs=test_pairs):

    similarity_total_score_0 = 0

    similarity_total_score_1 = 0

    cnt1=1

    cnt0=1



    testing_pairs = [tensorsFromPair(p) for p in pairs]

    result = []

    for pair in testing_pairs:

        ast_tensor = pair[0]

        comment_tensor = pair[1]

        score = pair[2]



        similarity = calculate_similarity(ast_tensor, comment_tensor, score, encoder_ast, encoder_comment)

        if score == 1.0:

            similarity_total_score_1 += similarity

            cnt1 += 1

            result.append(np.array([1,similarity.item()]))

        else:

            cnt0 += 1

            similarity_total_score_0 += similarity

            result.append(np.array([0,similarity.item()]))



    return result
result = get_result(encoder_ast, encoder_comment, pairs=test_pairs)
result = np.array(result)

1-np.abs(result[:,0] - result[:,1]).sum()/len(result)