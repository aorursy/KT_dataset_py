from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from __future__ import unicode_literals



import torch

from torch.jit import script, trace

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

import csv

import random

import re

import os

import unicodedata

import codecs

from io import open

import itertools

import math

import time



device = "cpu"
corpus = "../input/"

#corpus = os.path.join("data", corpus_name)
# to read or write files use open

# to access file module use os

lines_filepath = os.path.join(corpus, "movie_lines.txt")

conv_filepath = os.path.join(corpus, "movie_conversations.txt")
# visualise some lines

with open(lines_filepath, 'r',encoding = "ISO-8859-1") as file:

    lines = file.readlines()

for line in lines[:8]:

    print(line.strip()) 

# displayOrder: lineid charID movieID charNAME utterance

print()

# visualise some conversations

with open(conv_filepath, 'r',encoding = "ISO-8859-1") as file:

    conv = file.readlines()

for line in conv[:8]:

    print(line.strip()) 

# displayOrder: char1ID char2ID movieID lineIDs
# Splits each line of the file into a dictionary of fields(lineID, charID, movieID, characterNAME, text)

line_fields = ["lineID", "characterID", "movieID", "character", "text"]

lines = {} # empty dictionary

with open(lines_filepath, 'r', encoding='iso-8859-1') as f:

    for line in f:

        values = line.split(" +++$+++ ")

        # Extarct fields

        lineObj = {}

        for i, field in enumerate(line_fields):

            lineObj[field] = values[i]

        lines[lineObj['lineID']] = lineObj # key is lineID
# lines

list(lines.items())[:2]
# process the conversations

# Group fields of lines from 'LoadLines' into conversatons based on 'movie_conversations.txt'

conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

conversations = []

with open(conv_filepath, 'r', encoding='iso-8859-1') as f:

    for line in f:

        values = line.split(" +++$+++ ")

        # Extract fields

        convObj = {}

        for i, field in enumerate(conv_fields):

            convObj[field] = values[i]

#       Convert string resulted from split to list, since ConvObj["utterancesIDs"] == "['L8299', ...]"

        # remember that utterance ID is the LineID

        lineIDs = eval(convObj["utteranceIDs"])

        # Reassamble lines

        convObj["lines"] = []

        for lineID in lineIDs:

            convObj["lines"].append(lines[lineID])

        conversations.append(convObj)
conversations[0]
conversations[0]["lines"][0]["text"].strip() # with strip '/n' is discarded
# processing the dataset part 3

# Extract pair of sentences from conversations

# ques ans pair

qa_pairs = []

for conversation in conversations:

    for i in range(len(conversation["lines"]) - 1):

        inputLine = conversation["lines"][i]["text"].strip()

        targetLine = conversation["lines"][i+1]["text"].strip()

        # filter wrong results

        if inputLine and targetLine:

            qa_pairs.append([inputLine, targetLine])
qa_pairs[2]
corpus2 = "../working/"

#corpus = os.path.join("data", corpus_name)



# Define path to new file

datafile = os.path.join(corpus2, "formatted_movie_lines.txt")

delimiter = '\t'

# Unescape the delimiter

delimiter = str(codecs.decode(delimiter, "unicode_escape"))



# write new csv file note that it is tab seperated not comma seperated

print("\nWriting newly formatted file...")

with open(datafile, 'w', encoding='utf-8') as outputfile:

    writer = csv.writer(outputfile, delimiter = delimiter, lineterminator='\n')

    for pair in qa_pairs:

        writer.writerow(pair)

print("Done writing to file")
# visualise some lines

datafile = os.path.join(corpus2, "formatted_movie_lines.txt")

with open(datafile, 'rb') as file:

    lines = file.readlines()

for line in lines[:10]:

    print(line)
# Processing the words

PAD_token = 0 # used for padding short sentences

SOS_token = 1 # Start of sent token

EOS_token = 2 # End of Sent token



class Vocabulary:

    def __init__(self, name):

        self.name = name

        self.word2index = {}

        self.word2count = {}

        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

        self.num_words = 3 # Count SOS, EOS, PAD

        

    def addSentence(self, sentence):

        for word in sentence.split(' '):

            self.addWord(word)

    # add index(unique value) for each word

    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.num_words # just opp of 3rd line

            self.word2count[word] = 1

            self.index2word[self.num_words] = word

            self.num_words += 1

        else:

            self.word2count[word] += 1

            

    # remove words that don't pass min count

    def trim(self, min_count):

        keep_words = []

        for k, v in self.word2count.items():

            if v >= min_count:

                keep_words.append(k)

                

        # print words that are kept

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitailize dictionaries

        self.word2index = {}

        self.word2count = {}

        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

        self.num_words = 3 # Count SOS, EOS, PAD



        for word in keep_words:

            self.addWord(word)
sen = "hi ali bin arshad"

sen.split()
# turn unicode string to plain ASCII

# nfd: normal form decomposed

# mn: normalized

def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!= 'Mn')
''.join(['a','l','i'])
# Lowercase, trim, and remove non-letter characters

def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    # re.sub = Subsitute any .!? by a whitespace + the character

    # r is to not consider \1 as a character(r to escape a backslash)

    s = re.sub(r"([.!?])", r" \1", s)

    # remove any char that is not a sequence of lower or upper character

    # + means one or more

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    # remove seq of whitespace charactersc

    s = re.sub(r"\s+", r" ", s).strip()

    return s
# testing the funcion

normalizeString("aa123bcd!s's    ad?")
datafile = os.path.join(corpus, "formatted_movie_lines.txt")



# Read query/response pairs and return voc object

def readVocs(datafile, corpus_name):

    # Read the file and split into lines

    print("Reading and processing file.....Plaese Wait!")

    lines = open(datafile, encoding='utf-8').read().strip().split('\n') # conversatons are splitted by \n

    # Split every line into pair and normalize

    pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines] # for each char(represented by pair) we are gonna normalize

    print("Done Reading!")

    voc = Vocabulary(corpus_name)

    return voc, pairs

    

# a = [i for i in range(0,10)]

# a
# Returns True if boths sentences in a pair 'p' are under the MAX_LENGTH threshold

MAX_LENGTH = 10 # Max sentence length to consider

def filterPair(p):

    # Input sequences need to preserve the last word for EOS token

    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

    # if True keep the pair else ignore

# Filter pairs using filterPair condition

def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]



# Using the functions defined above, return a populated voc object and pairs list

def loadPrepareData(corpus, corpus_name, datafile, save_dir):

    print("Start preparing training data ...")

    voc, pairs = readVocs(datafile, corpus_name)

    print("Read {!s} sentence pairs".format(len(pairs)))

    pairs = filterPairs(pairs)

    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    print("Counting words...")

#      Word 2 index word conversion occurs here

    for pair in pairs:

        voc.addSentence(pair[0])

        voc.addSentence(pair[1])

    print("Counted words:", voc.num_words)

    return voc, pairs





# Load/Assemble voc and pairs

save_dir = os.path.join("data", "save")

voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

# Print some pairs to validate

print("\npairs:")

for pair in pairs[:10]:

    print(pair)

# We don't want our model to have words which appeared less than 3 times

MIN_COUNT = 3



def trimRareWords(voc, pairs, MIN_COUNT):

    # Trim words less than MIN_COUNT from voc

    voc.trim(MIN_COUNT) # Function from class voc

    # filter out pairs with trimmed words

    keep_pairs = []

    for pair in pairs:

        input_sequence = pair[0]

        output_sequence = pair[1]

        keep_input = True

        keep_output = True

        # Check input Sequence

        for word in input_sequence.split(' '):

            if word not in voc.word2index: # voc.word2index contains filtered words

                keep_input = False

                break

        # now check output sequence

        for word in output_sequence.split(' '):

            if word not in voc.word2index:

                keep_output = False

                break

                

        # only keep words that are not trimmed in in or out seq

        if keep_input and keep_output:

            keep_pairs.append(pair)

            

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs

        

# trim voc and pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)
# preparing data part1

# returns converted word 2 index

def indexesFromSentence(voc, sentence):

    return [voc.word2index[word] for word in sentence.split(' ')]+ [EOS_token]
pairs[1][0]
# Testing function

indexesFromSentence(voc,pairs[1][0])

# 2 at the end of output represents EOS_token
# Define some samples for testing

inp = []

out = []

# print(pairs[1])

for pair in pairs[:10]:

    inp.append(pair[0])

    out.append(pair[1])

print(inp)

print(len(inp))

indexes = [indexesFromSentence(voc, sentence) for sentence in inp]

indexes
a = ['A', 'B', 'C', 'D', 'E']

b = [1, 2, 3]

print(list(zip(a,b)))

list(itertools.zip_longest(a,b))
# Preparing data for model part 2

a = [[3, 4, 2],

 [7, 8, 9, 10, 4, 11, 12, 13, 2],

 [16, 4, 2],

 [8, 31, 22, 6, 2],

 [33, 34, 4, 4, 4, 2],

 [35, 36, 37, 38, 7, 39, 40, 41, 4, 2],

 [42, 2],

 [47, 7, 48, 40, 45, 49, 6, 2],

 [50, 51, 52, 6, 2],

 [58, 2]]

list(itertools.zip_longest(*a, fillvalue = 0))
def zeroPadding(l, fillValue = 0):

    return list(itertools.zip_longest(*l, fillvalue=fillValue))
leng = [len(ind) for ind in indexes]

max(leng)
# Test the function

test_result = zeroPadding(indexes)

print(len(test_result))

test_result
# data modelling part 3

# l contains the index value of each word

# remember that pad+token is 0

def binaryMatrix(l, value=0):

    m = []

    for i, seq in enumerate(l):

        m.append([])

        for token in seq:

            if token == PAD_token:

                m[i].append(0)

            else:

                m[i].append(1)

    return m
binary_result = binaryMatrix(test_result)

binary_result
#  Returns padded input tensor and a tensor of lengths for each or the sequences in the batch

# l is going to be questions not replies, i.e. only what the first char said

def inputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    padVar = torch.LongTensor(padList)

    return padVar, lengths
# returns padded target sequence tensor, padding mask and max target len

# this one is for all replies

def outputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    mask = binaryMatrix(padList)

    mask = torch.ByteTensor(mask)

    padVar = torch.LongTensor(padList)

    return padVar, mask, max_target_len
pairs[20000]
# Returns all items for a given batch of pairs

def batch2TrainData(voc, pair_batch):

    # Sorts the ques in descending order as shown in img above

    pair_batch.sort(key=lambda x:len(x[0].split(" ")), reverse=True)

    input_batch, output_batch = [], []

    for pair in pair_batch:

        input_batch.append(pair[0])

        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)

    # assert len(inp) == lengyh[0]

    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len
# Example for validation

small_batch_size = 5

batches = batch2TrainData(voc, [random.choice(pairs) for i in range(small_batch_size)])

input_variable, lengths, target_variable, mask, max_target_len = batches



print("input variable:")

print(input_variable)

print("lengths: ", lengths)

print("target_variable:")

print(target_variable)

print("mask:")

print(mask)

print("max_target_len:", max_target_len)

# encoder class inherited from nn.Module

# hidden size: how many RNN cells are there in hidden layer

# embedding: converts index to dense vector of values

# size of embedding = number of input features

# seq is the timestep

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):

        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers

        self.hidden_size = hidden_size

        self.embedding = embedding

        

        # the input_size & hidden_size params are both set to 'hidden_size'

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    

    # overridden forward function of class nn.Module

    def forward(self, input_seq, input_lengths, hidden=None):

        # input_seq: batch of input sentences; shape=(max_length, batch_size)

        # input_lengths: list of sentence lengths corresponding to each sentence in the batch

        # hidden state of shape:(n_layers times num_directions(2 in this case), batch_size, hidden_size)

        

        # convert word indexes to embeddings

        embedded = self.embedding(input_seq)

        # pack padded batch of sequences for RNN module

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through  GRU

        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional gru outputs

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # Return output from final hidden state

        # outputs: (timesteps, batch, hidden_size)

        return outputs, hidden

        # outputs: the output features h_t from the last layer of GRU, for each timestep(sum of bidirectional outputs)

        # hidden: hidden state for the last timestep, of shape=(n_layers x mum_directions, batch_size, hidden_size)

    
# Understanding Pack Padded Sequence
# decoder is built with attention mechanism
# Luong attention layer

class Attn(torch.nn.Module):

    def __init__(self, method, hidden_size): # method can be dot, general or concat

        super(Attn, self).__init__()

        self.method = method

        if self.method not in ['dot', 'general', 'concat']:

            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size

        if self.method == 'general':

            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':

            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)

            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

        

    def dot_score(self, hidden, encoder_output): # hidden is the decoder output at certain timestep

     # Element wise multiplication of current target state with the encoder output and sum them

        return torch.sum(hidden * encoder_output, dim=2)



    def general_score(self, hidden, encoder_output):

        energy = self.attn(encoder_output)

        return torch.sum(hidden * energy, dim=2)



    def concat_score(self, hidden, encoder_output):

        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()

        return torch.sum(self.v * energy, dim=2)

    

    def forward(self, hidden, encoder_outputs):

        # hidden of shape:(1,batch_size, hidden_size)

        # encoder output shape: (max_length, batch_size, hidden_size)

        # comment 3:(1, batch_size, hidden_size) * (max_length, batch_size, hidden_size) = (max_length, batch_size, hidden_size)

        

        # calculate the attention weights(energies)

        if self.method == 'general':

            attn_energies = self.general_score(hidden, encoder_outputs)

        elif self.method == 'concat':

            attn_energies = self.concat_score(hidden, encoder_outputs)

        elif self.method == 'dot':

            attn_energies = self.dot_score(hidden, encoder_outputs) # (max_length, batch_size)

        # Transpose max_length and batch_size dimensions

        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores(with added dimension)

        return F.softmax(attn_energies, dim=1).unsqueeze(1) # (batch_size, 1, max_length)

    # softmax: sum of entire row is 1

    
# understanding of comment 3

# summing across dim=2 means we're summing all the cols of each row

a = torch.randn(5,3,7)

print(a)

torch.sum(a, dim=2)
# Designing the Decoder 1

# We are using the attention to build decoder

# note that data is(max_length,batch_size)

# we feed data row by row to every GRU   

# output from encoder is fed to 1st timestep of decoder

# dropout drops random num of neurons in each layer, helps neurons not to be dependent on each other





class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):

        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.n_layers = n_layers

        self.dropout = dropout

        

        # Define layers

        self.embedding = embedding

        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))

        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        

        self.attn = Attn(attn_model, hidden_size)

        

    def forward(self, input_step, last_hidden, encoder_outputs):

        # input_step: one time step (one word) of input sequence batch; shape=(1, batch_size)

        # last_hidden: final hidden state of encoder GRU; shape=(n_layers x num_directions, batch_size, hidden_state)

        # encoder_outputs: encoder model's output; shape=(seq_len, batch, num_directions * hidden_size)

        # Note: we run this one step (batch of words) at a time

        

        # Get embedding of current input word

        # nn.Embedding as a lookup table where the key is the word index and the value is the corresponding word vector

        embedded = self.embedding(input_step)

        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU

        rnn_output, hidden = self.gru(embedded, last_hidden)

        # rnn_output of shape = (1, batch, num_directions * hidden_size)

        # hidden of shape = (num_layers * num_directions, batch, hidden_size)

        # hidden is the hidden state of the current time step of GRU

        

        # Calculate attention weights from the current GRU output

        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new weighted sum context vector

        # (batch_size, 1m max_lengths) bmm(batch multiplication) with (batch_size, max_length, hidden) = (batch_size, 1, hidden)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted vectors and GRU output

        rnn_output = rnn_output.squeeze(0)

        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1) # 1 is the dimencion of concatenation

        # concat_input: (batch_size, hidden_size * 2)

        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6

        output = self.out(concat_output)

        output = F.softmax(output, dim=1)

        # return output and final hidden state

        return output, hidden

        # output: sofmax normalized tensor giving probabilies of each word being the correct next word in decoded sequencw

        # shape: (batch_size, voc.num_words)

        # hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)

        
# calculate loss only for non 0 elements

# mask is what was returned above ▲

# we calc loss btw decoder output and target



# NLLL: Negative Log Likelihood Loss

# def maskNLLLoss(decoder_out, target, mask):

#     nTotal = mask.sum()# gives how many non zerro elements we have, that we consider

#     target = target.view(-1,1)

#     # decoder_out shape: (batch_size, vocab_size), target_size = (batch_size, 1)

#     gathered_tensor = torch.gather(decoder_out, 1, target)

#     # calc the NLLL

#     crossEntropy = -torch.log(gathered_tensor)

#     # Select the non-zer0 elements

#     loss = crossEntropy.masked_select(mask) # loss is only for non-zere

#     # calc teh mean of loss

#     loss = loss.mean()

# #     loss = loss.to(device) for cuda

#     return loss, nTotal.item()



def maskNLLLoss(inp, target, mask):

    nTotal = mask.sum()

    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))

    loss = crossEntropy.masked_select(mask).mean()

    loss = loss.to(device)

    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,

          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):



    # Zero gradients

    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()



#     # Set device options

#     input_variable = input_variable.to(device)

#     lengths = lengths.to(device)

#     target_variable = target_variable.to(device)

#     mask = mask.to(device)



    # Initialize variables

    loss = 0

    print_losses = []

    n_totals = 0



    # Forward pass through encoder

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)



    # Create initial decoder input (start with SOS tokens for each sentence)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

#     decoder_input = decoder_input.to(device)



    # Set initial decoder hidden state to the encoder's final hidden state

    decoder_hidden = encoder_hidden[:decoder.n_layers]



    # Determine if we are using teacher forcing this iteration

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False



    # Forward batch of sequences through decoder one time step at a time

    if use_teacher_forcing:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(

                decoder_input, decoder_hidden, encoder_outputs

            )

            # Teacher forcing: next input is current target

            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal

    else:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(

                decoder_input, decoder_hidden, encoder_outputs

            )

            # No teacher forcing: next input is decoder's own current output

            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

#             decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal



    # Perform backpropatation

    loss.backward()



    # Clip gradients: gradients are modified in place, solve exploding gradient problem

    # parameters are weights

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)



    # Adjust model weights

    encoder_optimizer.step()

    decoder_optimizer.step()



    return sum(print_losses) / n_totals
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):



    # Load batches for each iteration

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])

                      for _ in range(n_iteration)]



    # Initializations

    print('Initializing ...')

    start_iteration = 1

    print_loss = 0

    if loadFilename:

        start_iteration = checkpoint['iteration'] + 1



    # Training loop

    print("Training...")

    for iteration in range(start_iteration, n_iteration + 1):

        training_batch = training_batches[iteration - 1]

        # Extract fields from batch

        input_variable, lengths, target_variable, mask, max_target_len = training_batch



        # Run a training iteration with batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,

                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

        print_loss += loss



        # Print progress

        if iteration % print_every == 0:

            print_loss_avg = print_loss / print_every

            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))

            print_loss = 0



        # Save checkpoint

        if (iteration % save_every == 0):

            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))

            if not os.path.exists(directory):

                os.makedirs(directory)

            torch.save({

                'iteration': iteration,

                'en': encoder.state_dict(),

                'de': decoder.state_dict(),

                'en_opt': encoder_optimizer.state_dict(),

                'de_opt': decoder_optimizer.state_dict(),

                'loss': loss,

                'voc_dict': voc.__dict__,

                'embedding': embedding.state_dict()

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
class GreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder):

        super(GreedySearchDecoder, self).__init__()

        self.encoder = encoder

        self.decoder = decoder



    def forward(self, input_seq, input_length, max_length):

        # Forward input through encoder model

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder

        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Initialize decoder input with SOS_token

        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        # Initialize tensors to append decoded words to

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)

        all_scores = torch.zeros([0], device=device)

        # Iteratively decode one word token at a time

        for _ in range(max_length):

            # Forward pass through decoder

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word token and its softmax score

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # Record token and score

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)

            decoder_input = torch.unsqueeze(decoder_input, 0)

        # Return collections of word tokens and scores

        return all_tokens, all_scores
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):

    ### Format input sentence as a batch

    # words -> indexes

    indexes_batch = [indexesFromSentence(voc, sentence)]

    # Create lengths tensor

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations

    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    # Use appropriate device

#     input_batch = input_batch.to(device)

#     lengths = lengths.to(device)

    # Decode sentence with searcher

    tokens, scores = searcher(input_batch, lengths, max_length)

    # indexes -> words

    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words





def evaluateInput(encoder, decoder, searcher, voc):

    input_sentence = ''

    while(1):

        try:

            # Get input sentence

            input_sentence = input('> ')

            # Check if it is quit case

            if input_sentence == 'q' or input_sentence == 'quit': break

            # Normalize sentence

            input_sentence = normalizeString(input_sentence)

            # Evaluate sentence

            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            # Format and print response sentence

            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

            print('Bot:', ' '.join(output_words))



        except KeyError:

            print("Error: Encountered unknown word.")
# Configure models

model_name = 'cb_model'

attn_model = 'dot'

#attn_model = 'general'

#attn_model = 'concat'

hidden_size = 500

encoder_n_layers = 2

decoder_n_layers = 2

dropout = 0.1

batch_size = 64



# Set checkpoint to load from; set to None if starting from scratch

loadFilename = None

checkpoint_iter = 4000

#loadFilename = os.path.join(save_dir, model_name, corpus_name,

#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),

#                            '{}_checkpoint.tar'.format(checkpoint_iter))





# Load model if a loadFilename is provided

if loadFilename:

    # If loading on same machine the model was trained on

    checkpoint = torch.load(loadFilename)

    # If loading a model trained on GPU to CPU

    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    encoder_sd = checkpoint['en']

    decoder_sd = checkpoint['de']

    encoder_optimizer_sd = checkpoint['en_opt']

    decoder_optimizer_sd = checkpoint['de_opt']

    embedding_sd = checkpoint['embedding']

    voc.__dict__ = checkpoint['voc_dict']





print('Building encoder and decoder ...')

# Initialize word embeddings

embedding = nn.Embedding(voc.num_words, hidden_size)

if loadFilename:

    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

if loadFilename:

    encoder.load_state_dict(encoder_sd)

    decoder.load_state_dict(decoder_sd)

# Use appropriate device

encoder = encoder.to(device)

decoder = decoder.to(device)

print('Models built and ready to go!')
# Configure training/optimization

clip = 50.0

teacher_forcing_ratio = 1.0

learning_rate = 0.0001

decoder_learning_ratio = 5.0

n_iteration = 4000

print_every = 1

save_every = 500



# Ensure dropout layers are in train mode

encoder.train()

decoder.train()



# Initialize optimizers

print('Building optimizers ...')

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if loadFilename:

    encoder_optimizer.load_state_dict(encoder_optimizer_sd)

    decoder_optimizer.load_state_dict(decoder_optimizer_sd)



# Run training iterations

print("Starting Training!")

start = time.time()

trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,

           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,

           print_every, save_every, clip, corpus_name, loadFilename)

end = time.time()

print("-------------TIME TAKEN----------------")

print(end - start)

# Run the following to chat
# Set dropout layers to eval mode

encoder.eval()

decoder.eval()



# Initialize search module

searcher = GreedySearchDecoder(encoder, decoder)



evaluateInput(encoder, decoder, searcher, voc)