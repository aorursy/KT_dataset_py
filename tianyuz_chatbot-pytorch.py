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

import urllib.request

import zipfile
#### global variables

PAD_token = 0

SOS_token = 1

EOS_token = 2

MAX_LENGTH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########## download data ###########

def download_data(corpus):

    print('Downloading data ...')

    if not os.path.exists(corpus):

        urllib.request.urlretrieve('http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', 'cornell_movie_dialogs_corpus.zip')

        with zipfile.ZipFile('cornell_movie_dialogs_corpus.zip', 'r') as zip_ref:

            zip_ref.extractall(corpus)

        os.remove('cornell_movie_dialogs_corpus.zip')





########## process downloaded data: extract conversations  ###########

def extract_conversations(corpus):

    print('Extracting conversations ...')

    datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

    if not os.path.exists(datafile):

        delimiter = '\t'

        delimiter = str(codecs.decode(delimiter, 'unicode_escape'))



        lines = {}

        conversations = {}

        MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']

        MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']



        print('\nProcessing corpus ... ')

        lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)

        print('\nLoading conversations ...')

        conversations = loadConversations(os.path.join(corpus, 'movie_conversations.txt'),

                                         lines, MOVIE_CONVERSATIONS_FIELDS)



        print('\nWriting newly formatted file ...')

        with open(datafile, 'w', encoding = 'utf-8') as outputfile:

            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')

            for pair in extractSentencePairs(conversations):

                writer.writerow(pair)



    # print('\nSample lines from file:')

    # printLines(datafile)

    print('')

    return datafile





def printLines(file, n=10):

    with open(file, 'rb') as datafile:

        lines = datafile.readlines()

    for line in lines[:n]:

        print(line)





def loadLines(fileName, fields):

    lines = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:

        for line in f:

            values = line.split(' +++$+++ ')

            lineObj = {}

            for i, field in enumerate(fields):

                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines



def loadConversations(fileName, lines, fields):

    conversations = []

    with open(fileName, 'r', encoding = 'iso-8859-1') as f:

        for line in f:

            values = line.split(' +++$+++ ')

            convObj = {}

            for i, field in enumerate(fields):

                convObj[field] = values[i]

            utterance_id_pattern = re.compile('L[0-9]+')

            lineIds = utterance_id_pattern.findall(convObj['utteranceIDs'])

            convObj['lines'] = []

            for lineId in lineIds:

                convObj['lines'].append(lines[lineId])

            conversations.append(convObj)

    return conversations



def extractSentencePairs(conversations):

    qa_pairs = []

    for conversation in conversations:

        for i in range(len(conversation['lines']) - 1):

            inputLine = conversation['lines'][i]['text'].strip()

            targetLine = conversation['lines'][i+1]['text'].strip()

            if inputLine and targetLine:

                qa_pairs.append([inputLine, targetLine])

    return qa_pairs







########## String processing  ###########

def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if

                  unicodedata.category(c) != 'Mn')



def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])",r" \1", s)

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    s = re.sub(r"\s+", r" ", s).strip()

    return s



def readVocs(datafile, corpus_name):

    print('Reading lines ...')

    lines = open(datafile, encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    voc = Voc(corpus_name)

    return voc, pairs



def filterPair(p):

    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH



def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]



def loadPrepareData(corpus, corpus_name, datafile):

    print('Start preparing training data ...')

    voc, pairs = readVocs(datafile, corpus_name)

    print('Read {!s} sentence pairs'.format(len(pairs)))

    pairs = filterPairs(pairs)

    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))

    print('Counting words ...')

    for pair in pairs:

        voc.addSentence(pair[0])

        voc.addSentence(pair[1])

    print('Counted words:', voc.num_words)

    return voc, pairs





def trimRareWords(voc, pairs, MIN_COUNT):

    voc.trim(MIN_COUNT)



    keep_pairs = []

    for pair in pairs:

        input_sentence = pair[0]

        output_sentence = pair[1]

        keep_input = True

        keep_output = True



        for word in input_sentence.split(' '):

            if word not in voc.word2index:

                keep_input = False

                break

        if keep_input:

            for word in output_sentence.split(' '):

                if word not in voc.word2index:

                    keep_output = False

                    break

        if keep_input and keep_output:

            keep_pairs.append(pair)



    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs









########## String to tensor  ###########

def indexesFromSentence(voc, sentence):

    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]



def zeroPadding(l):

    fillvalue=PAD_token

    return list(itertools.zip_longest(*l, fillvalue=fillvalue))



def binaryMatrix(l):

    value=PAD_token

    m = []

    for i, seq in enumerate(l):

        m.append([])

        for token in seq:

            if token == PAD_token:

                m[i].append(0)

            else:

                m[i].append(1)

    return m



def inputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    padVar = torch.LongTensor(padList)

    return padVar, lengths



def outputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    mask = binaryMatrix(padList)

    mask = torch.BoolTensor(mask)

    padVar = torch.LongTensor(padList)

    return padVar, mask, max_target_len



def batch2TrainData(voc, pair_batch):

    pair_batch.sort(key = lambda x: len(x[0].split(' ')), reverse = True)

    input_batch, output_batch = [], []

    for pair in pair_batch:

        input_batch.append(pair[0])

        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)

    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len



def show_some_tensors():

    small_batch_size = 5

    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])

    input_variable, lengths, target_variable, mask, max_target_len = batches



    print("input_variable:", input_variable)

    print("lengths:", lengths)

    print("target_variable:", target_variable)

    print("mask:", mask)

    print("max_target_len:", max_target_len)









########## training and evaluation  ###########

def maskNLLLoss(inp, target, mask):

    nTotal = mask.sum()

    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))

    loss = crossEntropy.masked_select(mask).mean()

    loss = loss.to(device)

    return loss, nTotal.item()







# def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,

         # encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,

         encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio):

    max_length=MAX_LENGTH

    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()



    input_variable = input_variable.to(device)

    lengths = lengths.to(device)

    target_variable = target_variable.to(device)

    mask = mask.to(device)



    loss = 0

    print_losses = []

    n_totals = 0



    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)



    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    decoder_input = decoder_input.to(device)



    decoder_hidden = encoder_hidden[:decoder.n_layers]



    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False



    if use_teacher_forcing:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = target_variable[t].view(1, -1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal

    else:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(

                decoder_input, decoder_hidden, encoder_outputs

            )

            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal

    loss.backward()

    # _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)

    # _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    nn.utils.clip_grad_norm_(encoder.parameters(), clip)

    nn.utils.clip_grad_norm_(decoder.parameters(), clip)



    encoder_optimizer.step()

    decoder_optimizer.step()

    return sum(print_losses)/n_totals





def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,

              embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size,

              print_every, save_every, clip, corpus_name, checkpoint_iter, teacher_forcing_ratio):

    print('Initializing ...')

    start_iteration = 1

    print_loss = 0

    if checkpoint_iter:

        start_iteration = checkpoint_iter + 1

    print('Training ...')

    for iteration in range(start_iteration, n_iteration + 1):

        training_batch = batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])

        input_variable, lengths, target_variable, mask, max_target_len = training_batch



        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,

                    decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip,

                    teacher_forcing_ratio)

        print_loss += loss



        if iteration % print_every == 0:

            print_loss_avg = print_loss / print_every

            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))

            print_loss = 0



        if iteration % save_every == 0:

            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))

            if not os.path.exists(directory):

                os.makedirs(directory)

            torch.save({

                'itereation': iteration,

                'en':encoder.state_dict(),

                'de':decoder.state_dict(),

                'en_opt':encoder_optimizer.state_dict(),

                'de_opt':decoder_optimizer.state_dict(),

                'loss':loss,

                'voc_dict':voc.__dict__,

                'embedding':embedding.state_dict()

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))









# def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):

def evaluate(encoder, decoder, searcher, voc, sentence):

    max_length = MAX_LENGTH

    indexes_batch = [indexesFromSentence(voc, sentence)]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    input_batch = input_batch.to(device)

    lengths = lengths.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)

    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words



def evaluateInput(encoder, decoder, searcher, voc):

    input_sentence = ''

    while(1):

        try:

            input_sentence = input('> ')

            if input_sentence == 'q' or input_sentence == 'quit': break

            input_sentence = normalizeString(input_sentence)

            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

            print('Bot: ', ' '.join(output_words))

        except KeyError:

            print('Error: Encountered unknown word.')
class Voc:

    def __init__(self, name):

        self.name = name

        self.trimmed = False

        self.word2index = {}

        self.word2count = {}

        self.index2word = {PAD_token:'PAD', SOS_token:'SOS', EOS_token:'EOS'}

        self.num_words = 3



    def addSentence(self, sentence):

        for word in sentence.split(' '):

            self.addWord(word)



    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.num_words

            self.word2count[word] = 1

            self.index2word[self.num_words] = word

            self.num_words += 1

        else:

            self.word2count[word] += 1



    def trim(self, min_count):

        if self.trimmed:

            return

        else:

            keep_words = []

            for k,v in self.word2count.items():

                if v >= min_count:

                    keep_words.append(k)



            print('keep_words {} / {} = {:.4f}'.format(

                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)

            ))



            self.word2index = {}

            self.word2count = {}

            self.index2word = {PAD_token:'PAD', SOS_token:'SOS', EOS_token:'EOS'}

            self.num_words = 3

            for word in keep_words:

                self.addWord(word)



            self.trimmed = True





class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):

        super().__init__()

        self.n_layers = n_layers

        self.hidden_size = hidden_size

        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,

                         dropout=(0 if n_layers==1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden = None):

        embedded = self.embedding(input_seq)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden





# Luong attention layer

class Attn(nn.Module):

    def __init__(self, method, hidden_size):

        super().__init__()

        self.method = method

        if method not in ['dot', 'general', 'concat']:

            raise ValueError(self.method, 'is not an appropriate attention method.')

        self.hidden_size = hidden_size

        if method == 'general':

            self.attn = nn.Linear(hidden_size, hidden_size)

        elif method == 'concat':

            self.attn = nn.Linear(hidden_size * 2, hidden_size)

            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):

        return torch.sum(hidden*encoder_output, dim = 2)

    def genereal_socre(self, hidden, encoder_output):

        energy = self.attn(encoder_output)

        return torch.sum(hidden * energy, dim = 2)

    def concat_score(self, hidden, encoder_output):

        energy = self.attn(torch.cat((hidden.expend(encoder_output.size(0),-1,-1),encoder_output), 2)).tanh()

        return torch.sum(self.v * energy, dim = 2)

    def forward(self, hidden, encoder_outputs):

        if self.method == 'general':

            attn_energies = self.general_score(hidden, encoder_outputs)

        elif self.method == 'concat':

            attn_energies = self.concat_score(hidden, encoder_outputs)

        elif self.method == 'dot':

            attn_energies = self.dot_score(hidden, encoder_outputs)



        attn_energies = attn_energies.t()



        return F.softmax(attn_energies, dim=1).unsqueeze(1)







class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):

        super().__init__()

        self.attn_model = attn_model

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.n_layers = n_layers

        self.dropout = dropout



        self.embedding = embedding

        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))

        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        embedded = self.embedding(input_step)

        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        rnn_output = rnn_output.squeeze(0)

        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)

        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)

        output = F.softmax(output, dim=1)

        return output, hidden





class GreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)

        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores
print('Device to use:', device)



########## variables (no need to change anything else besides these) ##########

#### folder/file name

corpus = 'cornell movie-dialogs corpus'

corpus_name = 'cornell movie-dialogs corpus'

save_dir = 'save'



#### vocabulary trimming

MIN_COUNT = 3  # for trimming rate words



#### models configuration

model_name = 'cb_model'

attn_model = 'dot' # general, concat

hidden_size = 500

encoder_n_layers = 2

decoder_n_layers = 2

dropout = 0.1

batch_size = 64



#### training configuration

clip = 50.0

teacher_forcing_ratio = 0.5 # 1.0

learning_rate = 0.0001

decoder_learning_ratio = 5.0

n_iteration = 20000

print_every = 1000

save_every = 5000



#### continue training or new training

loadFilename = None

checkpoint_iter = None

#### if load from saved model, run the following

# checkpoint_iter = 38000

# loadFilename = os.path.join(save_dir, model_name, corpus_name,

#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),

#                            '{}_checkpoint.tar'.format(checkpoint_iter))



########## end of variables ##########









#### download data and do extract conversations

download_data(corpus)

corpus = os.path.join(corpus, corpus)

datafile = extract_conversations(corpus)



#### load conversations and make tensors

voc, pairs = loadPrepareData(corpus, corpus_name, datafile)

# print('\npairs:')

# for pair in pairs[:10]:

#     print(pair)



pairs = trimRareWords(voc, pairs, MIN_COUNT)

# show_some_tensors()



#### build models

print('Building encoder and decoder ...')

embedding = nn.Embedding(voc.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder = encoder.to(device)

decoder = decoder.to(device)

print('Models built and ready to go!')



print('Building optimizers ...')

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)



encoder.train()

decoder.train()



if loadFilename:

    # If loading on same machine the model was trained on

    checkpoint = torch.load(loadFilename)

    # If loading a model trained on GPU to CPU

    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))



    voc.__dict__ = checkpoint['voc_dict']

    embedding.load_state_dict(checkpoint['embedding'])

    encoder.load_state_dict(checkpoint['en'])

    decoder.load_state_dict(checkpoint['de'])

    encoder_optimizer.load_state_dict(checkpoint['en_opt'])

    decoder_optimizer.load_state_dict(checkpoint['de_opt'])



for state in encoder_optimizer.state.values():

    for k, v in state.items():

        if isinstance(v, torch.Tensor):

            state[k] = v.cuda()

for state in decoder_optimizer.state.values():

    for k, v in state.items():

        if isinstance(v, torch.Tensor):

            state[k] = v.cuda()



print('Starting Training!')

trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, hidden_size,

          save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, checkpoint_iter,

          teacher_forcing_ratio)



encoder.eval()

decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)



evaluateInput(encoder, decoder, searcher, voc)