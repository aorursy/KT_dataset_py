!pip install py-rouge
from __future__ import unicode_literals, print_function, division

from io import open

import unicodedata

import string

import re

import random

import json

import math 

import time

import rouge



import matplotlib.pyplot as plt

plt.switch_backend('agg')

import matplotlib.ticker as ticker

import numpy as np



import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

from torchtext.data.metrics import bleu_score



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print(device)
# read file

with open('../input/squadpreprocessed/data/dev_eval.json', 'r') as myfile:

    input_data=myfile.read()



# parse file

obj = json.loads(input_data)



data = []

# show values

for i in range(1, len(obj)+1):

    line = obj[str(i)]['context'] + "\t" + obj[str(i)]['question']

    data.append(line)



print(data[0:3])

PAD_token = 0

SOS_token = 1

EOS_token = 2





class Vocab:

    def __init__(self):

        self.word2index = {}

        self.word2count = {}

        self.index2word = {0:"PAD", 1: "SOS", 2: "EOS"}

        self.n_words = 3  # Count SOS and EOS



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



# Lowercase, trim, and remove non-letter characters





def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])", r" \1", s)

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s





def readNormalizedPairs():

    print("Reading lines...")



    # Read the file and split into lines

    lines = data[0:330]



    # Split every line into pairs and normalize

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines[0:300]]

    pairs_test = [[normalizeString(s) for s in l.split('\t')] for l in lines[300:330]]

    

    return pairs, pairs_test





def prepareData():

    vocab = Vocab()

    pairs, pairs_test = readNormalizedPairs()

    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")

    for pair in pairs+pairs_test:

        vocab.addSentence(pair[0])

        vocab.addSentence(pair[1])

    print("Counted words:")

    print(vocab.n_words)

    return vocab, pairs, pairs_test





vocab, pairs, pairs_test = prepareData()

print(random.choice(pairs))
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.LSTM(hidden_size, hidden_size)



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

        self.gru = nn.LSTM(hidden_size, hidden_size)

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
def indexesFromSentence(vocab, sentence):

    return [vocab.word2index[word] for word in sentence.split(' ')]





def tensorFromSentence(vocab, sentence):

    indexes = indexesFromSentence(vocab, sentence)

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)





def tensorsFromPair(pair):

    input_tensor = tensorFromSentence(vocab, pair[0])

    target_tensor = tensorFromSentence(vocab, pair[1])

    return (input_tensor, target_tensor)



# Helper Functions for printing time elapse



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



x = tensorsFromPair(random.choice(pairs))

print(x)

print(x[0].size())
teacher_forcing_ratio = 0.5

MAX_LENGTH = 3000



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden(), encoder.initHidden()



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

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing



    else:

        # Without teacher forcing: use its own predictions as the next input

        for di in range(target_length):

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()  # detach from history as input



            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:

                break



    loss.backward()



    encoder_optimizer.step()

    decoder_optimizer.step()



    return loss.item() / target_length
#Helper Function to show plots



def showPlot(points):

    plt.figure()

    fig, ax = plt.subplots()

    # this locator puts ticks at regular intervals

    loc = ticker.MultipleLocator(base=0.2)

    ax.yaxis.set_major_locator(loc)

    plt.plot(points)



def trainIters(encoder, decoder, n_iters, print_every=2400, plot_every=100, learning_rate=0.01):

    start = time.time()

    plot_losses = []

    print_loss_total = 0  # Reset every print_every

    plot_loss_total = 0  # Reset every plot_every



    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs))

                      for i in range(n_iters)]

    criterion = nn.NLLLoss(reduction='sum')



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



        if iter % plot_every == 0:

            plot_loss_avg = plot_loss_total / plot_every

            plot_losses.append(plot_loss_avg)

            plot_loss_total = 0



    showPlot(plot_losses)
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    with torch.no_grad():

        input_tensor = tensorFromSentence(vocab, sentence)

        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.initHidden(), encoder.initHidden()



        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)



        for ei in range(input_length):

            encoder_output, encoder_hidden = encoder(input_tensor[ei],

                                                     encoder_hidden)

            encoder_outputs[ei] += encoder_output[0, 0]



        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS



        decoder_hidden = encoder_hidden



        decoded_words = []



        for di in range(max_length):

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:

                decoded_words.append('<EOS>')

                break

            else:

                decoded_words.append(vocab.index2word[topi.item()])



            decoder_input = topi.squeeze().detach()



        return decoded_words



def evaluateRandomly(encoder, decoder, pairs, n=6):

    for i in range(n):

        pair = random.choice(pairs)

        print('>', pair[0])

        print('=', pair[1])

        output_words = evaluate(encoder, decoder, pair[0])

        output_sentence = ' '.join(output_words)

        print('<', output_sentence)

        print('')
hidden_size = 256

encoder1 = EncoderRNN(vocab.n_words, hidden_size).to(device)

decoder1 = DecoderRNN(hidden_size, vocab.n_words).to(device)



trainIters(encoder1, decoder1, 240000, print_every=2400)
# Evaluate Train Set 

evaluateRandomly(encoder1, decoder1, pairs)



# Evaluate Test Set

evaluateRandomly(encoder1, decoder1, pairs_test)
def evaluate_pair(encoder, decoder, context):

    output_words = evaluate(encoder, decoder, context)

    output_sentence = ' '.join(output_words)

    return output_sentence



def FindCandidatesAndReferencesForBLEU(pairs):

    dic  = {}

    candidates = {}

    for c, q in pairs:

        if c in dic.keys():

            dic[c].append(q.split())

        else:

            dic[c] = [q.split()]

        candidates[c] = evaluate_pair(encoder1, decoder1, c).split()

    return list(candidates.values()), list(dic.values())



# Estimate BLEU for set



def estimateBLEU(pairs, set_name):

    for i in range(1,5):

        candidate_corpus, references_corpus = FindCandidatesAndReferencesForBLEU(pairs)

        bleu_test = bleu_score(candidate_corpus, references_corpus, max_n=i, weights=[1./i]*i)

        print("BLEU-" + str(i) + " on " + set_name + " :" + str(bleu_test))

    return



# Estimate BLEU for test set

estimateBLEU(pairs_test, "Test")



# Estimate BLEU for train set

estimateBLEU(pairs, "Train")

def prepare_results(p, r, f):

    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)





for aggregator in ['Avg', 'Best', 'Individual']:

    print('Evaluation with {}'.format(aggregator))

    apply_avg = aggregator == 'Avg'

    apply_best = aggregator == 'Best'



    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],

                           max_n=4,

                           limit_length=True,

                           length_limit=100,

                           length_limit_type='words',

                           apply_avg=apply_avg,

                           apply_best=apply_best,

                           alpha=0.5, # Default F1_score

                           weight_factor=1.2,

                           stemming=True)



    all_hypothesis, all_references = FindCandidatesAndReferencesForBLEU(pairs)



    scores = evaluator.get_scores(all_hypothesis, all_references)



    for metric, results in sorted(scores.items(), key=lambda x: x[0]):

        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference

            for hypothesis_id, results_per_ref in enumerate(results):

                nb_references = len(results_per_ref['p'])

                for reference_id in range(nb_references):

                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))

                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))

            print()

        else:

            print(prepare_results(results['p'], results['r'], results['f']))

    print()