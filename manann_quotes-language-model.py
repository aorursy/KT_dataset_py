import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/quotes-500k/quotes.csv')
dff = df['quote']

text = dff.str.cat(sep=' ')

stopwords = set(STOPWORDS)

stopwords.add("said")

stopwords.add("one")



wc = WordCloud(max_font_size=40, max_words=200,stopwords=stopwords, contour_width=3, contour_color='steelblue')



wordcloud = wc.generate(text)

plt.figure(figsize=(12, 9))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# source of code https://github.com/pytorch/examples.git



import os

from io import open

import torch



class Dictionary(object):

    def __init__(self):

        self.word2idx = {}

        self.idx2word = []



    def add_word(self, word):

        if word not in self.word2idx:

            self.idx2word.append(word)

            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]



    def __len__(self):

        return len(self.idx2word)





class Corpus(object):

    def __init__(self, path):

        self.dictionary = Dictionary()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))

        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

        self.test = self.tokenize(os.path.join(path, 'test.txt'))



    def tokenize(self, path):

        """Tokenizes a text file."""

        assert os.path.exists(path)

        # Add words to the dictionary

        with open(path, 'r', encoding="utf8") as f:

            for line in f:

                words = line.split() + ['<eos>']

                for word in words:

                    self.dictionary.add_word(word)



        # Tokenize file content

        with open(path, 'r', encoding="utf8") as f:

            idss = []

            for line in f:

                words = line.split() + ['<eos>']

                ids = []

                for word in words:

                    ids.append(self.dictionary.word2idx[word])

                idss.append(torch.tensor(ids).type(torch.int64))

            ids = torch.cat(idss)



        return ids

import math

import torch

import torch.nn as nn

import torch.nn.functional as F



class RNNModel(nn.Module):

    """Container module with an encoder, a recurrent module, and a decoder."""



    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):

        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:

            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        else:

            try:

                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]

            except KeyError:

                raise ValueError( """An invalid option for `--model` was supplied,

                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)



        # Optionally tie weights as in:

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)

        # https://arxiv.org/abs/1608.05859

        # and

        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)

        # https://arxiv.org/abs/1611.01462

        if tie_weights:

            if nhid != ninp:

                raise ValueError('When using the tied flag, nhid must be equal to emsize')

            self.decoder.weight = self.encoder.weight



        self.init_weights()



        self.rnn_type = rnn_type

        self.nhid = nhid

        self.nlayers = nlayers



    def init_weights(self):

        initrange = 0.1

        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()

        self.decoder.weight.data.uniform_(-initrange, initrange)



    def forward(self, input, hidden):

        emb = self.drop(self.encoder(input))

        output, hidden = self.rnn(emb, hidden)

        output = self.drop(output)

        decoded = self.decoder(output)

        return decoded, hidden



    def init_hidden(self, bsz):

        weight = next(self.parameters())

        if self.rnn_type == 'LSTM':

            return (weight.new_zeros(self.nlayers, bsz, self.nhid),

                    weight.new_zeros(self.nlayers, bsz, self.nhid))

        else:

            return weight.new_zeros(self.nlayers, bsz, self.nhid)

seed = 2020

cuda = True

data_path = '../input/quotes-500k/'

batch_size = 20

eval_batch_size = 10

bptt = 35

model_name = 'LSTM'

emsize = 200

nhid = 200

nlayers = 2

dropout = 0.2

lr = 20

clip = 0.25

epochs = 8

tied = True

save = 'model.pth'

log_interval = 2000

onnx_export = ''

nhead = 2
import time

import math

import os

import torch

import torch.nn as nn

import torch.onnx





torch.manual_seed(seed)



device = torch.device("cuda" if cuda else "cpu")



###############################################################################

# Load data

###############################################################################



corpus = Corpus(data_path)





print("Data loaded ")



def batchify(data, bsz):

    # Work out how cleanly we can divide the dataset into bsz parts.

    nbatch = data.size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).

    data = data.narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.

    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)





train_data = batchify(corpus.train, batch_size)

val_data = batchify(corpus.valid, eval_batch_size)

test_data = batchify(corpus.test, eval_batch_size)



###############################################################################

# Build the model

###############################################################################



ntokens = len(corpus.dictionary)



model = RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)



criterion = nn.CrossEntropyLoss()



###############################################################################

# Training code

###############################################################################



def repackage_hidden(h):

    """Wraps hidden states in new Tensors, to detach them from their history."""



    if isinstance(h, torch.Tensor):

        return h.detach()

    else:

        return tuple(repackage_hidden(v) for v in h)





# get_batch subdivides the source data into chunks of length args.bptt.

# If source is equal to the example output of the batchify function, with

# a bptt-limit of 2, we'd get the following two Variables for i = 0:

# ┌ a g m s ┐ ┌ b h n t ┐

# └ b h n t ┘ └ c i o u ┘

# Note that despite the name of the function, the subdivison of data is not

# done along the batch dimension (i.e. dimension 1), since that was handled

# by the batchify function. The chunks are along dimension 0, corresponding

# to the seq_len dimension in the LSTM.



def get_batch(source, i):

    seq_len = min(bptt, len(source) - 1 - i)

    data = source[i:i+seq_len]

    target = source[i+1:i+1+seq_len].view(-1)

    return data, target





def evaluate(data_source):

    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0.

    ntokens = len(corpus.dictionary)

    if True:

        hidden = model.init_hidden(eval_batch_size)

        

    with torch.no_grad():

        for i in range(0, data_source.size(0) - 1, bptt):

            data, targets = get_batch(data_source, i)

            output, hidden = model(data, hidden)

            hidden = repackage_hidden(hidden)

            output_flat = output.view(-1, ntokens)

            total_loss += len(data) * criterion(output_flat, targets).item()

    return total_loss / (len(data_source) - 1)





def train():

    # Turn on training mode which enables dropout.

    model.train()

    total_loss = 0.

    start_time = time.time()

    ntokens = len(corpus.dictionary)

    if True:

        hidden = model.init_hidden(batch_size)

        

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):

        data, targets = get_batch(train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.

        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        model.zero_grad()

        

        hidden = repackage_hidden(hidden)

        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()



        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        for p in model.parameters():

            p.data.add_(-lr, p.grad.data)



        total_loss += loss.item()



        if batch % log_interval == 0 and batch > 0:

            cur_loss = total_loss / log_interval

            elapsed = time.time() - start_time

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '

                    'loss {:5.2f} | ppl {:8.2f}'.format(

                epoch, batch, len(train_data) // bptt, lr,

                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0

            start_time = time.time()





def export_onnx(path, batch_size, seq_len):

    print('The model is also exported in ONNX format at {}'.

          format(os.path.realpath(onnx_export)))

    model.eval()

    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)

    hidden = model.init_hidden(batch_size)

    torch.onnx.export(model, (dummy_input, hidden), path)





# Loop over epochs.

lr = lr

best_val_loss = None



# At any point you can hit Ctrl + C to break out of training early.

try:

    for epoch in range(1, epochs+1):

        epoch_start_time = time.time()

        train()

        val_loss = evaluate(val_data)

        print('-' * 89)

        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '

                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),

                                           val_loss, math.exp(val_loss)))

        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.

        if not best_val_loss or val_loss < best_val_loss:

            torch.save(model, save)

            best_val_loss = val_loss

        else:

            # Anneal the learning rate if no improvement has been seen in the validation dataset.

            lr /= 4.0

except KeyboardInterrupt:

    print('-' * 89)

    print('Exiting from training early')



# Load the best saved model.



model = torch.load(save)

    # after load the rnn params are not a continuous chunk of memory

    # this makes them a continuous chunk, and will speed up forward pass

    # Currently, only rnn model supports flatten_parameters function.

model.rnn.flatten_parameters()



# Run on test data.

test_loss = evaluate(test_data)

print('=' * 89)

print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(

    test_loss, math.exp(test_loss)))

print('=' * 89)



if len(onnx_export) > 0:

    # Export the model in ONNX format.

    export_onnx(onnx_export, batch_size=1, seq_len=bptt)

data_path = '/kaggle/input/quotes-500k/'

checkpoint = 'model.pth'

outf = 'generated.txt'

words = 1000

seed = 2020

cuda = True

temperature = 0.7   #temperature - higher will increase diversity

log_interval = 100

gen_text = ''


import torch



# Set the random seed manually for reproducibility.

torch.manual_seed(seed)



device = torch.device("cuda" if cuda else "cpu")



if temperature < 1e-3:

    parser.error("--temperature has to be greater or equal 1e-3")





model = torch.load(checkpoint).to(device)

model.eval()



corpus = Corpus(data_path)

ntokens = len(corpus.dictionary)

print("data loaded")

#is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'



hidden = model.init_hidden(1)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)





with open(outf, 'w') as outf:

    with torch.no_grad():  # no tracking history

        for i in range(words):

            

            output, hidden = model(input, hidden)

            word_weights = output.squeeze().div(temperature).exp().cpu()

            word_idx = torch.multinomial(word_weights, 1)[0]

            input.fill_(word_idx)



            word = corpus.dictionary.idx2word[word_idx]



            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            gen_text+=(word + ('\n' if i % 20 == 19 else ' '))

            if i % log_interval == 0:

                print('| Generated {}/{} words'.format(i, words))

print(gen_text)