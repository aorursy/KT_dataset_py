# implemented using https://github.com/pytorch/examples/tree/master/word_language_model



import torch

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import math

import nltk

import string



torch.manual_seed(1111)

if torch.cuda.is_available():

        print("WARNING: You have a CUDA device, so you should probably run with --cuda")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#logging

!touch app.log

import logging

for handler in logging.root.handlers[:]:

    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.DEBUG,filename='./app.log', filemode='w',format='%(process)d::%(asctime)s::%(message)s')
args = {

    "model":"LSTM",#RNN_TANH, RNN_RELU, LSTM, GRU                         

    "emsize":300,

    "nhid":300,

    "nlayers":3,

    "lr":20,

    "clip":0.25,

    "epochs":40, # upper epoch limit

    "batch_size":5,

    "bptt":40,#seq length

    "dropout":0.2,#dropout applied to layers (0 = no dropout)

    "tied":True,#'tie the word embedding and softmax weights'

    "seed":111,

    "log_interval":200,

    "save":"model.pt"

}

songdata = pd.read_csv("../input/songdata.csv")

songdata.head()

display(songdata.groupby("artist").count().sort_values(by=['song'],ascending=False))

subset=songdata[songdata['artist'].isin(["ABBA","Donna Summer","Gordon Lightfoot",

                                  "Rolling Stones","Bob Dylan", "Iggy Pop","The Beatles",

                                        "Cher","Bon Jovi","Michael Jackson","Green Day",

                                        "Red Hot Chili Peppers","Aerosmith","Paul McCartney",

                                        "Elvis Presley","Robbie Williams","Backstreet Boys",

                                        "Queen","Mariah Carey"])]

subset.head()
import torch.nn as nn

from torch import optim



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

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden



    def init_hidden(self, bsz):

        weight = next(self.parameters())

        if self.rnn_type == 'LSTM':

            return (weight.new_zeros(self.nlayers, bsz, self.nhid),

                    weight.new_zeros(self.nlayers, bsz, self.nhid))

        else:

            return weight.new_zeros(self.nlayers, bsz, self.nhid)
from sklearn.model_selection import train_test_split



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

    def __init__(self, dataframe):

        self.dictionary = Dictionary() 

        lyrics = dataframe['text'].apply(self.pre_process)

        train, test, y_train, y_test = train_test_split(lyrics, dataframe['artist'], test_size=0.15, random_state=13)

        train, val, y_train, y_val = train_test_split(train, y_train, test_size=0.15, random_state=13)

        self.train = self.tokenize(train.str.cat(sep=' end_song '))

        self.valid = self.tokenize(val.str.cat(sep=' end_song '))

        self.test = self.tokenize(test.str.cat(sep=' end_song '))

        self.train_raw = train.str.cat(sep=' end_song ')

        self.valid_raw = val.str.cat(sep=' end_song ')

        self.test_raw = test.str.cat(sep=' end_song ')

        

    def pre_process(self,text):

        text = text.replace("\r"," ")

        text = text.replace("\n"," ")

        text = text.lower()

        table = str.maketrans('', '', '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~')

        text = text.translate(table)

        return(text)

    

    def tokenize(self, string):

        tokens = 0

        words = nltk.word_tokenize(string)

        tokens += len(words)

        for word in words:

            self.dictionary.add_word(word)



        # Tokenize file content

        ids = torch.LongTensor(tokens)

        token = 0

        words = nltk.word_tokenize(string)

        for word in words:

            ids[token] = self.dictionary.word2idx[word]

            token += 1



        return ids
###############################################################################

# Load data

###############################################################################

subset=songdata[songdata['artist'].isin(["ABBA","Donna Summer","Gordon Lightfoot",

                                  "Rolling Stones","Bob Dylan", "Iggy Pop","The Beatles",

                                        "Cher","Bon Jovi","Michael Jackson","Green Day",

                                        "Red Hot Chili Peppers","Aerosmith","Paul McCartney",

                                        "Elvis Presley","Robbie Williams","Backstreet Boys",

                                        "Queen","Mariah Carey"])]

#songdata.groupby("artist").count().sort_values(by=['song'],ascending=False)

corpus = Corpus(subset)
# Starting from sequential data, batchify arranges the dataset into columns.

# For instance, with the alphabet as the sequence and batch size 4, we'd get

# ┌ a g m s ┐

# │ b h n t │

# │ c i o u │

# │ d j p v │

# │ e k q w │

# └ f l r x ┘.

# These columns are treated as independent by the model, which means that the

# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient

# batch processing.



def batchify(data, bsz):

    # Work out how cleanly we can divide the dataset into bsz parts.

    nbatch = data.size(0) // bsz

    print("batch_size:"+str(bsz))

    print("batches:"+str(nbatch))

    

    # Trim off any extra elements that wouldn't cleanly fit (remainders).

    data = data.narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.

    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)



eval_batch_size = 10

train_data = batchify(corpus.train, args['batch_size'])

val_data = batchify(corpus.valid, eval_batch_size)

test_data = batchify(corpus.test, eval_batch_size)
###############################################################################

# Build the model

###############################################################################



ntokens = len(corpus.dictionary)

model = RNNModel(args['model'], ntokens, args['emsize'], args['nhid'], args['nlayers'], args['dropout'], args['tied']).to(device)



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

    seq_len = min(args['bptt'], len(source) - 1 - i)

    data = source[i:i+seq_len]

    target = source[i+1:i+1+seq_len].view(-1)

    return data, target





def evaluate(data_source):

    # Turn on evaluation mode which disables dropout.

    model.eval()

    total_loss = 0.

    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():

        for i in range(0, data_source.size(0) - 1, args['bptt']):

            data, targets = get_batch(data_source, i)

            output, hidden = model(data, hidden)

            output_flat = output.view(-1, ntokens)

            total_loss += len(data) * criterion(output_flat, targets).item()

            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']

    

def train(epoch):

    # Turn on training mode which enables dropout.

    model.train()

    total_loss = 0.

    start_time = time.time()

    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(args['batch_size'])

    cur_loss = 0

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args['bptt'])):

        data, targets = get_batch(train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.

        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        hidden = repackage_hidden(hidden)

        

        # Reset the gradient after every epoch. 

        #optimizer.zero_grad()      

        model.zero_grad()

        

        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()

        

        # Optimizer take a step and update the weights.

        #optimizer.step()

        

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])

        for p in model.parameters():

            p.data.add_(-lr, p.grad.data)

        

        total_loss += loss.item()



        if batch % args['log_interval'] == 0 and batch > 0:

            cur_loss = total_loss / args['log_interval']

            elapsed = time.time() - start_time

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '

                    'loss {:5.2f} | ppl {:8.2f}'.format(

                epoch, batch, len(train_data) // args['bptt'], lr,

                elapsed * 1000 / args['log_interval'], cur_loss, 2**(cur_loss)))

            logging.debug('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '

                    'loss {:5.2f} | ppl {:8.2f}'.format(

                epoch, batch, len(train_data) // args['bptt'], lr,

                elapsed * 1000 / args['log_interval'], cur_loss, 2**(cur_loss)))

            total_loss = 0



            start_time = time.time()

    for name, param in model.named_parameters():

        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    writer.add_scalar('Train/Loss', cur_loss, epoch)

from tensorboardX import SummaryWriter

!rm -rf logs

!mkdir logs

writer = SummaryWriter('./logs')



# Loop over epochs.

lr = args['lr']

best_val_loss = None



# Initialize the optimizer

#learning_rate = args['lr']

#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)



# At any point you can hit Ctrl + C to break out of training early.

try:



    for epoch in range(1, args['epochs']+1):

        epoch_start_time = time.time()

        train(epoch)

        

        val_loss = evaluate(val_data)

        writer.add_scalar('Eval/Loss', val_loss, epoch)

        print('-' * 89)

        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '

                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),

                                           val_loss, 2**(val_loss)))

        print('-' * 89)

        logging.debug('-' * 89)

        logging.debug('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '

                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),

                                           val_loss, 2**(val_loss)))

        logging.debug('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.

        if not best_val_loss or val_loss < best_val_loss:

            with open(args['save'], 'wb') as f:

                torch.save(model, f)

            best_val_loss = val_loss

        else:

            # Anneal the learning rate if no improvement has been seen in the validation dataset.

            lr /= 4.0

        #if lr < 0.5:

            #lr=0.5

except KeyboardInterrupt:

    print('-' * 89)

    print('Exiting from training early')

    logging.debug('-' * 89)

    logging.debug('Exiting from training early')

    

writer.close()



# Load the best saved model.

with open(args['save'], 'rb') as f:

    model = torch.load(f)

    # after load the rnn params are not a continuous chunk of memory

    # this makes them a continuous chunk, and will speed up forward pass

    model.rnn.flatten_parameters()



# Run on test data.

test_loss = evaluate(test_data)

print('=' * 89)

print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(

    test_loss, 2**(test_loss)))

print('=' * 89)

logging.debug('=' * 89)

logging.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(

    test_loss, 2**(test_loss)))

logging.debug('=' * 89)
generate_args={

    "temperature": 1, #temperature - higher will increase diversity

    "words":200, #number of words to generate

    "outf":"generated.txt",

    "log_interval":30,

}



with open(args['save'], 'rb') as f:

    model = torch.load(f).to(device)

model.eval()

seed_word = "happy"

seed=torch.LongTensor(1,1).to(device)

seed[0]=corpus.dictionary.word2idx[seed_word]

hidden = model.init_hidden(1)

#input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

input = seed

with open(generate_args['outf'], 'w') as outf:

    outf.write(seed_word + ' ')

    with torch.no_grad():  # no tracking history

        for i in range(generate_args['words']):

            output, hidden = model(input, hidden)

            word_weights = output.squeeze().div(generate_args['temperature']).exp().cpu()

            word_idx = torch.multinomial(word_weights, 1)[0]

            input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            

            outf.write(word + ' ')



            if i % generate_args['log_interval'] == 0:

                print('| Generated {}/{} words'.format(i, generate_args['words']))
!cat generated.txt
### For posting logs to tensorboard  ##

# At first in settings, Make sure that Internet option is set to "Internet Connected"

# After executing this cell, there will come a link below, open that to view your tensor-board

'''

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip 

!unzip -o ngrok-stable-linux-amd64.zip



LOG_DIR = './logs' # Here you have to put your log directory

get_ipython().system_raw(

    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

    .format(LOG_DIR)

)

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \

    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

'''