# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os



import pandas as pd

df = pd.read_csv('/kaggle/input/massive-stock-news-analysis-db-for-nlpbacktests/raw_partner_headlines.csv')
df.head()
news = []

for i, j in df.iterrows():

    news.append(j['headline'])

    

print(len(news))
news[:1]
len(news)
news = news[:109233]
len(news)
os.path.join('/kaggle/working', 'finance_news.txt')
f = open('/kaggle/working/finance_news.txt', 'w')

f.write('\n'.join(news))

f.close()
import os

import pickle

import torch





SPECIAL_WORDS = {'PADDING': '<PAD>'}





def load_data(path):

    """

    Load Dataset from File

    """

    input_file = os.path.join(path)

    with open(input_file, "r") as f:

        data = f.read()



    return data





def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):

    """

    Preprocess Text Data

    """

    text = load_data(dataset_path)

    

    # Ignore notice, since we don't use it for analysing the data

    text = text[81:]



    token_dict = token_lookup()

    for key, token in token_dict.items():

        text = text.replace(key, ' {} '.format(token))



    text = text.lower()

    text = text.split()



    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))

    int_text = [vocab_to_int[word] for word in text]

    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))





def load_preprocess():

    """

    Load the Preprocessed Training data and return them in batches of <batch_size> or less

    """

    return pickle.load(open('preprocess.p', mode='rb'))





def save_model(filename, decoder):

    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'

    torch.save(decoder, save_filename)





def load_model(filename):

    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'

    return torch.load(save_filename)
data_dir = '/kaggle/working/finance_news.txt'

text = load_data(data_dir)
view_line_range = (0, 10)



import numpy as np



print('Dataset Stats')

print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))



lines = text.split('\n')

print('Number of lines: {}'.format(len(lines)))

word_count_line = [len(line.split()) for line in lines]

print('Average number of words in each line: {}'.format(np.average(word_count_line)))



print()

print('The lines {} to {}:'.format(*view_line_range))

print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
from collections import Counter



def create_lookup_tables(text):

    """

    Create lookup tables for vocabulary

    :param text: The text of tv scripts split into words

    :return: A tuple of dicts (vocab_to_int, int_to_vocab)

    """

    # TODO: Implement Function

    word_count = Counter(text)

    sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)

    int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}

    vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}

    

    # return tuple

    return (vocab_to_int, int_to_vocab)

def token_lookup():

    """

    Generate a dict to turn punctuation into a token.

    :return: Tokenized dictionary where the key is the punctuation and the value is the token

    """

    # TODO: Implement Function

    token = dict()

    token['.'] = '<PERIOD>'

    token[','] = '<COMMA>'

    token['"'] = 'QUOTATION_MARK'

    token[';'] = 'SEMICOLON'

    token['!'] = 'EXCLAIMATION_MARK'

    token['?'] = 'QUESTION_MARK'

    token['('] = 'LEFT_PAREN'

    token[')'] = 'RIGHT_PAREN'

    token['-'] = 'QUESTION_MARK'

    token['\n'] = 'NEW_LINE'

    return token

preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
train_on_gpu = torch.cuda.is_available()
from torch.utils.data import TensorDataset, DataLoader

import torch

import numpy as np





def batch_data(words, sequence_length, batch_size):

    """

    Batch the neural network data using DataLoader

    :param words: The word ids of the TV scripts

    :param sequence_length: The sequence length of each batch

    :param batch_size: The size of each batch; the number of sequences in a batch

    :return: DataLoader with batched data

    """

    # TODO: Implement function

    n_batches = len(words)//batch_size

    x, y = [], []

    words = words[:n_batches*batch_size]

    

    for ii in range(0, len(words)-sequence_length):

        i_end = ii+sequence_length        

        batch_x = words[ii:ii+sequence_length]

        x.append(batch_x)

        batch_y = words[i_end]

        y.append(batch_y)

    

    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))

    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

        

    

    # return a dataloader

    return data_loader

# test dataloader



test_text = range(50)

t_loader = batch_data(test_text, sequence_length=5, batch_size=10)



data_iter = iter(t_loader)

sample_x, sample_y = data_iter.next()



print(sample_x.shape)

print(sample_x)

print()

print(sample_y.shape)

print(sample_y)
import torch.nn as nn



class RNN(nn.Module):

    

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):

        """

        Initialize the PyTorch RNN Module

        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)

        :param output_size: The number of output dimensions of the neural network

        :param embedding_dim: The size of embeddings, should you choose to use them        

        :param hidden_dim: The size of the hidden layer outputs

        :param dropout: dropout to add in between LSTM/GRU layers

        """

        super(RNN, self).__init__()

        # TODO: Implement function

        

        # define embedding layer

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        

        # define lstm layer

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        

        

        # set class variables

        self.vocab_size = vocab_size

        self.output_size = output_size

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim

        self.n_layers = n_layers

        

        # define model layers

        self.fc = nn.Linear(hidden_dim, output_size)

    

    

    def forward(self, x, hidden):

        """

        Forward propagation of the neural network

        :param nn_input: The input to the neural network

        :param hidden: The hidden state        

        :return: Two Tensors, the output of the neural network and the latest hidden state

        """

        # TODO: Implement function   

        batch_size = x.size(0)

        x=x.long()

        

        # embedding and lstm_out 

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        

        # stack up lstm layers

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        

        # dropout, fc layer and final sigmoid layer

        out = self.fc(lstm_out)

        

        # reshaping out layer to batch_size * seq_length * output_size

        out = out.view(batch_size, -1, self.output_size)

        

        # return last batch

        out = out[:, -1]



        # return one batch of output word scores and the hidden state

        return out, hidden

    

    

    def init_hidden(self, batch_size):

        '''

        Initialize the hidden state of an LSTM/GRU

        :param batch_size: The batch_size of the hidden state

        :return: hidden state of dims (n_layers, batch_size, hidden_dim)

        '''

        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim

        weights = next(self.parameters()).data

        if(train_on_gpu):

            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 

                     weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                     weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        # initialize hidden state with zero weights, and move to GPU if available

        

        return hidden
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):

    """

    Forward and backward propagation on the neural network

    :param decoder: The PyTorch Module that holds the neural network

    :param decoder_optimizer: The PyTorch optimizer for the neural network

    :param criterion: The PyTorch loss function

    :param inp: A batch of input to the neural network

    :param target: The target output for the batch of input

    :return: The loss and the latest hidden state Tensor

    """

    

    # TODO: Implement Function

    

    # move data to GPU, if available

    if(train_on_gpu):

        rnn.cuda()

    

    # creating variables for hidden state to prevent back-propagation

    # of historical states 

    h = tuple([each.data for each in hidden])

    

    rnn.zero_grad()

    # move inputs, targets to GPU 

    inputs, targets = inp.cuda(), target.cuda()

    

    output, h = rnn(inputs, h)

    

    loss = criterion(output, targets)

    

    # perform backpropagation and optimization

    loss.backward()

    nn.utils.clip_grad_norm_(rnn.parameters(), 5)

    optimizer.step()



    # return the loss over a batch and the hidden state produced by our model

    return loss.item(), h

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):

    batch_losses = []

    

    rnn.train()



    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):

        

        # initialize hidden state

        hidden = rnn.init_hidden(batch_size)

        

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            

            # make sure you iterate over completely full batches, only

            n_batches = len(train_loader.dataset)//batch_size

            if(batch_i > n_batches):

                break

            

            # forward, back prop

            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          

            # record loss

            batch_losses.append(loss)



            # printing loss stats

            if batch_i % show_every_n_batches == 0:

                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(

                    epoch_i, n_epochs, np.average(batch_losses)))

                batch_losses = []



    # returns a trained rnn

    return rnn
# Data params

# Sequence Length

sequence_length = 10  # of words in a sequence

# Batch Size

batch_size = 128



# data loader - do not change

train_loader = batch_data(int_text, sequence_length, batch_size)
# Training parameters

# Number of Epochs

num_epochs = 10

# Learning Rate

learning_rate = 0.001



# Model parameters

# Vocab size

vocab_size = len(vocab_to_int)

# Output size

output_size = vocab_size

# Embedding Dimension

embedding_dim = 200

# Hidden Dimension

hidden_dim = 250

# Number of RNN Layers

n_layers = 2



# Show stats for every n number of batches

show_every_n_batches = 500
# create model and move to gpu if available

rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

if train_on_gpu:

    rnn.cuda()



# defining loss and optimization functions for training

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()



# training the model

trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)



# saving the trained model

save_model('./save/trained_rnn', trained_rnn)

print('Model Trained and Saved')
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import torch



_, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

trained_rnn = load_model('./save/trained_rnn')
import torch.nn.functional as F



def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):

    """

    Generate text using the neural network

    :param decoder: The PyTorch Module that holds the trained neural network

    :param prime_id: The word id to start the first prediction

    :param int_to_vocab: Dict of word id keys to word values

    :param token_dict: Dict of puncuation tokens keys to puncuation values

    :param pad_value: The value used to pad a sequence

    :param predict_len: The length of text to generate

    :return: The generated text

    """

    rnn.eval()

    

    # create a sequence (batch_size=1) with the prime_id

    current_seq = np.full((1, sequence_length), pad_value)

    current_seq[-1][-1] = prime_id

    predicted = [int_to_vocab[prime_id]]

    

    for _ in range(predict_len):

        if train_on_gpu:

            current_seq = torch.LongTensor(current_seq).cuda()

        else:

            current_seq = torch.LongTensor(current_seq)

        

        # initialize the hidden state

        hidden = rnn.init_hidden(current_seq.size(0))

        

        # get the output of the rnn

        output, _ = rnn(current_seq, hidden)

        

        # get the next word probabilities

        p = F.softmax(output, dim=1).data

        if(train_on_gpu):

            p = p.cpu() # move to cpu

         

        # use top_k sampling to get the index of the next word

        top_k = 5

        p, top_i = p.topk(top_k)

        top_i = top_i.numpy().squeeze()

        

        # select the likely next word index with some element of randomness

        p = p.numpy().squeeze()

        word_i = np.random.choice(top_i, p=p/p.sum())

        

        # retrieve that word from the dictionary

        word = int_to_vocab[word_i]

        predicted.append(word)     

        

        # the generated word becomes the next "current sequence" and the cycle can continue

        current_seq = np.roll(current_seq.cpu(), -1, 1)

        current_seq[-1][-1] = word_i

    

    gen_sentences = ' '.join(predicted)

    

    # Replace punctuation tokens

    for key, token in token_dict.items():

        ending = ' ' if key in ['\n', '(', '"'] else ''

        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)

    gen_sentences = gen_sentences.replace('\n ', '\n')

    gen_sentences = gen_sentences.replace('( ', '(')

    

    # return all the sentences

    return gen_sentences

gen_length = 50 # modify the length to your preference

prime_words = ['tesla'] # name for starting the script



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

for prime_word in prime_words:

    pad_word = SPECIAL_WORDS['PADDING']

    generated_script = generate(trained_rnn, vocab_to_int[prime_word], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)

    print(generated_script)