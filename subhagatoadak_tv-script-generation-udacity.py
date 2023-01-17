import os

for dirname in os.walk('/kaggle/input'):

    print(dirname)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "/kaggle/input/scripts/helper.py", dst = "../working/helper.py")

copyfile(src = "/kaggle/input/scripts/problem_unittests.py", dst = "../working/problem_unittests.py")



# load in data

import helper

data_dir = '/kaggle/input/tvdata/Seinfeld_Scripts.txt'

text = helper.load_data(data_dir)
view_line_range = (0, 10)



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

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
import problem_unittests as tests

import numpy as np

from collections import Counter



def create_lookup_tables(text):

    """

    Create lookup tables for vocabulary

    :param text: The text of tv scripts split into words

    :return: A tuple of dicts (vocab_to_int, int_to_vocab)

    """

    # TODO: Implement Function

    count_of_text= Counter(text)

    permuted_vocab = sorted(count_of_text, key=count_of_text.get, reverse=True)

    

    # create int_to_vocab dictionaries

    int_to_vocab = {ii: word for ii, word in enumerate(permuted_vocab)}

    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    

    return (vocab_to_int, int_to_vocab)





"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_create_lookup_tables(create_lookup_tables)
def token_lookup():

    """

    Generate a dict to turn punctuation into a token.

    :return: Tokenized dictionary where the key is the punctuation and the value is the token

    """

    # TODO: Implement Function

    token_dict = dict()

    token_dict["."] = "||period||"

    token_dict[","] = "||comma||"

    token_dict["\""] = "||quotationmark||"

    token_dict[";"] = "||semicolon||"

    token_dict["!"] = "||exclamationmark||"

    token_dict["?"] = "||questionmark||"

    token_dict["("] = "||lparentheses||"

    token_dict[")"] = "||rparentheses||"

    token_dict["-"] = "||dash||"

    token_dict["\n"] = "||return||"

    

    return token_dict



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_tokenize(token_lookup)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

# pre-process training data

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import helper

import problem_unittests as tests



int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
print("-"*50)

print(int_text[:10])

print("-"*50)

'''

print(vocab_to_int)

print("-"*50)

print(int_to_vocab)

print("-"*50)

print(token_dict)

'''
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import torch



# Check for a GPU

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('No GPU found. Please use a GPU to train your neural network.')
train_on_gpu
from torch.utils.data import TensorDataset, DataLoader





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

    # only full batches

    words = words[:n_batches*batch_size]

    

    # TODO: Implement function    

    features, targets = [], []



    for idx in range(0, (len(words) - sequence_length) ):

        features.append(words[idx : idx + sequence_length])

        targets.append(words[idx + sequence_length])   

        

    #print(features)

    #print(targets)



    data = TensorDataset(torch.from_numpy(np.asarray(features)), torch.from_numpy(np.asarray(targets)))

    data_loader = torch.utils.data.DataLoader(data, shuffle=False , batch_size = batch_size)



    # return a dataloader

    return data_loader



# there is no test for this function, but you are encouraged to create

# print statements and tests of your own

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

        

        # set class variables

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        # define model layers

        

        # embedding and LSTM layers

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

                

        # linear layer

        self.fc = nn.Linear(hidden_dim, output_size)

    

    

    def forward(self, nn_input, hidden):

        """

        Forward propagation of the neural network

        :param nn_input: The input to the neural network

        :param hidden: The hidden state        

        :return: Two Tensors, the output of the neural network and the latest hidden state

        """

        # TODO: Implement function   

        batch_size = nn_input.size(0)



        # embeddings and lstm_out

        embeds = self.embed(nn_input)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        

        # dropout and fully-connected layer

        output = self.fc(lstm_out)

        

        # reshape to be batch_size first

        output = output.view(batch_size, -1, self.output_size)

        out = output[:, -1] # get last batch of labels       

        # return one batch of output word scores and the hidden state

        return out, hidden

       

    

    

    

    def init_hidden(self, batch_size):

        '''

        Initialize the hidden state of an LSTM/GRU

        :param batch_size: The batch_size of the hidden state

        :return: hidden state of dims (n_layers, batch_size, hidden_dim)

        '''

        # Implement function

        

        # initialize hidden state with zero weights, and move to GPU if available

      # initialize hidden state with zero weights, and move to GPU if available

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_rnn(RNN, train_on_gpu)
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

    # perform backpropagation and optimization

    hid = tuple([each.data for each in hidden])



    rnn.zero_grad()

    

    if(train_on_gpu):

        inp, target = inp.cuda(), target.cuda()



    # get the output from the model

    output, hid = rnn(inp, hid)



    loss = criterion(output, target)

    loss.backward()

    nn.utils.clip_grad_norm_(rnn.parameters(), 5)

    optimizer.step()



    return loss.item(), hid



# Note that these tests aren't completely extensive.

# they are here to act as general checks on the expected outputs of your functions

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""



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

sequence_length = 6  # of words in a sequence

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

embedding_dim = 256

# Hidden Dimension

hidden_dim = 256

# Number of RNN Layers

n_layers = 2



# Show stats for every n number of batches

show_every_n_batches = 500
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""



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

helper.save_model('./save/trained_rnn', trained_rnn)

print('Model Trained and Saved')
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import torch

import helper

import problem_unittests as tests



_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

trained_rnn = helper.load_model('./save/trained_rnn')
"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

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

        

        p = p.to('cpu') # move to cpu

         

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

        if train_on_gpu:

            current_seq = current_seq.cpu()

        current_seq = np.roll(current_seq, -1, 1)

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
# run the cell multiple times to get different results!

gen_length = 400 # modify the length to your preference

prime_word = 'jerry' # name for starting the script



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

pad_word = helper.SPECIAL_WORDS['PADDING']

generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)

print(generated_script)
# save script to a text file

f =  open("generated_script_1.txt","w")

f.write(generated_script)

f.close()