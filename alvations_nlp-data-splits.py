import torch

from sklearn.model_selection import train_test_split



X = torch.rand(10, 3) # 10 data points, with 3 dimensions/features each.

Y = torch.rand(10)    # 10 corresponding y values for the data points.
print(X.shape)

X
print(Y.shape) 

Y
test_size = 0.2 # If we want to hold-out 20% of our data as test-set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
print(X_train.shape)  # 8 data points with 3 dimensions each

print(Y_train.shape)  # 8 y-values corresponding to the 8 data points
print(X_test.shape)  # 2 data points with 3 dimensions each

print(Y_test.shape)  # 2 y-values corresponding to the 2 data points
# To get the same 20% validation proportion as the test size, 

# you can use this idiom to get the valid_size

valid_size = 0.2 / (1 - test_size) 

# We split the validation set from the training set.

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=valid_size)
print(X_train.shape)  # 6 data points with 3 dimensions each

print(Y_train.shape)  # 6 y-values corresponding to the 6 data points
print(X_valid.shape)  # 2 data points with 3 dimensions each

print(Y_valid.shape)  # 2 y-values corresponding to the 2 data points
def train_valid_test_split(X, Y, valid_size, test_size, random_state=42):

    """

    Extending sklearn's train_test_split and using only floating point test_size

    https://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html

    """

    assert valid_size < 1 and test_size < 1 

    X_train_valid, X_test, Y_train_valid, Y_test =  train_test_split(X, Y, test_size=test_size, random_state=random_state)

    _valid_size = valid_size / (1 - test_size)

    X_train, X_valid, Y_train, Y_valid =  train_test_split(X_train_valid, Y_train_valid, 

                                                           test_size=_valid_size, random_state=random_state)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
X = torch.rand(100, 3) # 100 data points, with 3 dimensions/features each.

Y = torch.rand(100)    # 100 corresponding y values for the data points.



X_train, X_valid, X_test, Y_train, Y_valid, Y_test = train_valid_test_split(X, Y, valid_size=0.30, test_size=0.20)
print('Training set:\t',   X_train.shape, Y_train.shape)

print('Validation set:\t', X_valid.shape, Y_valid.shape)

print('Test set:\t',       X_test.shape, Y_test.shape)
from tqdm import tqdm



import torch

from torch import nn, optim
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(15, 10)})
# Here's some torch randomization magic to fix the outputs so that 

# we'll see the same outputs no matter how we run this notebook.



# For CPU version.

torch.backends.mkl.deterministic = True

# For CUDA version

##torch.backends.cuda.deterministic = True



torch.manual_seed(42)
# Create some random data.

X = torch.rand(100, 3) # 100 data points, with 3 dimensions/features each.

Y = torch.rand(100)    # 100 corresponding y values for the data points.



X_train, X_valid, X_test, Y_train, Y_valid, Y_test = train_valid_test_split(X, Y, valid_size=0.20, test_size=0.10)

print('Training set:\t',   X_train.shape, Y_train.shape)

print('Validation set:\t', X_valid.shape, Y_valid.shape)

print('Test set:\t',       X_test.shape, Y_test.shape)
num_data, input_dim = X_train.shape

num_data, output_dim = Y_train.unsqueeze(1).shape



print('No. of training data:', num_data)

print('Input dimension:', input_dim)

print('Output dimension:', output_dim)
# Step 1: Initialization. 

# Note: When using PyTorch a lot of the manual weights

#       initialization is done automatically when we define

#       the model (aka architecture)

model = nn.Sequential(

            nn.Linear(input_dim, output_dim), 

            nn.Sigmoid())

criterion = nn.MSELoss() 

learning_rate = 1.0

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 200



losses = []



for i in tqdm(range(num_epochs)):

    # Reset the gradient after every epoch. 

    optimizer.zero_grad() 

    # Step 2: Foward Propagation

    predictions = model(X_train)

    

    # Step 3: Back Propagation 

    # Calculate the cost between the predictions and the truth.

    loss_this_epoch = criterion(predictions, Y_train)

    # Note: The neat thing about PyTorch is it does the 

    #       auto-gradient computation, no more manually defining

    #       derivative of functions and manually propagating

    #       the errors layer by layer.

    loss_this_epoch.backward()

    

    # Step 4: Optimizer take a step. 

    # Note: Previously, we have to manually update the 

    #       weights of each layer individually according to the

    #       learning rate and the layer delta. 

    #       PyTorch does that automatically =)

    optimizer.step()

    

    # Log the loss value as we proceed through the epochs.

    losses.append(loss_this_epoch.data.item())

    

# Visualize the losses

plt.plot(losses)

plt.show()
# Create some random data.

X = torch.rand(100, 3) # 100 data points, with 3 dimensions/features each.

Y = torch.rand(100)    # 100 corresponding y values for the data points.



X_train, X_valid, X_test, Y_train, Y_valid, Y_test = train_valid_test_split(X, Y, valid_size=0.20, test_size=0.10)

num_data, input_dim = X_train.shape

num_data, output_dim = Y_train.unsqueeze(1).shape



print('No. of training data:', num_data)

print('Input dimension:', input_dim)

print('Output dimension:', output_dim)
# Step 1: Initialization. 

# Note: When using PyTorch a lot of the manual weights

#       initialization is done automatically when we define

#       the model (aka architecture)

model = nn.Sequential(

            nn.Linear(input_dim, output_dim), 

            nn.Sigmoid())

criterion = nn.MSELoss() 

learning_rate = 1.0

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 250



training_losses, validation_losses = [], []
for i in tqdm(range(num_epochs)):

    # Reset the gradient after every epoch. 

    optimizer.zero_grad() 

    # Step 2: Foward Propagation

    predictions = model(X_train)

    

    # Step 3: Back Propagation 

    # Calculate the cost between the predictions and the truth.

    loss_this_epoch = criterion(predictions, Y_train)

    # Note: The neat thing about PyTorch is it does the 

    #       auto-gradient computation, no more manually defining

    #       derivative of functions and manually propagating

    #       the errors layer by layer.

    loss_this_epoch.backward()

    

    # Step 4: Optimizer take a step. 

    # Note: Previously, we have to manually update the 

    #       weights of each layer individually according to the

    #       learning rate and the layer delta. 

    #       PyTorch does that automatically =)

    optimizer.step()  # The model has been updated in this epoch.

    

    # Log the loss value as we proceed through the epochs.

    training_losses.append(loss_this_epoch.data.item())

    # Bonus: On top of just logging the training loss. 

    # First, use the new model (after optimizer updates the weights)

    # And put it through a forward propagation without gradient.

    with torch.no_grad():

        valid_predictions = model(X_valid)

        valid_loss = criterion(valid_predictions, Y_valid)

        validation_losses.append(valid_loss)



# Visualize the losses

plt.plot(training_losses, label='Train loss')

plt.plot(validation_losses, label='Valid loss')

plt.legend(loc='upper right')

plt.show()
def repeat_experiments(X_train, Y_train, X_valid, Y_valid):

    

    # Step 1: Initialization. 

    # Note: When using PyTorch a lot of the manual weights

    #       initialization is done automatically when we define

    #       the model (aka architecture)

    model = nn.Sequential(

                nn.Linear(input_dim, output_dim), 

                nn.Sigmoid())

    criterion = nn.MSELoss() 

    learning_rate = 1.0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 250



    training_losses, validation_losses = [], []

    

    for i in tqdm(range(num_epochs)):

        # Reset the gradient after every epoch. 

        optimizer.zero_grad() 

        # Step 2: Foward Propagation

        predictions = model(X_train)



        # Step 3: Back Propagation 

        # Calculate the cost between the predictions and the truth.

        loss_this_epoch = criterion(predictions, Y_train)

        # Note: The neat thing about PyTorch is it does the 

        #       auto-gradient computation, no more manually defining

        #       derivative of functions and manually propagating

        #       the errors layer by layer.

        loss_this_epoch.backward()



        # Step 4: Optimizer take a step. 

        # Note: Previously, we have to manually update the 

        #       weights of each layer individually according to the

        #       learning rate and the layer delta. 

        #       PyTorch does that automatically =)

        optimizer.step()



        # Log the loss value as we proceed through the epochs.

        training_losses.append(loss_this_epoch.data.item())

        # Bonus: On top of just logging the training loss. 

        # First, use the new model (after optimizer updates the weights)

        # And put it through a forward propagation without gradient.

        with torch.no_grad():

            valid_predictions = model(X_valid)

            valid_loss = criterion(valid_predictions, Y_valid)

            validation_losses.append(valid_loss)

        

    # Visualize the losses

    plt.plot(training_losses, label='Train loss')

    plt.plot(validation_losses, label='Valid loss')

    plt.legend(loc='upper right')

    plt.show()



# Create some random data.

X = torch.rand(100, 3) # 100 data points, with 3 dimensions/features each.

Y = torch.rand(100)    # 100 corresponding y values for the data points.



X_train, X_valid, X_test, Y_train, Y_valid, Y_test = train_valid_test_split(X, Y, valid_size=0.20, test_size=0.10)

for i in range(3):

    torch.manual_seed(i)

    repeat_experiments(X_train, Y_train, X_valid, Y_valid)
# IPython candies...

from IPython.display import Image

from IPython.core.display import HTML



from IPython.display import clear_output
from collections import namedtuple



import numpy as np

from tqdm import tqdm



import pandas as pd



from gensim.corpora import Dictionary



import torch

from torch import nn, optim, tensor, autograd

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader



from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



device = 'cuda' if torch.cuda.is_available() else 'cpu'
from nltk import word_tokenize, sent_tokenize 



import os

import requests

import io #codecs





# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf

if os.path.isfile('language-never-random.txt'):

    with io.open('language-never-random.txt', encoding='utf8') as fin:

        text = fin.read()

else:

    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"

    text = requests.get(url).content.decode('utf8')

    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:

        fout.write(text)

        

# Tokenize the text.

tokenized_text = [list(map(str.lower, word_tokenize(sent))) 

                  for sent in sent_tokenize(text)]
class KilgariffDataset(nn.Module):

    def __init__(self, texts):

        self.texts = texts

        

        # Initialize the vocab 

        special_tokens = {'<pad>': 0, '<unk>':1, '<s>':2, '</s>':3}

        self.vocab = Dictionary(texts)

        self.vocab.patch_with_special_tokens(special_tokens)

        

        # Keep track of the vocab size.

        self.vocab_size = len(self.vocab)

        

        # Keep track of how many data points.

        self._len = len(texts)

        

        # Find the longest text in the data.

        self.max_len = max(len(txt) for txt in texts) 

        

    def __getitem__(self, index):

        vectorized_sent = self.vectorize(self.texts[index])

        x_len = len(vectorized_sent)

        # To pad the sentence:

        # Pad left = 0; Pad right = max_len - len of sent.

        pad_dim = (0, self.max_len - len(vectorized_sent))

        vectorized_sent = F.pad(vectorized_sent, pad_dim, 'constant')

        return {'x':vectorized_sent[:-1], 

                'y':vectorized_sent[1:], 

                'x_len':x_len}

    

    def __len__(self):

        return self._len

    

    def vectorize(self, tokens, start_idx=2, end_idx=3):

        """

        :param tokens: Tokens that should be vectorized. 

        :type tokens: list(str)

        """

        # See https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary.doc2idx 

        # Lets just cast list of indices into torch tensors directly =)

        

        vectorized_sent = [start_idx] + self.vocab.doc2idx(tokens) + [end_idx]

        return torch.tensor(vectorized_sent)

    

    def unvectorize(self, indices):

        """

        :param indices: Converts the indices back to tokens.

        :type tokens: list(int)

        """

        return [self.vocab[i] for i in indices]
# Initialize the dataset object.

num_sents = len(tokenized_text)



# 20% of test.

start_of_test_split = -1 * int(0.2 * num_sents)

# 20% of valid.

start_of_valid_split = -1 * int(0.4 * num_sents)



train_text = tokenized_text[:start_of_valid_split]

valid_text = tokenized_text[start_of_valid_split:start_of_test_split]

test_text = tokenized_text[start_of_test_split:]



train_dataset = KilgariffDataset(train_text)

valid_dataset = KilgariffDataset(valid_text)

test_dataset = KilgariffDataset(test_text)



# Another way to split the dataset is using 

# https://pytorch.org/docs/master/data.html#torch.utils.data.SubsetRandomSampler



# When training, take a batch of size 15.

batch_size = 15

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# When validating and testing, lets do the whole validation set as a batch

# When validate/test set is small, just fit in the whole validation/test set as a one batch.

valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=True) 

test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.8):

        super(LanguageModel, self).__init__()



        # Initialize the embedding layer with the 

        # - size of input (i.e. no. of words in input vocab)

        # - no. of hidden nodes in the embedding layer

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        

        # Initialize the GRU with the 

        # - size of the input (i.e. embedding layer)

        # - size of the hidden layer 

        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)

        

        # Initialize the "classifier" layer to map the RNN outputs

        # to the vocabulary. Remember we need to -1 because the 

        # vectorized sentence we left out one token for both x and y:

        # - size of hidden_size of the GRU output.

        # - size of vocabulary

        self.classifier = nn.Linear(hidden_size, vocab_size)

        

        # Global dropout.

        self.dropout = dropout

        

    def forward(self, inputs, use_softmax=False, hidden=None):

        # Look up for the embeddings for the input word indices.

        embedded = self.embedding(inputs)

        ##embedded = F.dropout(embedded, self.dropout)

        # Put the embedded inputs into the GRU.

        output, hidden = self.gru(embedded, hidden)

        

        # Matrix manipulation magic.

        batch_size, sequence_len, hidden_size = output.shape

        # Technically, linear layer takes a 2-D matrix as input, so more manipulation...

        output = output.contiguous().view(batch_size * sequence_len, hidden_size)

        # Apply dropout.

        output = F.dropout(output, self.dropout)

        # Put it through the classifier

        # And reshape it to [batch_size x sequence_len x vocab_size]

        output = self.classifier(output).view(batch_size, sequence_len, -1)

        

        return (F.softmax(output,dim=2), hidden) if use_softmax else (output, hidden)

        


# Training routine.

def train(num_epochs, train_dataloader, valid_dataloader, model, criterion, optimizer):

    training_losses, validation_losses = [], []

    plt.ion()

    for _e in range(num_epochs):

        for batch in tqdm(train_dataloader):

            # Zero gradient.

            optimizer.zero_grad()

            x = batch['x'].to(device)

            x_len = batch['x_len'].to(device)

            y = batch['y'].to(device)

            # Feed forward. 

            output, hidden = model(x, use_softmax=False)

            # Compute loss:

            # Shape of the `output` is [batch_size x sequence_len x vocab_size]

            # Shape of `y` is [batch_size x sequence_len]

            # CrossEntropyLoss expects `output` to be [batch_size x vocab_size x sequence_len]

            _, prediction = torch.max(output, dim=2)

            loss = criterion(output.permute(0, 2, 1), y)

            loss.backward()

            optimizer.step()

            training_losses.append(loss.float().data)

            

            with torch.no_grad():

                valid_x = next(iter(valid_dataloader))['x'].to(device)

                valid_y = next(iter(valid_dataloader))['y'].to(device)

                # Forward propagation on validation set. 

                # Set the model to eval() mode, no gradient, no dropout.

                model.eval()

                output, hidden = model(valid_x, use_softmax=False)

                valid_loss = criterion(output.permute(0, 2, 1), valid_y)

                validation_losses.append(valid_loss.float().data)

                # Rest the model to train() mode.

                model.train()

                

        # Visualize the losses

        clear_output(wait=True)

        plt.plot(training_losses, label='Train loss')

        plt.plot(validation_losses, label='Valid loss')

        plt.legend(loc='upper right')

        plt.pause(0.05)

def initialize_data_model_optim_loss(hyperparams):

    # Loss function.

    criterion = hyperparams.loss_func(ignore_index=train_dataset.vocab.token2id['<pad>'], 

                                      reduction='mean')



    # Model.

    model = LanguageModel(len(train_dataset.vocab), hyperparams.embed_size, 

                      hyperparams.hidden_size, hyperparams.num_layers, dropout=hyperparams.dropout).to(device)



    # Optimizer.

    optimizer = hyperparams.optimizer(model.parameters(), lr=hyperparams.learning_rate)

    

    return model, optimizer, criterion
# Set some hyperparameters.

device = 'cuda' if torch.cuda.is_available() else 'cpu'



_hyper = ['embed_size', 'hidden_size', 'num_layers',

          'loss_func', 'learning_rate', 'optimizer', 'dropout']

Hyperparams = namedtuple('Hyperparams', _hyper)





hyperparams = Hyperparams(embed_size=250, hidden_size=250, num_layers=1,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=0.03, optimizer=optim.Adam, dropout=0.5)
# Initialize the dataset object.

num_sents = len(tokenized_text)



# 20% of test.

start_of_test_split = -1 * int(0.2 * num_sents)

# 20% of valid.

start_of_valid_split = -1 * int(0.4 * num_sents)



train_text = tokenized_text[:start_of_valid_split]

valid_text = tokenized_text[start_of_valid_split:start_of_test_split]

test_text = tokenized_text[start_of_test_split:]



train_dataset = KilgariffDataset(train_text)

valid_dataset = KilgariffDataset(valid_text)

test_dataset = KilgariffDataset(test_text)



# When training, take a batch of size 50.

batch_size = 100

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# When validating and testing, lets do the whole validation set as a batch

valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
# Experiment 1. 



hyperparams = Hyperparams(embed_size=128, hidden_size=128, num_layers=3,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=3e-5, optimizer=optim.Adam, dropout=0.5)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
model # 140th epoch



torch.save(model, '140-epoch.pth')
test_x = next(iter(test_dataloader))['x'].to(device)

test_y = next(iter(test_dataloader))['y'].to(device)

# Forward propagation on validation set. 

# Set the model to eval() mode, no gradient, no dropout.

model.eval()

output, hidden = model(test_x, use_softmax=False)

print(criterion(output.permute(0, 2, 1), test_y)) # This is the cross-entropy loss score!! Perplexity = **** crossentropy =)







def generate_example(model, temperature=1.0, max_len=100, hidden_state=None):

    start_token, start_idx = '<s>', 2

    # Start state.

    inputs = torch.tensor(train_dataset.vocab.token2id[start_token]).unsqueeze(0).unsqueeze(0).to(device)



    sentence = [start_token]

    i = 0

    while i < max_len and sentence[-1] not in ['</s>', '<pad>']:

        i += 1

        

        embedded = model.embedding(inputs)

        output, hidden_state = model.gru(embedded, hidden_state)



        batch_size, sequence_len, hidden_size = output.shape

        output = output.contiguous().view(batch_size * sequence_len, hidden_size)    

        output = model.classifier(output).view(batch_size, sequence_len, -1).squeeze(0)

        #_, prediction = torch.max(F.softmax(output, dim=2), dim=2)

        

        word_weights = output.div(temperature).exp().cpu()

        if len(word_weights.shape) > 1:

            word_weights = word_weights[-1] # Pick the last word.    

        word_idx = torch.multinomial(word_weights, 1).view(-1)

        

        sentence.append(train_dataset.vocab[int(word_idx)])

        

        inputs = tensor([train_dataset.vocab.token2id[word] for word in sentence]).unsqueeze(0).to(device)

    print(' '.join(sentence))
generate_example(model, max_len=20) # 140th epoch
hyperparams = Hyperparams(embed_size=64, hidden_size=64, num_layers=3,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=3e-5, optimizer=optim.Adam, dropout=0.5)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
train(50, train_dataloader, valid_dataloader, model, criterion, optimizer)
train(50, train_dataloader, valid_dataloader, model, criterion, optimizer)
hyperparams = Hyperparams(embed_size=64, hidden_size=64, num_layers=6,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=3e-3, optimizer=optim.Adam, dropout=0.7)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
hyperparams = Hyperparams(embed_size=64, hidden_size=64, num_layers=2,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=0.0003, optimizer=optim.Adam, dropout=0.5)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
hyperparams = Hyperparams(embed_size=64, hidden_size=64, num_layers=2,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=0.00003, optimizer=optim.Adam, dropout=0.8)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
# When training, take a batch of size 50.

batch_size = 200

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# When validating and testing, lets do the whole validation set as a batch

valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
len(train_dataset)
hyperparams = Hyperparams(embed_size=64, hidden_size=64, num_layers=3,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=3e-5, optimizer=optim.Adam, dropout=0.7)



model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams)



train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
train(150, train_dataloader, valid_dataloader, model, criterion, optimizer)
len(train)