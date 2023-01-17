import numpy as np
with open('../input/sentiment-analysis-imdb/reviews.txt', 'r') as f:

    reviews = f.read()

with open('../input/sentiment-analysis-imdb/labels.txt', 'r') as f:

    labels = f.read()
print(reviews[:1000])

print(labels[:20])
# getting rid of punctuations

from string import punctuation

print(punctuation)



reviews = reviews.lower()

all_text = ''.join([c for c in reviews if c not in punctuation])
# Split by new lines

review_split = all_text.split('\n')

all_text = ' '.join(review_split)



# Create a list of words

words = all_text.split()
words[:30]
from collections import Counter



## Build dictionary that maps words to integer

counts = Counter(words)

vocab  = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii,  word in enumerate(vocab, 1)}



# use dict to tokenize each review in review_split

review_ints = []

for review in review_split:

    review_ints.append([vocab_to_int[word] for word in review.split()])
# testing implementation of above code

print('unique words :', len((vocab_to_int)))

print()

print('Tokenized words :', review_ints[:1])
# 1 = +ve and 0 = -ve

label_split = labels.split('\n')

encoded_labels = np.array([1 if label == 'positive' else 0 for label in label_split])
# To shape review in consistent sequence length

# getting rid of extremely short and lonh reviews

# padding/ truncating the remaining data so we have reviews of same length

# outlier review stats

review_lens = Counter([len(x) for x in  review_ints])

print('Zero length reviews: {}'.format(review_lens[0]))

print(' Max review length {}'.format(max(review_lens)))
# removing outliers

print('Number of reviews before removing outliers', len(review_ints))



# removing label and review with length 0

non_zero_idx = [ii for ii, review in enumerate(review_ints) if len(review) != 0]

review_ints = [review_ints[ii] for ii in non_zero_idx]

encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers', len(review_ints))
# too  remove very long sequence of data and regularizing sequence too.

# padding to make sure that all sequence lengths are of same size

def pad_feature(review_ints, seq_len):

    

    # getting correct rows and column size

    features = np.zeros((len(review_ints), seq_len), dtype=int)

    for i, row in enumerate(review_ints):

        features[i, -len(row):] = np.array(row)[:seq_len]

    

            

    return features
sequence_length = 200

features = pad_feature(review_ints, sequence_length)
# testing code above

assert len(features) == len(review_ints)

assert len(features[0]) == sequence_length

print(features[:30, :10])
from sklearn.model_selection import train_test_split
train_x, remain_data_x, train_y, remain_data_y = train_test_split(features, encoded_labels, train_size=0.8)
valid_x, test_x, valid_y, test_y = train_test_split(remain_data_x, remain_data_y, test_size= 0.5)
print('Training shape', train_x.shape)

print('Validation shape', valid_x.shape)

print('Testing shape', test_x.shape)
import torch

from torch.utils.data import TensorDataset, DataLoader
# Formatting data to be fed in dataloader to batch it

train_data = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))

valid_data = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))

test_data = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
# Batching with shuffle

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
# checking if gpu is availbale for training

train_on_gpu = torch.cuda.is_available()

if (train_on_gpu):

    print('Training on gpu')

else:

    print('Training on cpu')

from torch import nn, optim
# Model

class SentimentRNN(nn.Module):

    def __init__(self, vocab_size,output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        super().__init__()

    

        # Initializing model

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

    

        # define layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,

                            dropout=drop_prob, batch_first=True)

        # dropout layer

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

    

    def forward(self, x, hidden):

        batch_size = x.size(0)

        

        # embedding and lstm out

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

            

        # stack up the lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

            

        # Dropu out and FC

        out = self.dropout(lstm_out)

        out = self.fc(out)

        

        # sigmoid Function 

        sig_out = self.sig(out)

        

        # reshape to bacth_size first

        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        

        return sig_out, hidden

    

    def init_hidden(self, batch_size):

        '''Initializing hidden state'''

        weight = next(self.parameters()).data

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
vocab_size = len(vocab_to_int) +1 # +1 for 0 zero padding and out word tokens

output_size = 1

embedding_dim = 400

hidden_dim = 256

n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
# Initializing optimizer and criterion

lr = 0.001

optimizer = optim.Adam(net.parameters(), lr)

criterion = nn.BCELoss()
# training params



epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing



counter = 0

print_every = 100

clip=5 # gradient clipping



# move model to GPU, if available

if(train_on_gpu):

    net.cuda()



net.train()

# train for some number of epochs

for e in range(epochs):

    # initialize hidden state

    h = net.init_hidden(batch_size)



    # batch loop

    for inputs, labels in train_loader:

        counter += 1



        if(train_on_gpu):

            inputs, labels = inputs.cuda(), labels.cuda()



        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])



        # zero accumulated gradients

        net.zero_grad()



        # get the output from the model

        output, h = net(inputs.long(), h)



        # calculate the loss and perform backprop

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()



        # loss stats

        if counter % print_every == 0:

            # Get validation loss

            val_h = net.init_hidden(batch_size)

            val_losses = []

            net.eval()

            for inputs, labels in valid_loader:



                # Creating new variables for the hidden state, otherwise

                # we'd backprop through the entire training history

                val_h = tuple([each.data for each in val_h])



                if(train_on_gpu):

                    inputs, labels = inputs.cuda(), labels.cuda()



                output, val_h = net(inputs.long(), val_h)

                val_loss = criterion(output.squeeze(), labels.float())



                val_losses.append(val_loss.item())



            net.train()

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))