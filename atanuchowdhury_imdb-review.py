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
# Load and visualize the data
import numpy as np
import pandas as pd

data=pd.read_csv('/kaggle/input/imdb-movie-review-dataset/movie_data.csv')
    

import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text + " ".join(emoticons).replace('-', '')
    return text

preprocessor("</a>This :) is :( a test :-)!")
data['review'] = data['review'].apply(preprocessor)


data['review'][3]
# split the words in the review
reviews=[]
for i in range(data['review'].shape[0]):
    words=data['review'][i].split()
    reviews.append(words)
reviews[0]
# Now I want to join all the words in whole review
words=[]
for i in range(data['review'].shape[0]):
    word=[word for word in data['review'][i].split()]
    words+=word
    
    
len(words)
from collections import Counter

# build a dictionary that maps words to integers
counts=Counter(words)
vocab=sorted(counts,key=counts.get,reverse=True)
vocab_to_int={word:ii for ii,word in enumerate(vocab,start=1)}


## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints

reviews_ints=[]

for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review])
# stats about vocabulary
print("Unique words: ",len(vocab_to_int))
print()

# print tokens in first review
print('Tokenized review : \n',reviews_ints[:1])
# getting the labels corresponding to the Reviews 0="Negative" and 1="Positive"
labels=data['sentiment'].to_numpy()
len(labels)
# outlier review stats

review_lens=Counter([len(x) for x in reviews_ints])
print("Zero Length Reviews : {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# there is no empty review
# The review with maximum length is 2498 
def pad_features(reviews_ints,seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features=np.zeros((len(reviews_ints),seq_length),dtype=int)
    
    # For each review I grab that review and
    for i,row in enumerate(reviews_ints):
        features[i,-len(row):]=np.array(row)[:seq_length]
        
    return features
# test The Implementation

seq_length=200

features=pad_features(reviews_ints,seq_length=seq_length)

# test Statements
assert len(features)==len(reviews_ints),"Features should have as many rows as reviews"

assert len(features[0])==seq_length,"Each Features row should contains Sequence length values"

# Print first 10 values of first 30 batches
print(features[:30,:10])
features[0]
split_frac=0.8


### split data into training, validation, and test data (features and labels, x and y)

split_idx=int(len(features)*split_frac)

train_x,remaining_x=features[:split_idx],features[split_idx:]
train_y,remaining_y=labels[:split_idx],labels[split_idx:]

test_idx=int(len(remaining_x)*0.5)
val_x,test_x=remaining_x[:test_idx],remaining_x[test_idx:]
val_y,test_y=remaining_y[:test_idx],remaining_x[test_idx:]

## print out the shapes of your resultant feature data

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
import torch

from torch.utils.data import TensorDataset,DataLoader

# Create Tensor dataset
train_data=TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
val_data=TensorDataset(torch.from_numpy(val_x),torch.from_numpy(val_y))
test_data=TensorDataset(torch.from_numpy(test_x),torch.from_numpy(test_y))

# Dataloader
batch_size=50

# Make sure to shuffle your training data

train_loader=DataLoader(train_data,shuffle=True,batch_size=batch_size,drop_last=True)
valid_loader=DataLoader(val_data,shuffle=True,batch_size=batch_size,drop_last=True)
test_loader=DataLoader(test_data,shuffle=True,batch_size=batch_size,drop_last=True)

# Obtain One batch of Training Data
dataiter=iter(train_loader)
sample_x,sample_y=dataiter.next()

print("Sample Input Size: ",sample_x.size()) # batch_size,seq_length
print("Sample Input :",sample_x)
print()

print("Sample Label size :",sample_y.size()) # Batch_size
print("Sample Label :",sample_y)
# First Checking if GPU is Available

train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print("Training on GPU")

else:
    print("Training on CPU")
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        
# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
# Loss and optimization functions
lr=0.001
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)
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
        output, h = net(inputs, h)

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

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
# Get test data loss and accuracy

test_losses=[]
num_correct=0

# init hidden state

h=net.init_hidden(batch_size)

net.eval()

# Iterate Over test data
for inputs,lebels in test_loader:
    
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    
    h=tuple([each.data for each in h])
    
    if(train_on_gpu):
        inputs,labels=inputs.cuda(),labels.cuda()
        
    # Get predicted outputs
    output,h=net(inputs,h)
    
    # calculate loss
    test_loss=criterion(output.squeeze(),labels.float())
    test_losses.append(test_loss.item())
    
    # Convert output probabilities to predicted class(0 or 1) 
    pred=torch.round(output.squeeze())   # rounds to the nearest integer
    
    # Compare predictions to true label
    
    correct_tensor=pred.eq(labels.float().view_as(pred))
    correct=np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct+=np.sum(correct)
    
    
    
    
#stats

# Average test loss
print("Test Loss: {:.3f}".format(np.mean(test_losses)))

# Accuracy over all test data
test_acc=num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))    
    
    
    
    
# negative test review
test_review_neg='The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
from string import punctuation

def tokenize_review(test_review):
    test_review=test_review.lower() # Lowercase
    # Get Rid of punctuation
    test_text=''.join(c for c in test_review if c not in punctuation)
    
    # Splitting by spaces
    test_words=test_text.split()
    
    # Tokens
    test_ints=[]
    test_ints.append([vocab_to_int[word] for word in test_words])
    
    
    return test_ints

# test code and generate tokenized review

test_ints=tokenize_review(test_review_neg)
print(test_ints)
# Test sequence Padding
seq_length=200
features=pad_features(test_ints,seq_length)
print(features)
# test conversion to tensor and pass into your model
feature_tensor=torch.from_numpy(features)
print(feature_tensor.size())
def predict(net,test_review,sequence_length=200):
    net.eval()
    
    # Tokenize Review
    test_ints=tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features=pad_features(test_ints,seq_length)
    
    # Convert to tensor to pass into your model
    feature_tensor=torch.from_numpy(features)
    batch_size=feature_tensor.size(0)
    
    # Initialize hidden State
    h=net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor=feature_tensor.cuda()
        
    # Get the output from the model
    output,h=net(feature_tensor,h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred=torch.round(output.squeeze())
    #printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    
    # Print custom response
    if(pred.item()==1):
        print("Positive Review Detected !")
        
    else:
        print("Negative Review Detected !")
    
# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

# call function
seq_length=200 # good to use the length that was trained on

predict(net, test_review_pos, seq_length)
