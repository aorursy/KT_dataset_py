!ls ../input
import numpy as np



# read data from text files

with open('../input/reviews.txt', 'r') as f:

    reviews = f.read()

with open('../input/labels.txt', 'r') as f:

    labels = f.read()
print(reviews[:2000])

print()

print(labels[:20])
from string import punctuation



print(punctuation)



# get rid of punctuation

reviews = reviews.lower() # lowercase, standardize

all_text = ''.join([c for c in reviews if c not in punctuation])
# split by new lines and spaces

reviews_split = all_text.split('\n')

all_text = ' '.join(reviews_split)



# create a list of words

words = all_text.split()
words[:30]
len(words)
# feel free to use this import 

from collections import Counter



temp = Counter(words)

temp = temp.most_common()



## Build a dictionary that maps words to integers

vocab_to_int = {}

i = 1

for pair in temp:

    vocab_to_int.update({pair[0]:i})

    i+=1





## use the dict to tokenize each review in reviews_split

## store the tokenized reviews in reviews_ints

reviews_ints = []

for review in reviews_split:

    word_list = review.split()

    num_list = []

    for word in word_list:

        num_list.append(vocab_to_int[word])

    reviews_ints.append(num_list)



# stats about vocabulary

print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+

print()



# print tokens in first review

print('Tokenized review: \n', reviews_ints[:1])
# 1=positive, 0=negative label conversion

labels = labels.split("\n")

encoded_labels = [1 if i == "positive" else 0 for i in labels]

# outlier review stats

review_lens = Counter([len(x) for x in reviews_ints])

print("Zero-length reviews: {}".format(review_lens[0]))

print("Maximum review length: {}".format(max(review_lens)))
print('Number of reviews before removing outliers: ', len(reviews_ints))



idx = [i for i,review in enumerate(reviews_ints) if len(reviews_ints[i])!=0]

## remove any reviews/labels with zero length from the reviews_ints list.



reviews_ints = [reviews_ints[i] for i in idx] 

encoded_labels = [encoded_labels[i] for i in idx]



print('Number of reviews after removing outliers: ', len(reviews_ints))
def pad_features(reviews_ints, seq_length):

    ''' Return features of review_ints, where each review is padded with 0's 

        or truncated to the input seq_length.

    '''

    features = []

    

    ## implement function

    for review in reviews_ints:

        if len(review)<seq_length:

            features.append(list(np.zeros(seq_length-len(review)))+review)

        elif len(review)>seq_length:

            features.append(review[:seq_length])

        else:

            features.append(review)

    

    features = np.asarray(features, dtype=int)

    return features
# Test your implementation!



seq_length = 200



features = pad_features(reviews_ints, seq_length=seq_length)



## test statements - do not change - ##

assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."

assert len(features[0])==seq_length, "Each feature row should contain seq_length values."



# print first 10 values of the first 30 batches 

print(features[:30,:10])
from sklearn.model_selection import train_test_split



split_frac = 0.8



## split data into training, validation, and test data (features and labels, x and y)

train_x,remaining_x,train_y,remaining_y = train_test_split(features, encoded_labels, test_size = 0.2)

test_x,valid_x,test_y,valid_y = train_test_split(remaining_x,remaining_y, test_size = 0.5)



#as the labels returned by train_test_split is list, we will convert them to ndarray

train_y = np.asarray(train_y)

test_y = np.asarray(test_y)

valid_y = np.asarray(valid_y)

## print out the shapes of your resultant feature data

print((train_x.shape), (test_x.shape), (valid_x.shape))

type(train_y)
import torch

from torch.utils.data import TensorDataset, DataLoader



# create Tensor datasets

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))



# dataloaders

batch_size = 50



# make sure to SHUFFLE your data

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
# obtain one batch of training data

dataiter = iter(train_loader)

sample_x, sample_y = dataiter.next()



print('Sample input size: ', sample_x.size()) # batch_size, seq_length

print('Sample input: \n', sample_x)

print()

print('Sample label size: ', sample_y.size()) # batch_size

print('Sample label: \n', sample_y)
# First checking if GPU is available

train_on_gpu=torch.cuda.is_available()



if(train_on_gpu):

    print('Training on GPU.')

else:

    print('No GPU available, training on CPU.')
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

        

        # define all layers

        #embedding

        #LSTM

        #fully_connected

        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,

                            dropout=drop_prob, batch_first = True)

        self.FC = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

        

        

        



    def forward(self, x, hidden):

        """

        Perform a forward pass of our model on some input and hidden state.

        """

        batch_size = x.size(0)

        

        x = x.long()

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        

        #stack_up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        

        out = self.FC(lstm_out)

        

        sig_out = self.sig(out)

        

        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]

        # return last sigmoid output and hidden state

        return sig_out, hidden

    

    

    def init_hidden(self, batch_size):

        ''' Initializes hidden state '''

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,

        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda(),

                      weight.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers,batch_size,self.hidden_dim).zero_(),

                      weight.new(self.n_layers,batch_size,self.hidden_dim).zero_())



        return hidden

        
# Instantiate the model w/ hyperparams

vocab_size = len(vocab_to_int) + 1

output_size = 1

embedding_dim = 200 

hidden_dim = 256

n_layers = 2



net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)



print(net)
# loss and optimization functions

lr=0.001



criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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

        #print(counter)



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



test_losses = [] # track loss

num_correct = 0



# init hidden state

h = net.init_hidden(batch_size)



net.eval()

# iterate over test data

for inputs, labels in test_loader:



    # Creating new variables for the hidden state, otherwise

    # we'd backprop through the entire training history

    h = tuple([each.data for each in h])



    if(train_on_gpu):

        inputs, labels = inputs.cuda(), labels.cuda()

    

    # get predicted outputs

    output, h = net(inputs, h)

    

    # calculate loss

    test_loss = criterion(output.squeeze(), labels.float())

    test_losses.append(test_loss.item())

    

    # convert output probabilities to predicted class (0 or 1)

    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    

    # compare predictions to true label

    correct_tensor = pred.eq(labels.float().view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    num_correct += np.sum(correct)





# -- stats! -- ##

# avg test loss

print("Test loss: {:.3f}".format(np.mean(test_losses)))



# accuracy over all test data

test_acc = num_correct/len(test_loader.dataset)

print("Test accuracy: {:.3f}".format(test_acc))
# negative test review

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

def preprocess(review, vocab_to_int):

    review = review.lower()

    word_list = review.split()

    num_list = []

    #list of reviews

    #though it contains only one review as of now

    reviews_int = []

    for word in word_list:

        if word in vocab_to_int.keys():

            num_list.append(vocab_to_int[word])

    reviews_int.append(num_list)

    return reviews_int

    
def predict(net, test_review, sequence_length=200):

    ''' Prints out whether a give review is predicted to be 

        positive or negative in sentiment, using a trained model.

        

        params:

        net - A trained net 

        test_review - a review made of normal text and punctuation

        sequence_length - the padded length of a review

        '''

    #change the reviews to sequence of integers

    int_rev = preprocess(test_review, vocab_to_int)

    #pad the reviews as per the sequence length of the feature

    features = pad_features(int_rev, seq_length=seq_length)

    

    #changing the features to PyTorch tensor

    features = torch.from_numpy(features)

    

    #pass the features to the model to get prediction

    net.eval()

    val_h = net.init_hidden(1)

    val_h = tuple([each.data for each in val_h])



    if(train_on_gpu):

        features = features.cuda()



    output, val_h = net(features, val_h)

    

    #rounding the output to nearest 0 or 1

    pred = torch.round(output)

    

    #mapping the numeric values to postive or negative

    output = ["Positive" if pred.item() == 1 else "Negative"]

    

    # print custom response based on whether test_review is pos/neg

    print(output)

        
# positive test review

test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

# call function

# try negative and positive reviews!

seq_length=200

predict(net, test_review_pos, seq_length)