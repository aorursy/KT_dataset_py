# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test_data_sub=pd.read_csv('../input/test.csv')

train_data=pd.read_csv('../input/train.csv')

submission_data=pd.read_csv('../input/sample-submission.csv')
test_data_sub.head()
submission_data.head()
train_data.head()
test_data_sub.head()
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
appos = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not"

}
text_data=train_data['text']
from string import ascii_letters

text=text_data[3]

print(text)

lower_case = text.lower()

words = lower_case.split()

reformed = [appos[word] if word in appos else word for word in words]

reformed_test=list()

for word in reformed:

    if word not in stop_words and '@' not in word and '&' not in word :

        reformed_test.append(word)

reformed = " ".join(reformed_test) 

punct_text= "".join([ch for ch in reformed if ch in ascii_letters or ch==" "])

punct_text
from string import punctuation

def review_formatting(reviews):

    all_reviews=list()

    for text in reviews:

        lower_case = text.lower()

        words = lower_case.split()

        reformed = [appos[word] if word in appos else word for word in words]

        reformed_test=list()

        for word in reformed:

            if word not in stop_words and '@' not in word and '&' not in word:

                reformed_test.append(word)

        reformed = " ".join(reformed_test) 

        punct_text= "".join([ch for ch in reformed if ch in ascii_letters or ch==" "])

        all_reviews.append(punct_text)

    all_text = " ".join(all_reviews)

    all_words = all_text.split()

    return all_reviews, all_words
from collections import Counter 

def get_dict(all_reviews, all_words):

# Count all the words using Counter Method

    count_words = Counter(all_words)

    total_words=len(all_words)

    sorted_words=count_words.most_common(total_words)

    vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}

    return vocab_to_int
def encode_reviews(reviews, vocab_to_int):

    """

    encode_reviews function will encodes review in to array of numbers

    """   

    encoded_reviews=list()

    for review in reviews:

        review = review.lower()

        encoded_review=list()

        for word in review.split():

            if word not in vocab_to_int.keys():

                encoded_review.append(0)

            else:

                encoded_review.append(vocab_to_int[word])

        encoded_reviews.append(encoded_review)

    return encoded_reviews
all_reviews, all_words=review_formatting(text_data)

vocab_to_int=get_dict(all_reviews, all_words)

encoded_reviews=encode_reviews(all_reviews, vocab_to_int)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

review_len=[len(encoded_review) for encoded_review in encoded_reviews]

pd.Series(review_len).hist()

plt.show()

pd.Series(review_len).describe()
#we need to remove tweet with zeros length

updated_reviews=list()

updated_labels=list()

null_index_list=list()

for i, encoded_review in enumerate(encoded_reviews):

    if (len(encoded_review)==0):

        null_index_list.append(i)

    else:

        updated_reviews.append(encoded_reviews[i])

        updated_labels.append(train_data['target'].values[i])

print(len(encoded_reviews),len(null_index_list)+len(updated_reviews),len(updated_labels))

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

review_len=[len(encoded_review) for encoded_review in updated_reviews]

pd.Series(review_len).hist()

plt.show()

pd.Series(review_len).describe()
def pad_sequences(encoded_reviews, sequence_length=10):

    ''' 

    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.

    '''

    features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)

    

    for i, review in enumerate(encoded_reviews):

        review_len=len(review)

        if (review_len<=sequence_length):

            zeros=list(np.zeros(sequence_length-review_len))

            new=zeros+review

        else:

            new=review[:sequence_length]

        features[i,:]=np.array(new)

    return features
features=pad_sequences(updated_reviews, sequence_length=10)
true_labels=[1 if label==4 else 0 for label in updated_labels]
#split_dataset into 80% training , 10% test and 10% Validation Dataset

train_x=np.array(features[:int(0.90*len(features))])

train_y=np.array(true_labels[:int(0.90*len(features))])

valid_x=np.array(features[int(0.90*len(features)):])

valid_y=np.array(true_labels[int(0.90*len(features)):])

print(len(train_y), len(train_x), len(valid_y), len(valid_x))
import torch

from torch.utils.data import DataLoader, TensorDataset



#create Tensor Dataset

train_data=TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

valid_data=TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))



#dataloader

batch_size=500

train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)

valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
# obtain one batch of training data

dataiter = iter(valid_loader)

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

        

        # embedding and LSTM layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 

                            dropout=drop_prob, batch_first=True)

        

        # dropout layer

        self.dropout = nn.Dropout(0.3)

        

        # linear and sigmoid layers

        self.fc1 = nn.Linear(hidden_dim, 256)

        self.fc2 = nn.Linear(256, 32)

        self.fc3 = nn.Linear(32, output_size)

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

        out = self.fc1(out)

        out = self.fc2(out)

        out = self.fc3(out)

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

hidden_dim = 1000

n_layers = 2



net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)



print(net)
# loss and optimization functions

lr=0.001



criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# training params



epochs = 2 # 3-4 is approx where I noticed the validation loss stop decreasing



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

        if(inputs.shape[0] != batch_size):

                    continue

                

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

                if(inputs.shape[0] != batch_size):

                    continue

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
def preprocess(input_test):

    all_reviews, all_words=review_formatting(input_test)

    encoded_reviews=encode_reviews(all_reviews, vocab_to_int)

    final_text=pad_sequences(encoded_reviews, sequence_length=10)

    return final_text
def test_model(input_test):

    output_list=list()

    batch_size=50   

    net.eval()

    with torch.no_grad():

        test_review=preprocess(input_test)

        for review in test_review:

            # convert to tensor to pass into your model

            feature_tensor = torch.from_numpy(review).view(1,-1)

            if(train_on_gpu):

                feature_tensor= feature_tensor.cuda()

            batch_size = feature_tensor.size(0)

            # initialize hidden state

            h = net.init_hidden(batch_size)

            # get the output from the model

            output, h = net(feature_tensor, h)

            pred = torch.round(output.squeeze()) 

            output_list.append(pred)

        labels=[int(i.data.cpu().numpy()) for i in output_list]

        return labels

labels=test_model(test_data_sub['text'])
test_data_sub.head()
out_labels=[4*i for i in labels]

len(out_labels)
len(test_data_sub)
output = pd.DataFrame()

output['Id'] = test_data_sub['Id']

output['target'] = out_labels

output.to_csv('submission.csv', index=False)
output[:10]