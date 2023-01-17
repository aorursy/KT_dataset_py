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
train = pd.read_csv("/kaggle/input/twitter-sentiment-analysis/train.csv")
test = pd.read_csv("/kaggle/input/twitter-sentiment-analysis/test.csv")
train.head()
tweets = train['tweet']
labels = train['label']
import string
import nltk
from string import punctuation
from nltk.corpus import stopwords
appos = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "didn't" : "did not", "doesn't" : "does not",
"don't" : "do not", "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not", "he'd" : "he would",
"he'll" : "he will", "he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am",
"isn't" : "is not","it's" : "it is", "it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not",
"mustn't" : "must not", "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is",
"shouldn't" : "should not", "that's" : "that is", "there's" : "there is", "they'd" : "they would", "they'll" : "they will",
"they're" : "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not",
"we've" : "we have", "what'll" : "what will","what're" : "what are","what's" : "what is", "what've" : "what have",
"where's" : "where is", "who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is",
"who've" : "who have", "won't" : "will not", "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will",
"you're" : "you are", "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not"}
def preprocess(text):
    all_tweets = list()
    for txt in text:
        lower_case = txt.lower()
        words = lower_case.split()
        formatted = [appos[word] if word in appos else word for word in words]
        formatted_test = list()
        for word in formatted:
            if word not in stopwords.words("english"):
                formatted_test.append(word)
        formatted = " ".join(formatted_test)
        punct_text = "".join([ch for ch in formatted if ch not in punctuation])
        all_tweets.append(punct_text)
    for i in range(len(all_tweets)):
        if all_tweets[i].startswith("user"):
            all_tweets[i] = all_tweets[i].replace("user", '')
    all_text = " ".join(all_tweets)
    all_words = all_text.split()
    
    return all_tweets, all_words
all_tweets, all_words = preprocess(tweets)
all_tweets[3]
import re

for i in range(len(all_tweets)):
    all_tweets[i] = re.sub('[^a-zA-Z0-9]', ' ', all_tweets[i])

all_words = []
for sentence in all_tweets:
    for word in sentence.split():
        all_words.append(word)
from collections import Counter

word_counts = Counter(all_words)
word_list = sorted(word_counts, reverse = True)
word2int = {word : i+1 for (i, word) in enumerate(word_list)}
int2word = {i : word for word, i in word2int.items()}

encoded_tweets = [[word2int[word] for word in tweet.split()] for tweet in all_tweets]
encoded_labels = np.array([label for idx, label in enumerate(labels) if len(encoded_tweets[idx]) > 0])
encoded_tweets = [tweet for tweet in encoded_tweets if len(tweet) > 0]
def pad_tweet(encoded_tweets, tweet_length):
    Tweets = []
    
    for tweet in encoded_tweets:
        if len(tweet) >= tweet_length:
            Tweets.append(tweet[:tweet_length])
        else:
            Tweets.append([0] * (tweet_length - len(tweet)) + tweet)
    return np.array(Tweets)
padded_reviews = pad_tweet(encoded_tweets, 15)
import torch
import torch.nn as nn
train_ratio = 0.9
valid_ratio = 0.1
total = padded_reviews.shape[0]
train_cutoff = int(total * train_ratio)

x_train, y_train = padded_reviews[:train_cutoff], encoded_labels[:train_cutoff]
x_valid, y_valid = padded_reviews[train_cutoff:], encoded_labels[train_cutoff:]


from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))

batch_size = 32

train_loader = DataLoader(train_data, batch_size, shuffle = True, drop_last = True)
valid_loader = DataLoader(valid_data, batch_size, shuffle = True, drop_last = True)
class sentimentLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers, drop_prob = 0.3):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first = True, dropout = drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, input_words, h):
        #Input dimension = batch_size x tweet_length
        batch_size = input_words.shape[0]
        embedd = self.embed(input_words) #dimension = batch_size x tweet_length x embedding_dim
        
        lstm_out, h = self.lstm(embedd, h) #dimension = batch_size x tweet_length x hidden_size
        lstm_out = self.dropout(lstm_out)
        #stacking up the lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size) #dimension = (batch_size * tweet_length) x hidden_size
        
        fc_out = self.fc(lstm_out) #dimension = (batch_size * tweet_length) x output_size
        
        sig_out = self.sig(fc_out) #dimension = (batch_size * tweet_length) x output_size
        sig_out = sig_out.view(batch_size, -1) #dimension = batch_size x (tweet_length * output_size)
        sig_out = sig_out[:, -1] #Extract only the last output of the element of each example in the batch
        
        return sig_out, h
    
    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        
        h = (weights.new(self.n_layers, batch_size, self.hidden_size).zero_(),
             weights.new(self.n_layers, batch_size, self.hidden_size).zero_())
        
        return h
input_size = len(word2int)+1
embedding_dim = 400
hidden_size = 256
output_size = 1
n_layers = 2
model = sentimentLSTM(input_size, embedding_dim, hidden_size, output_size, n_layers)
model
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.BCELoss()
print_every = 100
step = 0
n_epochs = 5
clip = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_losses = []

if torch.cuda.is_available():
    model.cuda()

model.train()
for epoch in range(1, n_epochs+1):
    h = model.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        h = tuple([each.data for each in h]) 
        
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if step % print_every == 0:
            #Validation
            valid_losses = []
            valid_h = model.init_hidden(batch_size)
            model.eval()
            
            for valid_inputs, valid_labels in valid_loader:
                valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                
                valid_h = tuple([each.data for each in valid_h])
                
                valid_output, valid_h = model(valid_inputs, valid_h)
                valid_loss = criterion(valid_output.squeeze(), valid_labels.float())
                valid_losses.append(valid_loss.item())
                train_losses.append(loss.item())
                
            print("Epoch: {}/{}".format((epoch), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            model.train()
test_tweets = test['tweet']
test_tweets, test_words = preprocess(test_tweets)
for i in range(len(test_tweets)):
    test_tweets[i] = re.sub('[^a-zA-Z0-9]', ' ', test_tweets[i])

test_words = []
for sentence in test_tweets:
    for word in sentence.split():
        test_words.append(word)
encoded_test_tweets = []
for tweet in test_tweets:
    encoded_tweet = []
    for word in tweet.split():
        if word not in word2int.keys():
            encoded_tweet.append(0)
        else:
            encoded_tweet.append(word2int[word])
    encoded_test_tweets.append(encoded_tweet)
padded_test_tweets = pad_tweet(encoded_test_tweets, 15)
def test_model(test_input):
    output_list = list()
    model.eval()
    with torch.no_grad():
        for tweet in test_input:
            feature_tensor = torch.from_numpy(tweet).view(1, -1)
            if(torch.cuda.is_available()):
                feature_tensor = feature_tensor.cuda()
            batch_size = feature_tensor.size(0)
            #initialize hidden state
            h = model.init_hidden(batch_size)
            #get the output from the model
            output, h = model(feature_tensor, h)
            pred = torch.round(output.squeeze())
            output_list.append(pred)
        test_labels = [int(i.data.cpu().numpy()) for i in output_list]
        return test_labels
test_labels = test_model(padded_test_tweets)
output = pd.DataFrame()
output['id'] = test['id']
output['label'] = test_labels
output
output.to_csv("subm.csv", index = False)
