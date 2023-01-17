import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import os

import plotly.graph_objects as go

import time



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



torch.manual_seed(103)

torch.cuda.manual_seed(103)

np.random.seed(103)



deviceCount = torch.cuda.device_count()

print(deviceCount)



cuda0 = None

if deviceCount > 0:

  print(torch.cuda.get_device_name(0))

  cuda0 = torch.device('cuda:0')
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df.sample(5)
df.info()
sns.countplot(x='target', data=df)

plt.gca().set_ylabel('tweets')
start_time = time.time()



%cd '/kaggle'

!wget -q http://nlp.stanford.edu/data/glove.twitter.27B.zip

!unzip -q glove.twitter.27B.zip



!ls

%cd '/kaggle/working'



print(f'\nDuration: {time.time() - start_time:.0f} seconds')
text_embedding_dimension = 200

key_embedding_dimension = 25





path_to_glove_file = '/kaggle/glove.twitter.27B.{}d.txt'.format(text_embedding_dimension)



embeddings_index_200 = {}

with open(path_to_glove_file) as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, "f", sep=" ")

        embeddings_index_200[word] = coefs



print("Found %s word vectors." % len(embeddings_index_200))







path_to_glove_file = '/kaggle/glove.twitter.27B.{}d.txt'.format(key_embedding_dimension)



embeddings_index_25 = {}

with open(path_to_glove_file) as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, "f", sep=" ")

        embeddings_index_25[word] = coefs



print("Found %s word vectors." % len(embeddings_index_25))
def clean_text(text):

    

    # lower case characters only

    text = text.lower() 

    

    # remove urls

    text = re.sub('http\S+', ' ', text)

    

    # only alphabets, spaces and apostrophes 

    text = re.sub("[^a-z' ]+", ' ', text)

    

    # remove all apostrophes which are not used in word contractions

    text = ' ' + text + ' '

    text = re.sub("[^a-z]'|'[^a-z]", ' ', text)

    

    return text.split()



df['text'] = df['text'].apply(lambda x: clean_text(x))



df.sample(5)
unknown_words = []

total_words = 0



def find_unknown_words(words):

    

    global total_words

    total_words = total_words + len(words)

    

    for word in words:

        if not (word in embeddings_index_200):

            unknown_words.append(word)

    

    return words





df['text'].apply(lambda words: find_unknown_words(words))



print( f'{len(unknown_words)/total_words*100:5.2} % of words are unknown' )
def analyze_unknown_words(unknown_words):

    

    unknown_words = np.array(unknown_words)

    (word, count) = np.unique(unknown_words, return_counts=True)

    

    word_freq = pd.DataFrame({'word': word, 'count': count}).sort_values('count', ascending=False)



    fig = go.Figure(data=[go.Table(

          header=dict(values=list(word_freq.columns),

                    fill_color='paleturquoise',

                    align='left'),

          cells=dict(values=[word_freq['word'], word_freq['count']],

                    fill_color='lavender',

                    align='left'))

          ])

    fig.update_layout(width=300, height=300, margin={'b':0, 'l':0, 'r':0, 't':0, 'pad':0})

    fig.show()

        

analyze_unknown_words(unknown_words)
contractions  = { "i'm" : "i am", "it's" : "it is", "don't" : "do not", "can't" : "cannot", 

                  "you're" : "you are", "that's" : "that is", "we're" : "we are", "i've" : "i have", 

                  "he's" : "he is", "there's" : "there is", "i'll" : "i will", "i'd" : "i would", 

                  "doesn't" : "does not", "what's" : "what is", "didn't" : "did not", 

                  "wasn't" : "was not", "hasn't" : "has not", "they're" : "they are", 

                  "let's" : "let us", "she's" : "she is", "isn't" : "is not", "ain't" : "not", 

                  "aren't" : "are not", "haven't" : "have not", "you'll" : "you will", 

                  "we've" : "we have", "you've" : "you have", "y'all" : "you all", 

                  "weren't" : "were not", "couldn't" : "could not", "would've" : "would have", 

                  "they've" : "they have", "they'll" : "they will", "you'd" : "you would", 

                  "they'd" : "they would", "it'll" : "it will", "where's" : "where is", 

                  "we'll" : "we will", "we'd" : "we would", "he'll" : "he will", 

                  "gov't" : "government", "shouldn't" : "should not", "bioterror" : "biological terror", 

                  "bioterrorism" : "biological terrorism", "wouldn't" : "would not", 

                  "won't" : "will not" }





def expand_contractions(words):

    

    for i in range(len(words)):

        if words[i] in contractions:

            words[i] = contractions[words[i]]

            

    return (' '.join(words)).split()





# precautionary cleaning for any remaing apostrophes

def remove_apostrophes(words):

    words = ' '.join(words)

    words = re.sub("'", '', words)

    return words.split()





df['text'] = df['text'].apply(lambda words: expand_contractions(words))



df['text'] = df['text'].apply(lambda words: remove_apostrophes(words))
unknown_words = []

total_words = 0



df['text'].apply(lambda words: find_unknown_words(words))



print( f'{len(unknown_words)/total_words*100:5.2} % of words are unknown' )
words_freq = {}



def word_frequency(words):

  for word in words:

    if word in words_freq:

      words_freq[word] += 1

    else:

      words_freq[word] = 1



df['text'].apply(lambda words: word_frequency(words))



word = []

count = []

for w in words_freq:

  word.append(w)

  count.append( words_freq[w] )



word = np.array(word)

count = np.array(count)



word_freq = pd.DataFrame({'word': word, 'count': count}).sort_values('count', ascending=False)



fig = go.Figure(data=[go.Table(

      header=dict(values=list(word_freq.columns),

                fill_color='paleturquoise',

                align='left'),

      cells=dict(values=[word_freq['word'], word_freq['count']],

                fill_color='lavender',

                align='left'))

      ])

fig.update_layout(width=300, height=300, margin={'b':0, 'l':0, 'r':0, 't':0, 'pad':0})

fig.show()
stop_words = [ 'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'you', 'for', 'on', 'it', 'my', 'that',

               'with', 'are', 'at', 'by', 'this', 'have', 'from', 'be', 'was', 'do', 'will', 'as', 'up', 

               'me', 'am', 'so', 'we', 'your', 'has', 'when', 'an', 's', 'they', 'about', 'been', 'there',

               'who', 'would', 'into', 'his', 'them', 'did', 'w', 'their', 'm', 'its', 'does', 'where', 'th',

               'b', 'd', 'x', 'p', 'o', 'r', 'c', 'n', 'e', 'g', 'v', 'k', 'l', 'f', 'j', 'z', 'us', 'our',

               'all', 'can', 'may' ] 



def remove_stop_words(words):

  result = []

  for word in words:

    if not (word in stop_words):

      result.append(word)

  return result



df['text'] = df['text'].apply(lambda words: remove_stop_words(words))
df.sample(5)
def text_embed(words):

    

    unknown_indices = []

    mean = np.zeros(text_embedding_dimension)

    

    for i in range(len(words)):

        if words[i] in embeddings_index_200:

            words[i] = embeddings_index_200[ words[i] ]

            mean += words[i]

        else:

            unknown_indices.append(i)

            

    mean /= len(words)-len(unknown_indices)

    

    # unknown words in the text are represented using the mean of the known words

    for i in unknown_indices:

        words[i] = mean

    

    return np.array(words)



df['text'] = df['text'].apply(lambda words: text_embed(words))
def keyword_embed(keyword, text):

    

    if pd.isna(keyword):

        keyword = np.zeros(25)

    else:

        keyword = keyword.lower()

        keyword = re.sub("[^a-z ]+", ' ', keyword)

        keywords = keyword.split()



        if len(keywords) == 0:

            keyword = np.zeros(key_embedding_dimension)

        else:

            keyword = np.zeros(key_embedding_dimension)

            word_count = 0

            for word in keywords:

                if word in embeddings_index_25:

                    keyword += embeddings_index_25[word]

                    word_count += 1



            if word_count > 0:

                keyword = keyword / word_count

 

    return keyword



df['keyword'] = df.apply(lambda x: keyword_embed(x['keyword'], x['text']), axis=1)
df.drop('location', axis=1).sample(5)
# cross_validation_ratio = 0.2

cross_validation_ratio = 0.05



mask = np.random.rand(len(df)) > cross_validation_ratio



train_df = df[mask]



val_df = df[~mask]
x_train_text = train_df['text'].values

x_train_key = train_df['keyword'].values



x_val_text = val_df['text'].values

x_val_key = val_df['keyword'].values



y_train = train_df['target'].values

y_val = val_df['target'].values
x_train_key = np.array( [i for i in x_train_key] ).reshape(-1, key_embedding_dimension)

x_val_key = np.array( [i for i in x_val_key] ).reshape(-1, key_embedding_dimension)
class ANN_Model(nn.Module):

    def __init__(self):

        super().__init__()                          

        self.fc1 = nn.Linear(key_embedding_dimension, 10)

        self.fc2 = nn.Linear(10, 1)

        self.bn1 = nn.BatchNorm1d(10)

        self.dropout1 = nn.Dropout(p=0.1)



    def forward(self, X):

        X = self.fc1(X)

        X = self.bn1(X)

        X = F.relu(X)

        X = self.dropout1(X)

        X = self.fc2(X)

        X = torch.sigmoid(X)

        return X
ann_model = ANN_Model()



if cuda0 != None:

  ann_model.to(cuda0)



criterion_key = nn.BCELoss()

optimizer_key = torch.optim.Adam(ann_model.parameters(), lr=0.01)

# scheduler_key = torch.optim.lr_scheduler.ExponentialLR(optimizer_key, gamma=0.8)
ann_model
start_time = time.time()



train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []



for epoch in range(300):  

    

    ann_model.train()



    tweet = torch.FloatTensor(x_train_key)

    label = torch.FloatTensor(y_train)



    if cuda0 != None:

        tweet = tweet.cuda()

        label = label.cuda()



    pred = ann_model(tweet)

    pred = pred.reshape(-1)



    loss = criterion_key(pred, label)



    optimizer_key.zero_grad()

    loss.backward()

    optimizer_key.step()



    train_losses.append(loss.item())

    train_accuracies.append( ( (pred>0.5) == (label==1) ).sum().item() / len(x_train_key) )





    ann_model.eval()



    with torch.no_grad():



        tweet = torch.FloatTensor(x_val_key)

        label = torch.FloatTensor(y_val)



        if cuda0 != None:

            tweet = tweet.cuda()

            label = label.cuda()



        pred = ann_model(tweet)

        pred = pred.reshape(-1)



        loss = criterion_key(pred, label)



    val_losses.append(loss.item())

    val_accuracies.append( ( (pred>0.5) == (label==1) ).sum().item() / len(x_val_key) )

    

    if (epoch+1)%50 == 0:

        print('Epoch {} Summary:'.format(epoch+1))

        print(f'Train Loss: {train_losses[-1]:7.2f}  Train Accuracy: {train_accuracies[-1]*100:6.3f}%')

        print(f'Validation Loss: {val_losses[-1]:7.2f}  Validation Accuracy: {val_accuracies[-1]*100:6.3f}%')

        print('')



    # scheduler_key.step()



print(f'\nDuration: {time.time() - start_time:.0f} seconds')
x_axis = [i+1 for i in range(len(train_losses))]



plt.plot(x_axis, train_losses, label='training loss')

plt.plot(x_axis, val_losses, label='validation loss')

plt.title('Loss for each epoch')

plt.legend();

plt.show()



plt.plot(x_axis, train_accuracies, label='training accuracy')

plt.plot(x_axis, val_accuracies, label='validation accuracy')

plt.title('Accuracy for each epoch')

plt.legend();

plt.show()
ann_model.eval()



# predictions for the training set

with torch.no_grad():



    tweet = torch.FloatTensor(x_train_key)



    if cuda0 != None:

        tweet = tweet.cuda()



    pred_train_key = ann_model(tweet)

    pred_train_key = pred_train_key.reshape(-1)

    



# predictions for the cross validation set

with torch.no_grad():



    tweet = torch.FloatTensor(x_val_key)



    if cuda0 != None:

        tweet = tweet.cuda()



    pred_val_key = ann_model(tweet)

    pred_val_key = pred_val_key.reshape(-1)
class LSTMnetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_size = 50

        self.input_size = text_embedding_dimension

        self.num_layers = 1

        self.bidirectional = True

        self.num_directions = 1

        self.dropout1 = nn.Dropout(p=0.3)



        if self.bidirectional:

            self.num_directions = 2

 

        self.lstm = nn.LSTM( self.input_size, self.hidden_size, self.num_layers, 

                             bidirectional=self.bidirectional )

        

        self.linear = nn.Linear(self.hidden_size*self.num_directions,1)



    def forward(self, tweet):

        

        lstm_out, _ = self.lstm( tweet.view(len(tweet), 1, -1) )



        x = self.dropout1( lstm_out.view(len(tweet),-1) )

        

        output = self.linear(x)

        

        pred = torch.sigmoid( output[-1] )

        

        return pred
lstm_model = LSTMnetwork()



if cuda0 != None:

  lstm_model.to(cuda0)



criterion_text = nn.BCELoss()

optimizer_text = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

scheduler_text = torch.optim.lr_scheduler.ExponentialLR(optimizer_text, gamma=0.1)
lstm_model
ann_model_weight = 0.3

lstm_model_weight = 1-ann_model_weight
start_time = time.time()



train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []



for epoch in range(4):  



    epoch_start_time = time.time()



    print('Epoch : {}'.format(epoch+1))



    trainLoss = 0

    correct = 0



    lstm_model.train()



    for i in range(len(x_train_text)):



        lstm_model.zero_grad()



        tweet = torch.FloatTensor(x_train_text[i])

        label = torch.FloatTensor( np.array([y_train[i]]) )



        if cuda0 != None:

            tweet = tweet.cuda()

            label = label.cuda()



        pred = lstm_model(tweet)



        loss = criterion_text(pred, label)



        lambdaParam = torch.tensor(0.001)

        l2_reg = torch.tensor(0.)



        if cuda0 != None:

          lambdaParam = lambdaParam.cuda()

          l2_reg = l2_reg.cuda() 



        for param in lstm_model.parameters():

          if cuda0 != None:

            l2_reg += torch.norm(param).cuda()

          else:

            l2_reg += torch.norm(param)



        loss += lambdaParam * l2_reg



        pred = pred.item()*lstm_model_weight + pred_train_key[i].item()*ann_model_weight

        

        if pred > 0.5:

            pred = 1

        else:

            pred = 0



        if pred == int( label.item() ):

            correct += 1



        trainLoss += loss.item()



        optimizer_text.zero_grad()

        loss.backward()

        optimizer_text.step()



        if (i+1)%1000 == 0:

            print('Processed {} tweets out of {}'.format(i+1, len(x_train_text)))



    train_losses.append(trainLoss/len(x_train_text))

    train_accuracies.append( correct/len(x_train_text) )



    valLoss = 0

    correct = 0



    lstm_model.eval()



    with torch.no_grad():



        for i in range(len(x_val_text)):



            tweet = torch.FloatTensor(x_val_text[i])

            label = torch.FloatTensor( np.array([y_val[i]]) )



            if cuda0 != None:

                tweet = tweet.cuda()

                label = label.cuda()



            pred = lstm_model( tweet )



            loss = criterion_text(pred, label)



            valLoss += loss.item()



            pred = pred.item()*lstm_model_weight + pred_val_key[i].item()*ann_model_weight



            if pred > 0.5:

                pred = 1

            else:

                pred = 0



            if pred == int( label.item() ):

                correct += 1



    val_losses.append(valLoss/len(x_val_text))

    val_accuracies.append( correct/len(x_val_text) )



    print('Epoch Summary:')

    print(f'Train Loss: {train_losses[-1]:7.2f}  Train Accuracy: {train_accuracies[-1]*100:6.3f}%')

    print(f'Validation Loss: {val_losses[-1]:7.2f}  Validation Accuracy: {val_accuracies[-1]*100:6.3f}%')

    print(f'Duration: {time.time() - epoch_start_time:.0f} seconds')

    print('')



    scheduler_text.step()



print(f'\nDuration: {time.time() - start_time:.0f} seconds')
x_axis = [i+1 for i in range(len(train_losses))]



plt.plot(x_axis, train_losses, label='training loss')

plt.plot(x_axis, val_losses, label='validation loss')

plt.title('Loss for each epoch')

plt.legend();

plt.show()



plt.plot(x_axis, train_accuracies, label='training accuracy')

plt.plot(x_axis, val_accuracies, label='validation accuracy')

plt.title('Accuracy for each epoch')

plt.legend();

plt.show()
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))



test_df['text'] = test_df['text'].apply(lambda words: expand_contractions(words))



test_df['text'] = test_df['text'].apply(lambda words: remove_apostrophes(words))



test_df['text'] = test_df['text'].apply(lambda words: remove_stop_words(words))



test_df['text'] = test_df['text'].apply(lambda words: text_embed(words))



test_df['keyword'] = test_df.apply(lambda x: keyword_embed(x['keyword'], x['text']), axis=1)



test_df.drop('location', axis=1).sample(5)
x_test_text = test_df['text'].values

x_test_key = test_df['keyword'].values



x_test_key = np.array( [i for i in x_test_key] ).reshape(-1, key_embedding_dimension)
test_predictions = []
ann_model.eval()



with torch.no_grad():



    tweet = torch.FloatTensor(x_test_key)



    if cuda0 != None:

        tweet = tweet.cuda()



    pred_test_key = ann_model(tweet)

    pred_test_key = pred_test_key.reshape(-1)
lstm_model.eval()



with torch.no_grad():



    for i in range(len(x_test_text)):



        tweet = torch.FloatTensor(x_test_text[i])



        if cuda0 != None:

            tweet = tweet.cuda()



        pred = lstm_model( tweet )



        pred = pred.item()*lstm_model_weight + pred_test_key[i].item()*ann_model_weight



        if pred > 0.5:

            pred = 1

        else:

            pred = 0



        test_predictions.append(pred)
test_predictions = np.array(test_predictions)



ids = test_df['id'].values
output = pd.DataFrame({'id': ids, 'target': test_predictions})



output.to_csv('/kaggle/working/my_submission.csv', index=False)