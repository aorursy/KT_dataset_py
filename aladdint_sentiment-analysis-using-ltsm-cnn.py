# check the remianing of memory
!free -h
!nvidia-smi
# Load libraries
!pip install -U tensorflow==1.15.2
import tensorflow
print(tensorflow.__version__) # make sure the version of tensorflow
import numpy as np # for scientific computing
import pandas as pd # for data analysis
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
import missingno as msno # for missing data visualization
import collections
import nltk
import codecs
import string
import re
from tqdm import tqdm
from collections import defaultdict
from collections import Counter 
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dropout
plt.style.use('ggplot')
np.random.seed(42) # set the random seeds
# Load data
train = pd.read_csv('../input/cs98x-twitter-sentiment/training.csv', low_memory = False)
test = pd.read_csv('../input/cs98x-twitter-sentiment/test.csv', low_memory = False)
train.head()
# Show the test dataset
test.head()
# Show the information of dataset
print('Information of the train dataset')
print(train.info())
print('-'*70)
print('Information of the test dataset')
# Show the information of test dataset
print(test.info())
# if using not whole training and test data
# train = train.iloc[:1000, :]
# test = test.iloc[:1000, :]

# Keep Id of test dataset for submission
ID = test['id']
# Let's searching the missing values in the training
msno.matrix(df=train, figsize=(13,10), color=(0,0,0))
# Visualizing target
plt.figure(figsize=(9,6))
sns.countplot(x='target', data=train)
# Create the function to change the structure of date columns
def change_dates(df):  
  date = df['date']
  date_splitted = date.str.split()
  weeks_list, month_list, day_list, time_list, PDT_list, year_list = [], [], [], [], [], []
  for dates in date_splitted:
    weeks_list.append(dates[0])
    month_list.append(dates[1])
    day_list.append(dates[2])
    time_list.append(dates[3])
    PDT_list.append(dates[4])
    year_list.append(dates[5])

  df['week'] = weeks_list
  df['month'] = month_list
  df['day'] = day_list
  df['time'] = time_list
  df['PDT'] = PDT_list
  df['year'] = year_list
  df = df.drop('date', axis=1) # Remove the date column
  return df

train = change_dates(train) # Apply the function we created above
train.head()
# Visualizing the number of tweets posted in the week
plt.figure(figsize=(9,6))
plt.title('Number of tweets per weeks')
sns.countplot(x='week', data=train)
# Visualizing the positive and negative tweets per weeks
plt.figure(figsize=(9,6))
plt.title('Number of positive and negative tweets per weeks')
sns.countplot(x='week', data=train, hue='target' )
# Visualizing the number of tweets posted on month
plt.figure(figsize=(9,6))
plt.title('Number of tweets per month')
sns.countplot(x='month', data=train)
# Visualizing target
plt.figure(figsize=(9,6))
plt.title('Number of positive and negative tweets per month')
sns.countplot(x='month', data=train, hue='target')
# Let's plot the number of letters in the positive and negative tweets
sns.set()
fig = plt.figure(figsize = (15,5))
negative_len=train[train['target']==0]['text'].str.len() # extract the number of letters in the negative tweets
ax1 = fig.add_subplot(1,2,1)
ax1.hist(negative_len, alpha=0.6, bins=20, color='r',label = 'Negative tweets')
ax1.legend()
positive_len=train[train['target']==4]['text'].str.len() # extract the number of letters in the positive tweets
ax2 = fig.add_subplot(1,2,2)
ax2.hist(positive_len, alpha=0.6, bins=20,color='b',label = 'Positive tweets')
ax2.legend()

ax1.set_xlabel('number of letters')
ax1.set_ylabel('number of tweets')
ax2.set_xlabel('number of letters')
ax2.set_ylabel('number of tweets')

fig.suptitle('Number of letters in a tweet')
fig.tight_layout()
# Let's plot the number of words in the positive and negative tweets
sns.set()
fig = plt.figure(figsize = (15,5))
negative_len=train[train['target']==0]['text'].str.split().apply(len) # extract the number of words in the negative tweets
ax1 = fig.add_subplot(1,2,1)
ax1.hist(negative_len, alpha=0.6, bins=20, color='r',label = 'Negative tweets')
ax1.legend()
positive_len=train[train['target']==4]['text'].str.split().apply(len) # extract the number of words in the positive tweets
ax2 = fig.add_subplot(1,2,2)
ax2.hist(positive_len, alpha=0.6, bins=20,color='b',label = 'Positive tweets')
ax2.legend()

ax1.set_xlabel('number of words')
ax1.set_ylabel('number of tweets')
ax2.set_xlabel('number of words')
ax2.set_ylabel('number of tweets')

fig.suptitle('Number of words in a tweet')
fig.tight_layout()
# Create the function to make the corpus
def create_corpus(target):
    corpus=[]
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

corpus_positive = create_corpus(4) # corpus for the positive tweets
corpus_negative = create_corpus(0) # corpus for the negative tweets

print('Length of positive corpus is', len(corpus_positive))
print('Length of negative corpus is', len(corpus_negative))
# Let's create the function to extract the most common 30 words from the corpus 
def common_word(corpus):
    for idx, low in enumerate(corpus):
        corpus[idx] = low.lower()
    counter = Counter(corpus)
    stop_word_dict = {}
    for i in range(1,31):
        for stop in counter.most_common(i):
            stop_key = str(stop[0])
            stop_value = stop[1]
            stop_word_dict[stop_key] = stop_value
    return stop_word_dict

common_positive = common_word(corpus_positive)
common_positive_sorted = sorted(common_positive.items(), key=lambda x:x[1], reverse=True)

# Let's show the words and how many did they appear in the positive tweets
word_count_list, common_word_list = [], []
for word_count in common_positive_sorted:
  common_word_list.append(word_count[0])
  word_count_list.append(word_count[1])

plt.figure(figsize=(15,8))
plt.title('Top 30 words appeared the most in the positive tweets')
plt.xlabel('top 30 words')
plt.ylabel('count')
plt.bar(common_word_list, word_count_list, alpha=0.6, color='r')
# Let's show the words and how many did they appear in the negative tweets
common_negative = common_word(corpus_negative)
common_negative_sorted = sorted(common_negative.items(), key=lambda x:x[1], reverse=True)
word_count_list, common_word_list = [], []
for word_count in common_negative_sorted:
  common_word_list.append(word_count[0])
  word_count_list.append(word_count[1])

plt.figure(figsize=(15,8))
plt.title('Top 30 words appeared the most in the negative tweets')
plt.xlabel('top 30 words')
plt.ylabel('count')
plt.bar(common_word_list, word_count_list, alpha=0.6, color='b')
# Let's create the function to show how many times each punctuation appeared in the corpus 
def punctuation_show(corpus, color):
    plt.figure(figsize=(10,5))
    plt.title('Number of punctuations appeared in the tweets')
    plt.xlabel('punctuations')
    plt.ylabel('count')
    punc_dict = defaultdict(int)
    punc = string.punctuation
    for i in corpus:
        if i in punc:
            punc_dict[i] += 1
    punc_key = punc_dict.keys()
    punc_value = punc_dict.values()
    plt.bar(punc_key, punc_value, alpha=0.6, color=color)

punctuation_show(corpus_positive, 'r') # This shows the result of positive tweets
# This shows the result of negative tweets
punctuation_show(corpus_negative, 'b')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Remove URL
url = re.compile(r'https?://\S+|www\.\S+')
train['text'] = train['text'].apply((lambda x: url.sub(r'',x)))
test['text'] = test['text'].apply((lambda x: url.sub(r'',x)))
# Remove HTML
html = re.compile(r'<.*?>')
train['text'] = train['text'].apply((lambda x: html.sub(r'',x)))
test['text'] = test['text'].apply((lambda x: html.sub(r'',x)))
# Remove the words which contain the number
train['text'] = train['text'].apply((lambda x: re.sub('\w*\d\w*', '', x)))
test['text'] = test['text'].apply((lambda x: re.sub('\w*\d\w*', '', x)))

# Remove stop words
# train['text'] = train['text'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
# test['text'] = test['text'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])

# Make the tweet lower letters
train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())
# Remove punctuation 
table = str.maketrans('','',string.punctuation)
train['text'] = train['text'].apply((lambda x: x.translate(table)))
test['text'] = test['text'].apply((lambda x: x.translate(table)))


# Remove the tweet which doesn' have any word after preprocessing in the train data.
train = train[train['text'] != '']
# Let's count the max length of tweet in both train and test data for turning tweets into seaquences
max_len = 0
for i in train['text']:
  split_i = i.split()
  if len(split_i) > max_len:
    max_len = len(split_i)

for j in test['text']:
  split_j = j.split()
  if len(split_j) > max_len:
    max_len = len(split_j)
    
print('Max length of tweets :', max_len)
# Convert the tweets into the sequences in train and test data
max_fatures = 300000 # the number of words to be used for the input of embedding layer
tokenizer = Tokenizer(num_words=max_fatures, split=' ') #Create the instance of Tokenizer
tokenizer.fit_on_texts(train['text'].values)
train_converted = tokenizer.texts_to_sequences(train['text'].values)
test = tokenizer.texts_to_sequences(test['text'].values)
train_converted = pad_sequences(train_converted, maxlen=max_len) # Turning the vectors of train data into sequences 
test = pad_sequences(test, maxlen=max_len) # Turning the vectors of test data into sequences 
target_converted = pd.get_dummies(train['target']).values # One-hot expression
# Make sure that the shape of train and test data are same
print('The shape of train data :', train_converted.shape)
print('The shape of test data :', test.shape)
print('The shape of target of the training :', target_converted.shape)
# Make sure that the shape of train and test data are same
X_train, X_test, Y_train, Y_test = train_test_split(train_converted, target_converted, test_size = 0.1, random_state = 42)

# Use half of the test data for validation during training
validation_size = 50000
# validation_size = 500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

print('The shape of train data :', X_train.shape)
print('The shape of labels of train data :', Y_train.shape)
print('The shape of test data :', X_test.shape)
print('The shape of test label data :', Y_test.shape)
# Parameters
embed_dim = 1024 # The size of the vector space where words will be embedded
lstm_out = 196 # The output size of lstm layer
batch_size = 1024
EPOCHS = 2 

# Create the LSTM model
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = train_converted.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['acc'])
print(model.summary()) # Show the summary of the model

history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=batch_size, 
                    validation_data=(X_validate, Y_validate), verbose = 2)

# Plot the result of trained model 
train_acc = history.history['acc']
test_acc = history.history['val_acc']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train accuracy')
plt.plot(x, test_acc, label = 'test accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend() 
# Parameters
NUM_FILTERS = 256 # Number of filters to convolute
NUM_WORDS = 4 # Number of the words to be convoluted
embed_dim = 1024 # The size of the vector space where words will be embedded
batch_size = 1024
EPOCHS = 2

# Create the CNN model
model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length = train_converted.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["acc"])
print(model.summary()) # Show the summary of the model

history = model.fit(X_train, Y_train, batch_size=batch_size,
                    epochs=EPOCHS, validation_data=(X_validate, Y_validate))

# Plot the result of trained model
train_acc = history.history['acc']
test_acc = history.history['val_acc']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train accuracy')
plt.plot(x, test_acc, label = 'test accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend() 
# Let's compute the loss and accuracy of the trained model
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("The loss of this model: %.2f" % (score))
print("The accuracy of this model: %.2f" % (acc))
# Let's predict whether the tweets are positive or negative using real test data set 
predictions = model.predict(test)
# Let's show the first 5 predictions as an samples
print('Prediction samples', predictions[:5])
# Show the shape of prediction, and the number of rows should be same to number of test data
print('The shape of predictions:', predictions.shape)

# Let's round the prediction and turn them into [0, 1] or [1, 0]
prediction_binary = np.round(predictions)
print('Prediction binary expression samples', prediction_binary[:5])

# Let's turn the prediction from [0, 1] and [1, 0] into 0(negative) and 4(positive)
prediction_final = []
for each_pediction in prediction_binary:
  if each_pediction[0] == 1:
    prediction_final.append(0)
  else:
    prediction_final.append(4)




submission = pd.DataFrame({
    "Id": ID,
    "target": prediction_final
})

# Convert dataframe into csv file
submission.to_csv('twitter_sentiment_analysis22_cnn.csv', index=False)