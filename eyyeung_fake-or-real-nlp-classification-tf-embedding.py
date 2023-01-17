import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# inputting the files

test_path = '../input/nlp-getting-started/test.csv'

train_path = '../input/nlp-getting-started/train.csv'



test = pd.read_csv(test_path, index_col='id')

train = pd.read_csv(train_path, index_col='id')
# test shape - 3263 x 3

# train shape - 7613 x 4

print(" test shape: " ,test.shape)

print(" train shape: " ,train.shape)
train.head()
print("---% missing in train---")

print(train.isnull().sum()/7613*100)



print("---% missing in test---")

print(test.isnull().sum()/3263*100)
# this would set the target_mean as the mean of the target belonging to that keyword

train['target_mean'] = train.groupby('keyword')['target'].transform('mean')

# check if it is set properly

train[train['keyword']=='ablaze']
## The code of this graph is from the notebook "NLP with Disaster Tweets - EDA, Cleaning and BERT"



fig1 = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword'], hue=train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc='upper right')

plt.title('Target Distribution in Keywords')



plt.show()



## space is shown as %20 - what to do with multiword keywords?
train.keyword.unique()
print("# of unique values in train - # of unique values in test:" , train.keyword.nunique() - test.keyword.nunique())

print(set(train['keyword'].unique()) == set(test['keyword'].unique())) # so there are no new keyworkds in test that wasn't in train

print("# of unique locations in train - # of unique locations in test:" , train.location.nunique() - test.location.nunique())
# how many word in the tweets

train['word_counts'] = train['text'].apply(lambda x: len(str(x).split()))

test['word_counts'] = test['text'].apply(lambda x: len(str(x).split()))

# average length of chars in the word

train['avg_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test['avg_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# how many characters in the tweets

train['char_counts'] = train['text'].apply(lambda x: len(str(x)))

test['char_counts'] = test['text'].apply(lambda x: len(str(x)))
train.head()
fig2, axes = plt.subplots(ncols=3, nrows=1, figsize=(20, 10), dpi=100)



bins = 100

plt.subplot(1, 3, 1)

plt.hist(train[train.target == 0]['word_counts'], alpha = 0.6, bins=bins, label='Fake', color='green')

plt.hist(train[train.target == 1]['word_counts'], alpha = 0.6, bins=bins, label='Real', color='red')

plt.xlabel('word counts')

plt.ylabel('count')

plt.legend(loc='upper right')

plt.subplot(1, 3, 2)

plt.hist(train[train.target == 0]['char_counts'], alpha = 0.6, bins=bins, label='Fake', color='green')

plt.hist(train[train.target == 1]['char_counts'], alpha = 0.6, bins=bins, label='Real', color='red')

plt.xlabel('characters counts')

plt.ylabel('count')

plt.legend(loc='upper right')

plt.subplot(1, 3, 3)

plt.hist(train[train.target == 0]['avg_word_length'], alpha = 0.6, bins=bins, label='Fake', color='green')

plt.hist(train[train.target == 1]['avg_word_length'], alpha = 0.6, bins=bins, label='Real', color='red')

plt.xlabel('average word length')

plt.ylabel('count')

plt.legend(loc='upper right')

plt.show()
## definte the stopwords

STOPWORDS = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
from collections import defaultdict

import string

def make_pun_dict(val):

    pun_dict = defaultdict(int)

    for tweet in train[train.target == int(val)]['text']:

        pun = [pun for pun in str(tweet).lower().split() if pun in string.punctuation]

        for sym in pun:

            pun_dict[sym] +=1

    return pun_dict
real_pun_dict = make_pun_dict(1)

fake_pun_dict = make_pun_dict(0)
fig3 = plt.figure(figsize=(10, 5), dpi=100)

plt.bar(real_pun_dict.keys(), real_pun_dict.values(), color='r', label='Real', alpha=0.6)

plt.bar(fake_pun_dict.keys(), fake_pun_dict.values(), color='g', label='Fake', alpha=0.6)

plt.xlabel('symbol')

plt.ylabel('count')

plt.legend(loc='upper right')
fig4 = plt.figure(figsize=(10, 5), dpi=100)

train['pun_counts'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test['pun_counts'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

bins = 100

plt.hist(train[train.target == 0]['pun_counts'], alpha = 0.6, bins=bins, label='Fake', color='green')

plt.hist(train[train.target == 1]['pun_counts'], alpha = 0.6, bins=bins, label='Real', color='red')

plt.xlabel('punctuations counts')

plt.ylabel('count')

plt.xlim(0,40)

plt.legend(loc='upper right')
import regex as re

def remove_tag(text):

    tag =re.compile(r'<(.*?)>')

    return tag.sub(r'',text)



train['cleaned']=train['text'].apply(lambda x : remove_tag(x))

test['cleaned']=test['text'].apply(lambda x : remove_tag(x))
def remove_link(text):

    # all links start with 'http://t.co/' or 'https://t.co/'

    # the ? means 0 or 1, so s is optional in https

    link=re.compile(r"https?:\/\/t.co\/[A-Za-z0-9]+")

    return link.sub(r'',text)



train['cleaned']=train['cleaned'].apply(lambda x : remove_link(x))

test['cleaned']=test['cleaned'].apply(lambda x : remove_link(x))
def remove_mention(text):

    mention=re.compile(r"@[A-Za-z0-9_]+[ :]")

    return mention.sub(r'',text)



train['cleaned']=train['cleaned'].apply(lambda x : remove_mention(x))

test['cleaned']=test['cleaned'].apply(lambda x : remove_mention(x))
# testing the functions to remove link and tags

word = remove_link("word a b c <a>http://t.co/abdcd</a> hello @aria_ahrary @TheTawniest:hello")

word = remove_mention(word)

word = remove_tag(word)



print(word)
# code taken from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

def remove_emoji(text):

    emoji_pattern = re.compile("["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



train['cleaned']=train['cleaned'].apply(lambda x : remove_emoji(x))

test['cleaned']=test['cleaned'].apply(lambda x : remove_emoji(x))
# testing the functions to remove emoji

word = remove_emoji("Haha 😂")

print(word)
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

train['cleaned']=train['cleaned'].apply(lambda x : remove_punct(x))

test['cleaned']=test['cleaned'].apply(lambda x : remove_punct(x))
# testing the functions to remove punctuation

word = remove_punct("Hello. I am having a great time.")

print(word)
def remove_stopword(text):

    for word in STOPWORDS:

            token = " " + word + " "

            text = text.replace(token, " ")

            text = text.replace("  ", " ")

    return text



train['cleaned']=train['cleaned'].apply(lambda x : remove_stopword(x))

test['cleaned']=test['cleaned'].apply(lambda x : remove_stopword(x))
# testing the functions to remove punctuation

word = remove_stopword("Hello. I am having a great time.")

print(word)
train.tail()

# 7613 x 10
train_labels = np.array(train['target'])

print(train_labels)



#tweets = train['cleaned']

#tweets_test = test['cleaned']



tweets = train['text']

tweets_test = test['text']
from sklearn.model_selection import train_test_split

tweets_train, tweets_valid, labels_train, labels_valid = train_test_split(tweets, train_labels, test_size = 0.2, shuffle=True, random_state=0)

print("tweets_train shape:", tweets_train.shape)

print("labels_train shape:", labels_train.shape)



print("tweets_valid shape:", tweets_valid.shape)

print("labels_valid shape:", labels_valid.shape)
tweets.head()
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



vocab_size = 1000

embedding_dim=8

max_length = 27

num_epochs = 10

token = "<Unknown>"





# num_words=vocab_size

# oov_token="<Unknown>"

tokenizer = Tokenizer(num_words=vocab_size,oov_token=token)

tokenizer.fit_on_texts(tweets_train)

word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(tweets_train)

training_padded = pad_sequences(training_sequences, truncating = 'post', padding='post', maxlen=max_length)



valid_sequences = tokenizer.texts_to_sequences(tweets_valid)

valid_padded = pad_sequences(valid_sequences, truncating = 'post', padding='post', maxlen=max_length)

print("training_padded shape:",training_padded.shape)

print("valid_padded shape:",valid_padded.shape)
print(len(word_index))
import tensorflow as tf

model = tf.keras.Sequential([

    # make word embedding

    tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),

    #flatten the network or use pooling

    tf.keras.layers.Flatten(),

    # dense neural network

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid))
def plot_loss_and_accuracy(history, val,nrows,ncols,index):

    plt.subplot(nrows,ncols,index)

    plt.plot(history.history[val])

    plt.plot(history.history['val_'+val])

    plt.xlabel("# of Epochs")

    plt.ylabel(val)

    plt.legend([val, "val_"+val])



fig5, axes = plt.subplots(1,2, figsize=(10, 4), dpi=100)

plot_loss_and_accuracy(history,"accuracy",1,2,1)

plot_loss_and_accuracy(history,"loss",1,2,2)

plt.show()
vocab_size = 500

embedding_dim=16

max_length = 27

num_epochs = 10

token = "<Unknown>"





# num_words=vocab_size

# oov_token="<Unknown>"

tokenizer = Tokenizer(num_words=vocab_size,oov_token=token)

tokenizer.fit_on_texts(tweets_train)

word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(tweets_train)

training_padded = pad_sequences(training_sequences, truncating = 'post', padding='post', maxlen=max_length)



valid_sequences = tokenizer.texts_to_sequences(tweets_valid)

valid_padded = pad_sequences(valid_sequences, truncating = 'post', padding='post', maxlen=max_length)

model_LSTM = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(24,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model_LSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history_LSTM = model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid))
fig6, axes = plt.subplots(1,2, figsize=(10, 4), dpi=100)

plot_loss_and_accuracy(history_LSTM,"accuracy",1,2,1)

plot_loss_and_accuracy(history_LSTM,"loss",1,2,2)

plt.show()
model_GRU = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model_GRU.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history_GRU = model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid))
fig7, axes = plt.subplots(1,2, figsize=(10, 4), dpi=100)

plot_loss_and_accuracy(history_GRU,"accuracy",1,2,1)

plot_loss_and_accuracy(history_GRU,"loss",1,2,2)

plt.show()
model_Conv1D = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),

    tf.keras.layers.Conv1D(128,5,activation='relu'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model_Conv1D.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history_Conv1D = model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid))
fig8, axes = plt.subplots(1,2, figsize=(10, 4), dpi=100)

plot_loss_and_accuracy(history_Conv1D,"accuracy",1,2,1)

plot_loss_and_accuracy(history_Conv1D,"loss",1,2,2)

plt.show()
#import tensorflow_datasets as tfds



#tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for text in tweets_train), target_vocab_size = 2**10)
vocab_sizes = [250,500,1000,2000,5000]

embedding_dim=16

max_length = 27

num_epochs = 10

token = "<Unknown>"



models=[]



for vocab_size in vocab_sizes:

    tokenizer = Tokenizer(num_words=vocab_size,oov_token=token)

    tokenizer.fit_on_texts(tweets_train)

    word_index = tokenizer.word_index



    training_sequences = tokenizer.texts_to_sequences(tweets_train)

    training_padded = pad_sequences(training_sequences, truncating = 'post', padding='post', maxlen=max_length)



    valid_sequences = tokenizer.texts_to_sequences(tweets_valid)

    valid_padded = pad_sequences(valid_sequences, truncating = 'post', padding='post', maxlen=max_length)



    model = tf.keras.Sequential([

        # make word embedding

        tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),

        #flatten the network or use pooling

        tf.keras.layers.Flatten(),

        # dense neural network

        tf.keras.layers.Dense(6,activation='relu'),

        tf.keras.layers.Dense(1,activation='sigmoid')

    ])



    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



    models.append(model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid),verbose=0))

    print("Done with a round")



print("All Done")


fig6, axes = plt.subplots(5,2, figsize=(10, 15), dpi=100)

for i in range(len(models)):

    plot_loss_and_accuracy(models[i],"accuracy",5,2,(i+(i+1)))

    plot_loss_and_accuracy(models[i],"loss",5,2,(i+(i+2)))

plt.show()

vocab_sizes = 500

embedding_dims= [16,32]

max_length = 27

num_epochs = 10

token = "<Unknown>"



models=[]



for embedding_dim in embedding_dims:

    tokenizer = Tokenizer(num_words=vocab_size,oov_token=token)

    tokenizer.fit_on_texts(tweets_train)

    word_index = tokenizer.word_index



    training_sequences = tokenizer.texts_to_sequences(tweets_train)

    training_padded = pad_sequences(training_sequences, truncating = 'post', padding='post', maxlen=max_length)



    valid_sequences = tokenizer.texts_to_sequences(tweets_valid)

    valid_padded = pad_sequences(valid_sequences, truncating = 'post', padding='post', maxlen=max_length)



    model = tf.keras.Sequential([

        # make word embedding

        tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),

        #flatten the network or use pooling

        tf.keras.layers.Flatten(),

        # dense neural network

        tf.keras.layers.Dense(6,activation='relu'),

        tf.keras.layers.Dense(1,activation='sigmoid')

    ])



    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



    models.append(model.fit(training_padded, labels_train, epochs=num_epochs, validation_data=(valid_padded,labels_valid),verbose=0))

    print("Done with a round")



print("All Done")


fig7, axes = plt.subplots(2,2, figsize=(10, 10), dpi=100)

for i in range(len(models)):

    plot_loss_and_accuracy(models[i],"accuracy",2,2,(i+(i+1)))

    plot_loss_and_accuracy(models[i],"loss",2,2,(i+(i+2)))

plt.show()

vocab_size = 1000

embedding_dim=8

max_length = 27

num_epochs = 4

token = "<Unknown>"



tokenizer = Tokenizer(num_words=vocab_size,oov_token=token)

tokenizer.fit_on_texts(tweets)

word_index = tokenizer.word_index



all_sequences = tokenizer.texts_to_sequences(tweets)

all_padded = pad_sequences(all_sequences, truncating = 'post', padding='post', maxlen=max_length)



test_sequences = tokenizer.texts_to_sequences(tweets_test)

test_padded = pad_sequences(test_sequences, truncating = 'post', padding='post', maxlen=max_length)



model = tf.keras.Sequential([

        # make word embedding

    tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),

        #flatten the network or use pooling

    tf.keras.layers.Flatten(),

        # dense neural network

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



final = (model.fit(all_padded, train_labels, epochs=num_epochs, verbose=2))
prediction = model.predict(test_padded)
prediction_flatten = np.ravel(prediction) # flatten it in order to make into a series in a dataframe
submission = pd.DataFrame({

        "id": test.index,

        "target": prediction_flatten

    })



def threshold(val):

    if val >= 0.5:

        value = 1

    else:

        value = 0

    return value

submission['target'] = submission['target'].apply(threshold)



submission.head()
submission.to_csv('submission.csv', index=False)