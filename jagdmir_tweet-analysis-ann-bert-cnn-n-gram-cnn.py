# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import xticks

from nltk.corpus import stopwords

import nltk

import re

from nltk.stem import WordNetLemmatizer

import string

from nltk.tokenize import word_tokenize

from nltk.util import ngrams

from collections import defaultdict

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

import tensorflow as tf

from sklearn.metrics import f1_score

from wordcloud import WordCloud,STOPWORDS

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from keras.preprocessing.sequence import pad_sequences

from numpy import array

from numpy import asarray

from numpy import zeros

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense,Flatten,Embedding,Activation,Dropout

from keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D,LSTM
# load train and test datasets

train= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
# check the no. of rows and columns in the dataset

train.shape, test.shape
train.head()
train.isnull().sum().sort_values(ascending = False)
# function to draw bar plot

def draw_bar_plot(category,length,xlabel,ylabel,title,sub):

    plt.subplot(2,2,sub)

    plt.bar(category, length,color = 'grbkymc')

    plt.legend()

    plt.xlabel(xlabel, fontsize=15)

    plt.ylabel(ylabel, fontsize=15)

    plt.title(title, fontsize=15)

    #plt.show()
# function to draw histogram

def draw_hist(xlabel, ylabel,title,target,sub,color):

    plt.subplot(1,2,sub)

    plt.hist(train[train.target==target]["length"],color = color)

    plt.title(title,fontsize=25)

    plt.xlabel(xlabel,fontsize=15)

    plt.ylabel(ylabel,fontsize=15)

    plt.ylim(0,1200)

    plt.grid()
# function to draw graphs for stopwords and punctuations

def draw_bar_n_plot(data,title):

# lets visualize the top 10 stop words

    x,y=zip(*data)



    plt.figure(figsize = (25,10))

    plt.subplot(1,2,1)

    plt.bar(x,y,color='grbkymc')

    plt.title("Top 10 "+ title,fontsize=25)

    plt.xlabel(title,fontsize=15)

    plt.ylabel("Count",fontsize=15)

    plt.grid()



    plt.subplot(1,2,2)

    plt.plot(x,y,'g')

    plt.title("Top 10 "+ title,fontsize=25)

    plt.xlabel(title,fontsize=15)

    plt.ylabel("Count",fontsize=15)

    plt.grid()
# check class distribution



print("No. of Real Disaster Tweets (Target = 1):",len(train[train["target"]==1]))

print("No. of Fake Disaster Tweets (Target = 0):",len(train[train["target"]==0]))
# lets visualize the class distribution

plt.figure(figsize = (12,8))

draw_bar_plot(["Real","Fake"],[len(train[train.target==1]), len(train[train.target==0])],"Real Vs Fake","Number of Tweets","Class Distribution",1)
# we will now check the length of "real disaster" vs lenght of "fake disaster" tweets

# lets first add a new field to the dataset called "length"

def length(text):    

    return len(text)



train["length"]= train.text.apply(length)
# lets see the distribution of length of tweets real vs fake



plt.figure(figsize = (20,8))

draw_hist("Real Disaster Tweets","Length of Tweets","Length of Real Disaster Tweets",1, 1,"darkgreen")

draw_hist("Fake Disaster Tweets","Length of Tweets","Length of Fake Disaster Tweets",0, 2,"darkred")

# lets check the average lenght of real vs fake tweets

print(train.groupby("target").mean()["length"].sort_values(ascending = False))



# lets visualize the class distribution

plt.figure(figsize = (12,8))

draw_bar_plot(["Real","Fake"],[train[train.target==1].mean()["length"], train[train.target==0].mean()["length"]],"Real Vs Fake","Average Length","Average Text Length - Real Vs Fake",1)
# lets drop the column

train.drop("length",1,inplace=True)
#lets save stopwords in a variable

stop = list(stopwords.words("english"))
# stopwords present in the whole dataset

sw = []

for message in train.text:

    for word in message.split():

        if word in stop:

            sw.append(word)





# lets convert the list to a dictinoary which would contain the stop words and their frequency

wordlist = nltk.FreqDist(sw)

# lets save the 10 most frequent stopwords

top10 = wordlist.most_common(10)
# Graphs for top 10 stopwords present in all the tweets

draw_bar_n_plot(top10,"Stopwords")
# save list of punctuation/special characters in a variable

punctuation = list(string.punctuation)
# punctuations present in all the tweets 

pun = []

for message in train.text:

    for word in message.split():

        if word in punctuation:

            pun.append(word)





# lets convert the list to a dictinoary which would contain the punctuations and their frequency

wordlist = nltk.FreqDist(pun)

# lets save the 10 most frequent stopwords

top10 = wordlist.most_common(10)
# draw graphs for top10 Punctuations

draw_bar_n_plot(top10,"Punctuations")
# Let's check stop words and punctuations in "Real Disaster Tweets"

stop_real = []

pun_real  = []

for message in train[train.target==1]["text"]:

    for word in message.split():

        if word in stop:

            stop_real.append(word)

        if word in punctuation:

            pun_real.append(word)





# lets convert the list to a dictinoary which would contain the stop word and its frequency

stop_real_wordlist = nltk.FreqDist(stop_real)

pun_real_wordlist =  nltk.FreqDist(pun_real)



# lets save the 10 most frequent stopwords

stop_real_top10 = stop_real_wordlist.most_common(10)

pun_real_top10  = pun_real_wordlist.most_common(10)
# Let's check "Fake Disaster Tweets" and create a list of stop words and punctuations

stop_fake = []

pun_fake  = []

for message in train[train.target==0]["text"]:

    for word in message.split():

        if word in stop:

            stop_fake.append(word)

        if word in punctuation:

            pun_fake.append(word)





# lets convert the list to a dictinoary which would contain the stop word and its frequency

stop_fake_wordlist = nltk.FreqDist(stop_fake)

pun_fake_wordlist =  nltk.FreqDist(pun_fake)



# lets save the 10 most frequent stopwords

stop_fake_top10 = stop_fake_wordlist.most_common(10)

pun_fake_top10  = pun_fake_wordlist.most_common(10)
x_stop_real,y_stop_real=zip(*stop_real_top10)

x_pun_real, y_pun_real =zip(*pun_real_top10)



x_stop_fake,y_stop_fake=zip(*stop_fake_top10)

x_pun_fake, y_pun_fake=zip(*pun_fake_top10)





plt.figure(figsize = (30,30))

draw_bar_plot(x_stop_real,y_stop_real,"Stopwords","Count","Top 10 Stopwords - Real Tweets",1)

draw_bar_plot(x_stop_fake,y_stop_fake,"Stopwords","Count","Top 10 Stopwords - Fake Tweets",2)

draw_bar_plot(x_pun_real,y_pun_real,"Punctuations","Count","Top 10 Punctuations - Real Tweets",3)

draw_bar_plot(x_pun_fake,y_pun_fake,"Punctuations","Count","Top 10 Punctuations - Fake Tweets",4)
# create an object to convert the words to its lemma form

lemma = WordNetLemmatizer()
# lets make a combine list of stopwords and punctuations

sw_pun = stop + punctuation
# function to preprocess the messages

def preprocess(tweet):

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet) # removing urls 

    tweet = re.sub('[^\w]',' ',tweet) # remove embedded special characters in words (for example #earthquake)         

    tweet = re.sub('[\d]','',tweet) # this will remove numeric characters

    tweet = tweet.lower()

    words = tweet.split()  

    sentence = ""

    for word in words:     

        if word not in (sw_pun):  # removing stopwords & punctuations                

            word = lemma.lemmatize(word,pos = 'v')  # converting to lemma    

            if len(word) > 3: # we will consider words with length  greater than 3 only

                sentence = sentence + word + ' '             

    return(sentence)
# apply preprocessing functions on the train and test datasets

train['text'] = train['text'].apply(lambda s : preprocess(s))

test ['text'] = test ['text'].apply(lambda s : preprocess(s))
# function to remove emojis

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
# applying the function on the train and the test datasets

train['text'] = train['text'].apply(lambda s : remove_emoji(s))

test ['text'] = test ['text'].apply(lambda s : remove_emoji(s))

# function to create vocab

from collections import Counter

def create_vocab(df):

    vocab = Counter()

    for i in range(df.shape[0]):

        vocab.update(df.text[i].split())

    return(vocab)



# concatenate training and testing datasets

master=pd.concat((train,test)).reset_index(drop=True)



# call vocabulary creation function on master dataset

vocab = create_vocab(master)



# lets check the no. of words in the vocabulary

len(vocab)
# lets check the most common 50 words in the vocabulary

vocab.most_common(50)

# create the final vocab by considering words with more than one occurence

final_vocab = []

min_occur = 2

for k,v in vocab.items():

    if v >= min_occur:

        final_vocab.append(k)
# lets check the no. of the words in the final vocabulary

len(final_vocab)

# function to filter the dataset, keep only words which are present in the vocab

def filter(tweet):

    sentence = ""

    for word in tweet.split():  

        if word in final_vocab:

            sentence = sentence + word + ' '

    return(sentence)
# apply filter function on the train and test datasets

train['text'] = train['text'].apply(lambda s : filter(s))

test ['text'] = test ['text'].apply(lambda s : filter(s))
# lets take a look at the update training dataset

train.text.head()
# lets create seperate datasets from real and fake tweets

real = train[train.target==1].reset_index()

fake = train[train.target==0].reset_index()
# function to create top 100 n-grams

def get_ngrams(data,n):

    all_words = []

    for i in range(len(data)):

        temp = data["text"][i].split()

        for word in temp:

            all_words.append(word)



    tokenized = all_words

    esBigrams = ngrams(tokenized, n)



    esBigram_wordlist = nltk.FreqDist(esBigrams)

    top100 = esBigram_wordlist.most_common(100)

    top100 = dict(top100)

    df_ngrams = pd.DataFrame(sorted(top100.items(), key=lambda x: x[1])[::-1])

    return df_ngrams
# function to visualize the top 100 n-grams in real and fake disaster tweets

def draw_barplots(real,fake,title):

    plt.figure(figsize = (40,80),dpi=100)



    plt.subplot(1,2,1)

    sns.barplot(y=real[0].values[:100], x=real[1].values[:100], color='green')

    plt.title("Top 100" + title + "in Real Tweets",fontsize=15)

    

    plt.subplot(1,2,2)

    sns.barplot(y=fake[0].values[:100], x=fake[1].values[:100],color='red')

    plt.title("Top 100" + title + "in Fake Tweets",fontsize=15)
# lets create top 100 unigrams

real_unigrams = get_ngrams(real,1)

fake_unigrams = get_ngrams(fake,1)
# lets visualize top 100 unigrams

draw_barplots(real_unigrams,fake_unigrams," Unigrams ")
# lets create top 100 bigrams

real_bigrams = get_ngrams(real,2)

fake_bigrams = get_ngrams(fake,2)
# lets visualize top 100 bigrams



draw_barplots(real_bigrams,fake_bigrams," Bigrams ")
# lets create top 100 trigrams

real_trigrams = get_ngrams(real,3)

fake_trigrams = get_ngrams(fake,3)
# lets visualize top 100 trigrams

draw_barplots(real_trigrams,fake_trigrams," Trigrams ")
def word_cloud(df):

    comment_words = '' 

    stopwords = set(STOPWORDS) 



    # iterate through the csv file 

    for val in df.text: 



        # typecaste each val to string 

        val = str(val) 



        # split the value 

        tokens = val.split() 

        

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower()

        

        comment_words += " ".join(tokens)+" "

        #return comment_words



    wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white', 

            stopwords = stopwords, 

            min_font_size = 10).generate(comment_words) 

  

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.show() 
# world cloud for real disaster tweets

word_cloud(real)
# world cloud for fake disaster tweets

word_cloud(fake)
# function to calculate f1 score for each epoch

import keras.backend as K

def get_f1(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val

# Bag of Words model

from keras.preprocessing.text import Tokenizer



# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
# lets use only tweet text to build the model

X = train.text

y = train.target



test_id = test.id

test.drop(["id","location","keyword"],1,inplace = True)
# Test train split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# create and apply tokenizer on the training dataset

tokenizer = create_tokenizer(X_train)

X_train_set = tokenizer.texts_to_matrix(X_train, mode = 'freq')

# define the model

def define_model(n_words):

    # define network

    model = Sequential()

    model.add(Dense(128, input_shape=(n_words,), activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # compile network

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = [get_f1])

    # summarize defined model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
# create the model

n_words = X_train_set.shape[1]

model = define_model(n_words)
#fit network

model.fit(X_train_set,y_train,epochs=10,verbose=2)
# prediction on the test dataset

X_test_set = tokenizer.texts_to_matrix(X_test, mode = 'freq')

y_pred = model.predict_classes(X_test_set)
# important metrices

print(classification_report(y_test, y_pred))
# apply tokenizer on the test dataset

test_set = tokenizer.texts_to_matrix(test.text, mode = 'freq')
# make predictions on the test dataset

y_test_pred = model.predict_classes(test_set)
# lets prepare for the prediction submission

sub = pd.DataFrame()

sub['Id'] = test_id

sub['target'] = y_test_pred

sub.to_csv('submission_1.csv',index=False)
# Fitting a tokenizer on text will create a list of unique words with an integer assigned to it

t = Tokenizer()

t.fit_on_texts(X_train.tolist())
# lets save the size of the vocab

vocab_size = len(t.word_index) + 1
# load the whole embedding into memory

embeddings_index = dict()

f = open('../input/glove6b100dtxt/glove.6B.100d.txt', mode='rt', encoding='utf-8')

for line in f:

    values = line.split()

    word = values[0]

    coefs = asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# we will now perform the encoding

encoded_docs = t.texts_to_sequences(X_train.tolist())



# embedding layer require all the encoded sequences to be of the same length, lets take max lenght as 100

# and apply padding on the sequences which are of lower size

max_length = 100

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)
# create a weight matrix for words in training docs

mis_spelled = []

embedding_matrix = zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

    else:

        mis_spelled.append(word)
# lets check how many words are not spelled correctly 

len(mis_spelled)
# define model

model = Sequential()

e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)

model.add(e)

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[get_f1])

# summarize the model

model.summary()

# fit the model

model.fit(padded_docs, y_train, epochs=50, verbose=0)
loss, accuracy = model.evaluate(padded_docs, y_train, verbose=0)
print(accuracy)
encoded_docs = t.texts_to_sequences(X_test.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# prediction on the test dataset

y_pred = model.predict_classes(padded_docs)
encoded_docs = t.texts_to_sequences(test.text.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
y_test_pred = model.predict_classes(padded_docs)
# lets prepare for the prediction submission

sub = pd.DataFrame()

sub['Id'] = test_id

sub['target'] = y_test_pred

sub.to_csv('submission_2.csv',index=False)
# fit a tokenizer

#we have already fit a tokenizer on our data, pasting the same code again

#t = Tokenizer()

#t.fit_on_texts(X_train.tolist())
max_length = max([len(s) for s in train.text])

print('Maximum length: %d' % max_length)
# we will now perform the encoding

encoded_docs = t.texts_to_sequences(X_train.tolist())



# embedding layer require all the encoded sequences to be of the same length, lets take max lenght as 100

# and apply padding on the sequences which are of lower size



padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)
# define vocabulary size

#  we laready have vocab size

vocab_size
# define the model

def define_model(vocab_size, max_length):

    model = Sequential()

    model.add(Embedding(vocab_size, 100, input_length=max_length))

    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # compile network

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
# define model

model = define_model(vocab_size, max_length)

# fit network

model.fit(padded_docs, y_train, epochs=10, verbose=2)
loss, accuracy = model.evaluate(padded_docs, y_train, verbose=0)
print(accuracy)
encoded_docs = t.texts_to_sequences(X_test.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# prediction on the test dataset

y_pred = model.predict_classes(padded_docs)
encoded_docs = t.texts_to_sequences(test.text.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
y_test_pred = model.predict_classes(padded_docs)
# lets prepare for the prediction submission

sub = pd.DataFrame()

sub['Id'] = test_id

sub['target'] = y_test_pred

sub.to_csv('submission_cnn.csv',index=False)
# save the model

model.save('model.h5')
from keras.layers import Input

from keras.layers.merge import concatenate

from keras.models import Model

# define the model 

def define_model(length, vocab_size): 

    # channel 1 

    inputs1 = Input(shape=(length,)) 

    embedding1 = Embedding(vocab_size, 100)(inputs1) 

    conv1 = Conv1D(32, 4, activation='relu')(embedding1) 

    drop1 = Dropout(0.5)(conv1) 

    pool1 = MaxPooling1D()(drop1) 

    flat1 = Flatten()(pool1)

    

    # channel 2 

    inputs2 = Input(shape=(length,)) 

    embedding2 = Embedding(vocab_size, 100)(inputs2) 

    conv2 = Conv1D(32, 6, activation='relu')(embedding2) 

    drop2 = Dropout(0.5)(conv2) 

    pool2 = MaxPooling1D()(drop2) 

    flat2 = Flatten()(pool2) 

    

    # channel 3 

    inputs3 = Input(shape=(length,)) 

    embedding3 = Embedding(vocab_size, 100)(inputs3) 

    conv3 = Conv1D(32, 8, activation='relu')(embedding3) 

    drop3 = Dropout(0.5)(conv3) 

    pool3 = MaxPooling1D()(drop3) 

    flat3 = Flatten()(pool3)

    

    # merge 

    merged = concatenate([flat1, flat2, flat3]) 

    # interpretation 

    dense1 = Dense(10, activation='relu')(merged) 

    outputs = Dense(1, activation='sigmoid')(dense1) 

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs) 

    # compile 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # summarize 

    model.summary() 

    plot_model(model, show_shapes=True, to_file='multichannel.png') 

    return model

# we will now perform the encoding

encoded_docs = t.texts_to_sequences(X_train.tolist())



# embedding layer require all the encoded sequences to be of the same length, lets take max lenght as 100

# and apply padding on the sequences which are of lower size



padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define model

model = define_model(max_length,vocab_size)

# fit network

model.fit([padded_docs,padded_docs,padded_docs], array(y_train), epochs=7, batch_size=16)
loss, accuracy = model.evaluate([padded_docs,padded_docs,padded_docs], y_train, verbose=0)
print(accuracy)
encoded_docs = t.texts_to_sequences(X_test.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# prediction on the test dataset

_, acc = model.evaluate([padded_docs,padded_docs,padded_docs], array(y_test), verbose=0) 

print('Train Accuracy: %.2f' % (acc*100)) 
encoded_docs = t.texts_to_sequences(test.text.tolist())

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
y_test_pred = model.predict([padded_docs,padded_docs,padded_docs])
# lets prepare for the prediction submission

sub = pd.DataFrame()

sub['Id'] = test_id

sub['target'] = y_test_pred

sub.to_csv('submission_multi-cnn.csv',index=False)
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
# load train and test datasets

train= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
### Add tokens to the data make it BERT compatible

def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)
test_pred = model.predict(test_input)
submission=pd.DataFrame()

submission['Id']=test_id

submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission_3.csv', index=False)
