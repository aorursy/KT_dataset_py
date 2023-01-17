# Loading basic packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
import string
train_df = pd.read_csv('../input/train.csv') # loading training set
test_df = pd.read_csv('../input/test.csv') # loading test set
print(train_df.shape)
print(test_df.shape)
train_df.info()
# Let's plot the distribution of classes
dist = train_df.groupby(["Sentiment"]).size()
dist = (dist / dist.sum())*100
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(dist.keys(), dist.values);
# We can also check out test dataset
test_df.head()
#Lets group by sentences at see how it looks 
dist = train_df.groupby(["SentenceId"]).size()
dist.head()
#Lets look at the average
dist.mean()
train_df['SentenceLength'] = train_df.Phrase.str.len()
train_df.head()
idx = train_df.groupby(['SentenceId'])['SentenceLength'].transform(max) == train_df['SentenceLength']
train_df = train_df[idx]
train_df.info()
# Let's check out the distribution after changes
dist = train_df.groupby(["Sentiment"]).size()
dist = (dist / dist.sum())*100
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(dist.keys(), dist.values);
import spacy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

!python -m spacy download en
nlp = spacy.load('en')

import collections
import collections as collect # collect.Counter
from collections import Counter
from collections import defaultdict

# Cleaning text before feeding it to spacy
punctuations = string.punctuation # punctuation characters
english_stopwords = stopwords.words('english') # dictionary with the most common English words

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-'] # lemmatizing text
        tokens = [tok for tok in tokens if tok not in english_stopwords and tok not in punctuations] # removing unwanted characters
        tokens = [tok for tok in tokens if re.match(r'[^\W\d]*$', tok)] # keeping alphabetic characters only
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
# Grab all text associated with negative
neg_text = [text for text in train_df[train_df['Sentiment'] == 0]['Phrase']]

# Grab all text associated with positive
pos_text = [text for text in train_df[train_df['Sentiment'] == 4]['Phrase']]
# Clean up all text for positive and negative reviews
neg_clean = cleanup_text(neg_text)
neg_clean = ' '.join(neg_clean).split()
pos_clean = cleanup_text(pos_text)
pos_clean = ' '.join(pos_clean).split()

# Count all unique words for both classes
neg_counts = Counter(neg_clean)
pos_counts = Counter(pos_clean)
# Let's plot top 25 most frequently occuring words for negative reviews
neg_common_words = [word[0] for word in neg_counts.most_common(25)]
neg_common_counts = [word[1] for word in neg_counts.most_common(25)]

plt.figure(figsize=(14, 12))
ax = sns.barplot(x=neg_common_words, y=neg_common_counts)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.title('Most common negative words')
plt.show()
# Let's plot top 25 most frequently occuring words for positive reviews
pos_common_words = [word[0] for word in pos_counts.most_common(25)]
pos_common_counts = [word[1] for word in pos_counts.most_common(25)]

plt.figure(figsize=(14, 12))
bx = sns.barplot(x=pos_common_words, y=pos_common_counts)
bx.set_xticklabels(bx.get_xticklabels(),rotation=45)
plt.title('Most common positive words')
plt.show()
#train_df = train_df.sample(n=1000) # take a sample of 1000 observations for training purposes
train_df['spacy_sentence_vec'] = train_df['Phrase'].map(lambda t: nlp(t).vector)
# the spacy_sentence_vec column should now contain a vector representation of each sentence with 384 dimensions
train_df.head()
# Let's start with default data with only the longest sentences
training = pd.read_csv('http://vindebryg.dk/sds_uni/train.tsv', sep='\t') # loading training set
testing = pd.read_csv('http://vindebryg.dk/sds_uni/test.tsv', sep='\t') # loading test set
idx = train_df.groupby(['SentenceId'])['SentenceLength'].transform(max) == train_df['SentenceLength']
train_df = train_df[idx]
# We need to define features once again
X = training['Phrase']
y = training['Sentiment']
# Splitting into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Vectorizing the text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,', stop_words="english") 
# CountVectorizer can lowercase letters and disregard punctuation & stopwords
X_cv = vect.fit_transform(X)
X_train_cv = vect.transform(X_train)
X_test_cv = vect.transform(X_test)
print (X_train_cv.shape)
from sklearn.naive_bayes import MultinomialNB
# Fitting into multinomial naive bayes on a train set
clf=MultinomialNB()
clf.fit(X_train_cv, y_train)
clf.score(X_test_cv, y_test)
# Let's check how is it doing on test set
X_test=vect.transform(testing['Phrase'])
# Firstly, I updated our vocabulary and transformed raw test data into vectorized form
clf=MultinomialNB()
clf.fit(X_cv, y)
predicted_result=clf.predict_proba(X_test)
predicted_result.shape
result=pd.DataFrame()
result["id"]=testing["PhraseId"]
result["negative"]=predicted_result[:,0]
result["som. negative"]=predicted_result[:,1]
result["neutral"]=predicted_result[:,2]
result["som. positive"]=predicted_result[:,3]
result["positive"]=predicted_result[:,4]
print(result[0:10])
# we take the values that we want to predict
y = train_df['Sentiment']
# Here are the values we want to explain it with
X = np.vstack(train_df['spacy_sentence_vec'])
X.shape
# Splitting into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# quick evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, classifier.predict(X_test)))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', 
                             use_idf=True, 
                             smooth_idf=True)
X = vectorizer.fit_transform(train_df['Phrase'])

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load up a classifier
classifier = LogisticRegression()

# Fit the model 
classifier.fit(X_train, y_train)
print(classification_report(y_test, classifier.predict(X_test)))
# Importing the keras library for deep learning

# Essential elements of Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# Lets change the predicted value format
y = train_df['Sentiment']
y = keras.utils.to_categorical(y, num_classes=5)
# Here are the values we want to predict it with
X = np.vstack(train_df['spacy_sentence_vec'])
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
classifier = Sequential()
# we build the layer from the ground up!

#### Input Layer ####
classifier.add(Dense(units = 384, activation='relu', input_dim = 384))
classifier.add(Dropout(rate = 0.3))

#### Hidden ####

classifier.add(Dense(units = 32, activation='relu'))
classifier.add(Dropout(rate = 0.3))

#### Output ####
classifier.add(Dense(units = 5, activation='softmax'))
# Compile the model
classifier.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
# The checkpointer is will save the best performing model for later use
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
# Let's fit the model into neural net
history = classifier.fit(X_train, y_train, batch_size= 32, epochs= 50, validation_data=(X_test, y_test), callbacks=[checkpointer])
# Let's check out how our network looks like
classifier.summary()
# summarize history for accuracy
plt.figure(figsize=(10,6));
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy in both training and test - Last: ' + str(round(history.history['val_acc'][-1], 3)))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# summarize history for loss
plt.figure(figsize=(10,6));
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss in both training and test - Last: ' + str(round(history.history['val_loss'][-1], 3)))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
classifier = load_model('model.h5')
classifier.evaluate(X_test,y_test)
#Import all the nessesary things
import sys, gc, os, re, csv, codecs, numpy as np, pandas as pd
import keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, Lambda
from keras.layers import GlobalMaxPool1D ,Bidirectional, SpatialDropout1D, CuDNNLSTM
from keras.models import Model, load_model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

%matplotlib inline

#Setup x and y for the model
y = list(train_df['Sentiment'])
#x = list(cleanup_text(train_df['Phrase'])) # Remove the stopwords
x = train_df['Phrase'] # Keep the stopwords
#Use the labelencoder to encode y
le = preprocessing.LabelEncoder()
le.fit(y)

def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

# One-hot encode, used to create the proper matrix
def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)
#Encode the data in the proper format
x_enc = x  #The text that has to be used
y_enc = encode(le, y) #The values to predict

x_train = np.asarray(x_enc[:])
y_train = np.asarray(y_enc[:]) 

# We can take a quick look at the data
print(x_train[-1])
print(y_train[-1])
# We select to use the ELMo encoder - it gave us the best performance.
elmo_url = "https://tfhub.dev/google/elmo/2" 
# We set the model to be trainable, this has given the best results.
embed = hub.Module(elmo_url, trainable=True)
#funciton to create ELMo model that we downloaded 
def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
# <model architecture>
input_text = Input(shape=(1,), dtype=tf.string)

embeddinglayer = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

dropout_layer = Dropout(0.6)(embeddinglayer)         

dense_layer = Dense(128, activation='relu')(dropout_layer)

pred = Dense(5, activation='softmax')(dense_layer)

model = Model(inputs=[input_text], outputs=pred)

#</end of model architecture>

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

#This way we can take a look at the model
model.summary()
#Setup the model
with tf.Session() as session:
    K.set_session(session)
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])  
    model_callback_list = []
    model_callback_list.append(ModelCheckpoint('elmo_v1.h5', monitor='val_acc',
                                                       verbose=1, 
                                                       save_best_only=True,
                                                       save_weights_only=True,
                                                       mode='max'))
    model_callback_list.append(EarlyStopping(monitor='val_acc',
                                       patience=10))  
    
    history = model.fit(x_train, y_train, epochs=50,                            
                                          batch_size=32, 
                                          shuffle=True, 
                                          verbose=1, 
                                          validation_split=0.2,
                                          callbacks=model_callback_list)
# Plot the model loss as before
plt.figure(figsize=(10,6));
plt.plot(history.history["val_loss"], label='Test loss');
plt.plot(history.history["loss"], label='Train loss');
plt.title('Loss in both training and test - Last: ' + str(round(history.history['loss'][-1], 3)))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend();
plt.show();
# Take a look at the model accuracy progress
plt.figure(figsize=(10,6));
plt.plot(history.history["val_acc"], label='Test acc');
plt.plot(history.history["acc"], label='Train acc');
plt.title('Accuracy in both training and test - Last:' + str(round(history.history['val_acc'][-1], 3)))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend();
plt.show();