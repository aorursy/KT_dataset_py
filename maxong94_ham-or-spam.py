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
df =pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding= "ISO-8859–1" )
print(df.columns)
print(df["v1"].notnull().value_counts())
print(df["v2"].notnull().value_counts())
print(df["Unnamed: 2"].notnull().value_counts())
print(df["Unnamed: 3"].notnull().value_counts())
print(df["Unnamed: 4"].notnull().value_counts())
df_clean= df.copy()
df_clean.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis = 1,inplace = True)
df_clean.columns = ["type","text"]
df_clean["type"].value_counts()
df_clean.head(5)

# within ham texts
ham = df_clean[df_clean["type"] == "ham"]
spam = df_clean[df_clean["type"] == "spam"]

import re
def show(index):
    ham_list = ham["text"].tolist()
    text = ham_list[index]
    print(text)
    pattern = re.compile(r"[^\w]")
    print(re.sub(pattern," ",text))

show(22)
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns 
stopwords_list = stopwords.words('english')
def clean_text(df):
    cleaned_reviews = []
    text = df["text"].tolist()
    vocab = Counter()
    for sentence in text:
        pattern = re.compile(r"[^\w]")
        sentence = re.sub(pattern," ",sentence)
        # tokenize each sentence into a list of words 
        token_list = word_tokenize(sentence)
        # change all tokens to lower caps 
        token_list= [tokens.lower() for tokens in token_list]
        # remove words that are in the stopwords list
        cleaned = [tokens for tokens in token_list if tokens not in stopwords_list]
        cleaned = [tokens for tokens in cleaned if tokens.isalpha()]
        w = [tokens for tokens in cleaned if len(tokens) >1]
        vocab.update(w)
        cleaned = " ".join(w)
        cleaned_reviews.append(cleaned)
    return vocab, cleaned_reviews

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
def plot_word_count(df):
    # call the function to get our vocab count and cleaned_reviews 
    vocab,cleaned_reviews = clean_text(df)
    # change counter to dictionary object
    dict_vocab = dict(vocab)
    # sort dictionary base on values 
    sorted_vocab_dict = sorted(dict_vocab.items(),key = lambda x: x[1],reverse = True)
    # create x and y list to append to for plotting 
    y = []
    x = []
    for i in sorted_vocab_dict[:50]:
        y.append(i[1])
        x.append(i[0])
    
    fig,ax = plt.subplots(figsize = (20,8))
    plot = ax.bar(x,y)
    plt.xticks(rotation = 50)
    plt.title("{}".format(df["type"].to_numpy()[0]))
    return plot 

plot_word_count(ham)
plot_word_count(spam)

ham_vocab,ham_cleaned_reviews = clean_text(ham)
spam_vocab,spam_cleaned_reviews = clean_text(spam)

# find length of maximum sms within sms ham
max_ham_sms = len(max(ham_cleaned_reviews,key = len))
max_spam_sms = len(max(spam_cleaned_reviews,key = len))
print("Longest ham sms :", max_ham_sms)
print("Longest spam sms : ",max_spam_sms )
print("No of unique tokens for ham",len(ham_vocab))
print("No of unique tokens for spam",len(spam_vocab))

ham_sms_size = [len(i) for i in ham_cleaned_reviews]
spam_sms_size = [len(i) for i in spam_cleaned_reviews]
fig,(ax1,ax2) = plt.subplots(2,1,figsize = (20,8))
ax1.boxplot(ham_sms_size,vert = False)
ax1.title.set_text("Distribution of sms length for ham")
ax2.boxplot(spam_sms_size,vert = False)
ax2.title.set_text("Distribution of sms length for spam")

def percent(pad_limit):
    num_of_sms_less_than_pad_limit = [i for i in ham_sms_size if i < pad_limit]
    percent = len(num_of_sms_less_than_pad_limit)/len(ham_sms_size)* 100
    return percent
percent(300)
vocab_clean, cleaned_reviews = clean_text(df_clean)
print(len(vocab_clean))
print(len(cleaned_reviews))
cleaned_reviews[0]
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_words_to_keep = 10000
# according to the distribution of 
max_sequence_length = 300
tokenizer = Tokenizer(num_words = max_words_to_keep)
# here our inputs are our list of cleaned_reviews 
tokenizer.fit_on_texts(cleaned_reviews)
# the wordIndex is the same as our vocab counter 
wordIndex = tokenizer.word_index

#to transform all the texts from cleaned_reviews to sequences of integers. 

X = tokenizer.texts_to_sequences(cleaned_reviews)
X = pad_sequences(X,maxlen = max_sequence_length)
print("Shape of X tensor: ",X.shape)
y = df_clean["type"].replace({"ham":0,"spam":1}).to_numpy()
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42, test_size = 0.2)
print("X_train size : ",X_train.shape)
print("X_test size : ",X_test.shape)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import keras
METRICS = [keras.metrics.TruePositives(name = 'tp'),
           keras.metrics.FalsePositives(name = 'fp'),
           keras.metrics.TrueNegatives(name = "tn"),
           keras.metrics.FalseNegatives(name = "fn"),
           keras.metrics.BinaryAccuracy(name = "accuracy"),
           keras.metrics.Precision(name = "precision"),
           keras.metrics.Recall(name = "recall"),
           keras.metrics.AUC(name = "auc")]

embedding_dim = 64
def make_model(output_bias = None):
    model = Sequential([
    Embedding(max_words_to_keep,embedding_dim, input_length = max_sequence_length),
    SpatialDropout1D(0.8),    
    LSTM(10,dropout = 0.8),
    Dense(1, activation = "sigmoid",bias_initializer = output_bias)
    ])
    model.compile(loss = "binary_crossentropy",optimizer = "adam",metrics = 'accuracy')
    return model 
model= make_model()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes = True)
Epochs = 30
batch_size = 64
file_path = 'model1.h5'
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_path, save_freq = 'epoch')


history = model.fit(X_train, y_train, epochs = Epochs, batch_size = batch_size, validation_split = 0.1,callbacks=[early_stopping, model_checkpoint])

history.history
fig,ax = plt.subplots(1,2,figsize = (20,8))
ax[0].plot(history.epoch,history.history["loss"])
ax[0].plot(history.epoch,history.history["val_loss"])
ax[0].legend(["training","validation"])
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.epoch,history.history["accuracy"])
ax[1].plot(history.epoch,history.history["val_accuracy"])
ax[1].legend(["training","validation"])
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
y_test_pred = model.predict_classes(X_test, batch_size = 64)
y_train_pred = model.predict_classes(X_train, batch_size = 64)

from sklearn.metrics import confusion_matrix 
def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
  
    print('Ham Detected (True Negatives): ', cm[0][0])
    print('Ham Incorrectly Detected (False Positives): ', cm[0][1])
    print('Spam Missed (False Negatives): ', cm[1][0])
    print('Spam Detected (True Positives): ', cm[1][1])
    print('Total Spam in dataset: ', np.sum(cm[1]))
    print('Total Ham in dataset:', np.sum(cm[0]))

plot_cm(y_train,y_train_pred)


plot_cm(y_test,y_test_pred)

accr = model.evaluate(X_test,y_test)

print("Accuracy of model on test data: ",accr[1])
embedding_dim = 64
def make_model(output_bias = None):
    model = Sequential([
    Embedding(max_words_to_keep,embedding_dim, input_length = max_sequence_length),
    Dropout(0.8, noise_shape = (1,embedding_dim)),    
    LSTM(10,dropout = 0.8),
    Dense(1, activation = "sigmoid",bias_initializer = output_bias)
    ])
    model.compile(loss = "binary_crossentropy",optimizer = "adam",metrics = 'accuracy')
    return model 
model2= make_model()
model2.summary()
plot_model(model2,show_shapes = True)
file_path2 = 'model2.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_path2, save_freq = 'epoch')

history2 = model2.fit(X_train, y_train, epochs = Epochs, batch_size = batch_size, validation_split = 0.1,callbacks=[early_stopping, model_checkpoint])

fig,ax = plt.subplots(1,2,figsize = (20,8))
ax[0].plot(history2.epoch,history2.history["loss"])
ax[0].plot(history2.epoch,history2.history["val_loss"])
ax[0].legend(["training","validation"])
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history2.epoch,history2.history["accuracy"])
ax[1].plot(history2.epoch,history2.history["val_accuracy"])
ax[1].legend(["training","validation"])
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
accr = model2.evaluate(X_test,y_test)
print("Accuracy of model on test data: ",accr[1])