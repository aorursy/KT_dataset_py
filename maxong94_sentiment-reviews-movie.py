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
import tensorflow as tf
from tensorflow import keras
imdb = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
imdb.head(5)

imdb['sentiment'].value_counts()
#import re 
#import nltk
#import string
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#cleaned_reviews = []
#docu = document
#docu = docu.lower()
#docu = re.sub(r"<br />"," ",docu)
#pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")
## substitute the characters above with 
#docu = re.sub(pattern,"", docu)
## tokenize them so we can remove the stopwords from english language
#tokens = word_tokenize(docu)
#stop_words = set(stopwords.words("english"))
#tokens = [w for w in tokens if w not in stop_words]
#table = str.maketrans("","",string.punctuation)
#stripped = [w.translate(table) for w in tokens]
#PS = PorterStemmer()
#cleaned_tokens = [PS.stem(w) for w in stripped if w.isalpha()]
#joined_tokens = " ".join(cleaned_tokens)
#cleaned_reviews.append(joined_tokens)
#cleaned_reviews

#Bag of words model
#This will be our tutorial to show how the various metheods of producing encoding through the bag of words model will work in terms of the performance of our sentiment reviews movie model. 
#
#We will use the Tokenizer in Keras API to score words. 
#
#1) binary 
#
#2) Count 
#
#3) tfidf 
#
#4) frequency 


import re 
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter

stopwords_list = stopwords.words('english')
def clean_text(df):
    cleaned_reviews= []
    reviews = df['review'].tolist()
    vocab = Counter()
    # change each element within the list to lower punctuation
    """Docu will be a each row in the dataframe"""
    for sentence in reviews:
        sentence = re.sub(r"<br />"," ",sentence)
        pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")
        # substitute the characters above with 
        sentence = re.sub(pattern," ", sentence)
        # tokenize them so we can remove the stopwords from english language
        token_list = word_tokenize(sentence)
        token_list = [tokens.lower() for tokens in token_list]
        # remove stopwords from english 
        token_list = [tokens for tokens in token_list if tokens not in stopwords_list]
        # remove words if they are less than 2
        token_list = [tokens for tokens in token_list if len(tokens) >2 ]
        cleaned_tokens = [tokens for tokens in token_list if tokens.isalpha()]
        vocab.update(cleaned_tokens)
        joined_tokens = " ".join(cleaned_tokens)
        cleaned_reviews.append(joined_tokens)
    return vocab, cleaned_reviews
imdb_copy= imdb.copy()
imdb_copy['review'][0]
df_positive_reviews = imdb_copy[imdb_copy['sentiment'] == 'positive']
df_negative_reviews = imdb_copy[imdb_copy['sentiment'] == 'negative']

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
    plt.title("{}".format(df["sentiment"].to_numpy()[0]))
    return plot 

plot_word_count(df_positive_reviews)
plot_word_count(df_negative_reviews)

from tensorflow.keras.preprocessing.text import Tokenizer 

max_nb_words = 50000

max_sequence_length = 250

embedding_dim = 100

tokenizer = Tokenizer(num_words = max_nb_words,
             lower = True)

vocab,cleaned_reviews = clean_text(imdb_copy)
cleaned_reviews[:2]
# to fit the tokenzier
tokenizer.fit_on_texts(cleaned_reviews)
word_index = tokenizer.word_index
print("Unique tokens:", len(word_index))
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(cleaned_reviews)
# pre padding
X = pad_sequences(X,maxlen = max_sequence_length)
print("shape of data tensor:", X.shape)
y = imdb_copy["sentiment"].replace({"positive":1,"negative":0}).to_numpy()
len(y)
#After cleaning, let us use the TF-IDF vectorizer to help us transform our document and vocab as inputs for our model. 
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest 
# from sklearn.feature_selection import f_classif
# # we will use unigrams and bigrams in our vocab list 
# 
# k_value = 20000
# def tfid_vectorize(texts, labels):
#     tfidf_vectorizer = TfidfVectorizer(# both unigrams and bigrams
#                         ngram_range = (1,2),
#                         # whether the feature will be made of word or character n-gram
#                         analyzer = "word",
#                         # cut of minimum number of times vocab word to appear in document. Minimum 2 to be accepted as vocab 
#                         min_df = 2)
#     text_transformed = tfidf_vectorizer.fit_transform(texts)
#     
#     # select best k features, with feature importance measured by f_classif
#     # set k = 20000
#     # initialize
#     selector = SelectKBest(score_func = f_classif, k = min(k_value,text_transformed.shape[1]))
#     selector.fit(text_transformed,labels)
#     transformed_texts = selector.transform(text_transformed)
#     return transformed_texts
#vectorized_data = tfid_vectorize(X_cleaned,y)

#print("Shape of transformed text matrix",vectorized_data.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size = 0.2,
                                                 random_state = 42)
print("Shape of training data",X_train.shape," Shape of y_train",y_train.shape)
print("Shape of test data",X_test.shape, " Shape of y_test",y_test.shape)
n_words = X_train.shape[1]
n_words
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Dropout
def define_model(n_words):
    model = Sequential()
    model.add(Embedding(max_nb_words,embedding_dim,input_length = n_words))
    # similar to dropout, but drops the whole slice along 0 axis. 
    #model.add(SpatialDropout1D(0.9))
    model.add(Dropout(rate = 0.9, noise_shape = (1,embedding_dim)))
    model.add(LSTM(5,dropout = 0.8, recurrent_dropout = 0.3)) # this dropout drops the inputs and outputs, not the hidden states of our LSTM
    # sigmoid because only 2 classes, if there are more than 2 classes, use softmax
    model.add(Dense(1,activation = "sigmoid"))
    # compile network
    model.compile(loss = "binary_crossentropy",optimizer = "adam",metrics = ["accuracy"])
    model.summary()
    return model
model = define_model(n_words)
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes = True)
EPOCHS = 40
BATCH_SIZE = 64
file_path = 'model.h5'
# Create callback for early stopping on validation loss. If the loss does
# not decrease on two consecutive tries, stop training
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_path, save_freq = 'epoch')
# put multiple call backs?


model.load_weights('model.h5')
# Train and validate model
# To start training, call the model.fit method—the model is "fit" to the training data.
# Note that fit() will return a History object which we can use to plot training vs. validation accuracy and loss.

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=BATCH_SIZE, callbacks=[early_stopping,model_checkpoint])
history.history
import matplotlib.pyplot as plt
# Let's plot training and validation accuracy as well as loss.
def plot_history(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(accuracy) + 1)
    
    # Plot accuracy  
    plt.figure(1)
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.figure(2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)
accr = model.evaluate(X_test,y_test)
print("Test set \n Loss: {} \n Accuracy: {}".format(accr[0],accr[1]))
