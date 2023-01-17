# %pip install gensim

# %pip install scikit-plot

#%pip install PrettyTable
%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sqlite3 as sql

import seaborn as sns

import re

from nltk.corpus import stopwords

import nltk

nltk.download('stopwords')

import time

# import umap

#pip install PrettyTable

#pip install scikit-plot

#pip install gensim

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

import scikitplot as skplt

import gensim

from datetime import timedelta

import os

from scipy import sparse

from prettytable import PrettyTable

from itertools import product

from mpl_toolkits.mplot3d import Axes3D



import keras

from keras.datasets import imdb

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

# fix random seed for reproducibility

np.random.seed(7)

# import plotly_express as px

# from plotly.offline import plot
# for kaggle comment out below code

os.chdir("../input/amazon-fine-food-reviews/")
#establishing the sql connection

amazon_df_con = sql.connect('database.sqlite')



#reading the data from sql connection

amazon_df = pd.read_sql("SELECT * FROM Reviews WHERE Score!= 3", con=amazon_df_con)



print("size of our dataset is", amazon_df.shape)

amazon_df.head()
#lets consider every review having score greater than 3 as positive

# and less than 3 as negative 

amazon_df['Score'] = amazon_df.Score.apply(lambda x: 'Positive' if x>3 else 'Negative')



# seeing the first 5 rows of amazon_df dataframe

amazon_df.head()
amazon_df.Score.unique()
print("size of our data is", amazon_df.shape)

print("")

amazon_df.info()
# lets sort the data based on the time

amazon_df = amazon_df.sort_values('Time')



#lets drop the duplicate datpoints

amazon_df.duplicated().sum()
print(amazon_df.duplicated(subset=['ProductId', 'Text']).sum())

print(amazon_df.duplicated(subset=['ProductId', 'Time', 'Text']).sum())

print(amazon_df.duplicated(subset=['UserId', 'Time', 'Text']).sum())
#so there are no exact duplicate rows

#lets remove the datapoints having the same productId and Time and UserId

amazon_df = amazon_df.drop_duplicates(subset=['ProductId','Time','UserId'])

amazon_df = amazon_df.drop_duplicates(subset=['ProductId','Time','Text'])

amazon_df = amazon_df.drop_duplicates(subset=['UserId','Time','Text'])
amazon_df.duplicated(subset=['UserId','Time','Text']).sum()
amazon_df.query('HelpfulnessNumerator>HelpfulnessDenominator')
# so lets remove the above two observations

num_great = amazon_df.query('HelpfulnessNumerator>HelpfulnessDenominator')

amazon_df = amazon_df.drop(num_great.index)
amazon_df.query('HelpfulnessNumerator>HelpfulnessDenominator')
print(amazon_df.loc[215861])

amazon_df.loc[215861]['Text']
# As we can say from above that 215861 indexed row has invalid text and summary, lets drop that row also

amzon_df = amazon_df.drop(215861)
print("so we are left with {} observations and {} features".format(amazon_df.shape[0], amazon_df.shape[1]))
for sent in amazon_df['Text'].values[50:52]:

    if len(re.findall('<.*?>', sent)):

        print(sent)
stopw = set(stopwords.words('english'))

snow = nltk.stem.SnowballStemmer('english')

# lets remove words like not, very from stop words as they are meaninging in the reviews 

reqd_words = set(['only','very',"doesn't",'few','not'])

stopw = stopw - reqd_words
def clean_html(review):

    '''This function cleans html tags if any

    , in the review'''

    

    cleaner = re.compile('<.*?>')

    clean_txt  = re.sub(cleaner, ' ', review)

    return clean_txt



def cleanpunc(sentence): 

    '''function to clean the word of any punctuation

    or special characters'''

    

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned

print(stopw)

print('****************************')

print(snow.stem('beautiful'))
def clean_text(list_of_texts):

    final_cleaned_reviews = []

    all_positive_words = []

    all_negative_words = []

    start_time = time.time()

    for i,review in enumerate(list_of_texts.values):

        review_filtered_words = []

        html_free = clean_html(review)

        for h_free_word in html_free.split():

            for clean_word in cleanpunc(h_free_word).split():

                if (((clean_word.isalpha()) & (len(clean_word)>2)) & \

                    ((clean_word.lower() not in stopw))):

                    final_word = snow.stem(clean_word.lower())

                    review_filtered_words.append(final_word)

                    if amazon_df['Score'].values[i] == 'Positive':

                        all_positive_words.append(final_word)

                    else:

                        all_negative_words.append(final_word)

        final_str = " ".join(review_filtered_words)

        final_cleaned_reviews.append(final_str)

    end_time = time.time()

    print('time took is ', (end_time-start_time))

    return final_cleaned_reviews
amazon_df[:3]
amazon_df = amazon_df[:100000]

final_cleaned_reivews = clean_text(amazon_df.Text)
%%time

# adding the cleaned reviews to dataframe

amazon_df['CleanedText']=final_cleaned_reivews
# now lets clean our Summary column and append it our dataframe

final_cleaned_summary = clean_text(amazon_df.Summary)
amazon_df['Cleaned_summary'] = final_cleaned_summary

len(final_cleaned_summary)
amazon_df.head()
print(amazon_df.Score.value_counts(normalize=True))

sns.countplot(amazon_df.Score);
train_df = amazon_df[:60000]

cv_df = amazon_df[60000:80000]

test_df = amazon_df[80000:100000]
train_df['Score'].value_counts(normalize=True)
cv_df['Score'].value_counts(normalize=True)
test_df['Score'].value_counts(normalize=True)
train_df.CleanedText.values[:3]
def word_freq_seq(train_reviews,validation_reviews, test_reviews):

    count_vect = CountVectorizer()

    count_vect.fit(train_reviews)

    count_vect_xtrain = count_vect.transform(train_reviews)

    word_frequencies = count_vect_xtrain.sum(axis=0)

    word_count_list = [(word, count) for word, count in zip(count_vect.get_feature_names(), np.array(word_frequencies)[0])]

    word_freq_df = pd.DataFrame(sorted(word_count_list, key=lambda x: x[1], reverse=True), columns = ['word', 'frequency'])

    word_freq_df['freq_index'] = np.array(word_freq_df.index)+1

    print(word_freq_df.head())

    ax = sns.barplot(data=word_freq_df[:20], y='word', x='frequency')

    ax.set_title("top 20 words")

    plt.tight_layout()

    plt.show()

    

    # creating the vocabulary dict which contains the top 5k words and there frequency indexing.

    train_vocab_dict = {}

    for row in word_freq_df[:5000].iterrows():

        train_vocab_dict[row[1]['word']] = [row[1]['frequency'], row[1]['freq_index']]

    

    

    train_reviews_list = []

    cv_reviews_list = []

    test_reviews_list = []

    

    

    

    def gen_seq_from_dict(reviews_list, vocab_index_dict):

        final_reviews_index_list = []

        for review in reviews_list:

            review_list = []

            for word in review.lower().split():

                try:

                    review_list.append(vocab_index_dict[word][1])

                except:

                    pass

            final_reviews_index_list.append(np.array(review_list))

        return final_reviews_index_list

    

        

    train_encoded_reviews = gen_seq_from_dict(train_reviews, train_vocab_dict)

    valid_encoded_reviews = gen_seq_from_dict(validation_reviews, train_vocab_dict)

    test_encoded_reviews = gen_seq_from_dict(test_reviews, train_vocab_dict)

    

    return train_encoded_reviews, valid_encoded_reviews, test_encoded_reviews

    

    

    
train_encoded_reviews, valid_encoded_reviews, test_encoded_reviews = word_freq_seq(train_df.CleanedText, cv_df.CleanedText, test_df.CleanedText)
train_encoded_reviews[:2]
valid_encoded_reviews[:2]
test_encoded_reviews[:2]
len(valid_encoded_reviews)
# truncate and/or pad input sequences

max_review_length = 400

X_train = sequence.pad_sequences(train_encoded_reviews, maxlen=max_review_length)

X_cv = sequence.pad_sequences(valid_encoded_reviews, maxlen=max_review_length)

X_test = sequence.pad_sequences(test_encoded_reviews, maxlen=max_review_length)



print(X_train.shape)

print(X_train[1])
def text_to_num(series):

    num_array = []

    for x in series:

        if x == 'Positive':

            num_array.append(1)

        else:

            num_array.append(0)

    return np.array(num_array)
y_train = text_to_num(train_df.Score)

y_cv = text_to_num(cv_df.Score)

y_test = text_to_num(test_df.Score)
# http://faroit.com/keras-docs/1.2.2/preprocessing/text/

from keras.preprocessing.text import Tokenizer
tok = Tokenizer(num_words=600)

tok.fit_on_texts(train_df.CleanedText)
tok_t2m_train_reviews = tok.texts_to_matrix(train_df.CleanedText, mode='count')
tok_t2m_valid_reviews = tok.texts_to_matrix(cv_df.CleanedText, mode='count')
tok_t2m_test_reviews = tok.texts_to_matrix(test_df.CleanedText, mode='count')
tok_t2m_train_reviews.shape
tok_t2m_train_reviews[100]
tok_seq = Tokenizer(num_words=5000)

tok_seq.fit_on_texts(train_df.CleanedText)
tok_t2s_train_reviews = tok_seq.texts_to_sequences(train_df.CleanedText)

tok_t2s_valid_reviews = tok_seq.texts_to_sequences(cv_df.CleanedText)

tok_t2s_test_reviews = tok_seq.texts_to_sequences(test_df.CleanedText)
train_encoded_reviews[0]
tok_t2s_train_reviews[0]
# As keras models same shape vectors to feed into models, lets do sequence padding so all the reviews has same number of features.

max_review_length = 400

tok_t2sp_train_reviews = sequence.pad_sequences(tok_t2s_train_reviews, maxlen=max_review_length)

tok_t2sp_valid_reviews = sequence.pad_sequences(tok_t2s_valid_reviews, maxlen=max_review_length)

tok_t2sp_test_reviews = sequence.pad_sequences(tok_t2s_test_reviews, maxlen=max_review_length)



print(tok_t2sp_train_reviews.shape)

print(tok_t2sp_train_reviews[0])
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# confirm Keras sees the GPU

from keras import backend

assert len(backend.tensorflow_backend._get_available_gpus()) > 0
# Lets first build one lstm layer model

def one_lstm_model(max_review_length):

    embedding_vecor_length = 32

    model = Sequential()

    model.add(Embedding(5000+1, embedding_vecor_length, input_length=max_review_length))

    model.add(LSTM(100))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print(model.summary())

    return model
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from keras.layers.convolutional import Conv1D, MaxPooling1D
# lets build another model with 2 lstm models and one conv layer

def two_lstm_conv_model(max_review_length):

    embedding_vecor_length = 32

    model2 = Sequential()

    model2.add(Embedding(5000+1, embedding_vecor_length, input_length=max_review_length))

    model2.add(Dropout(0.2))

    model2.add(Conv1D(32, 3, padding='same', activation='relu'))

    model2.add(MaxPooling1D())

    model2.add(LSTM(100, return_sequences=True))

    model2.add(Dropout(0.2))

    model2.add(LSTM(100))

    model2.add(Dense(1, activation='sigmoid'))

    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model2.summary())

    return model2
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
## Function to plot history graphs



def plot_graphs(history):

  '''Plots epochs vs train and validation accuracies 

  and also epochs vs train and test losses'''

  train_acc_list = history.history['accuracy']

  train_loss_list = history.history['loss']

  val_acc_list = history.history['val_accuracy']

  val_loss_list = history.history['val_loss']

  epochs_list = np.array(history.epoch)+1

  

  fig, ax = plt.subplots(1,2, sharex=True)

  #ax.set_xlabel("epochs")

  fig.suptitle("model graphs")

  plt.xlabel('epochs')

  ax[0].plot(epochs_list, train_acc_list, color='b', label='train_accuracies')

  ax[0].plot(epochs_list, val_acc_list, color='r', label='validation_accuracies')

  ax[0].legend()

  ax[0].grid()

  ax[1].plot(epochs_list, train_loss_list, color='b', label='train_losses')

  ax[1].plot(epochs_list, val_loss_list, color='r', label='validation_losses')

  ax[1].legend()

  ax[1].grid()

  #ax[1,0].plot(epochs_list, val_acc_list, color='b', label='validation_accuracies')

  #ax[1,0].legend()

  #ax[1,1].plot(epochs_list, val_loss_list, color='r', label='validation_losses')

  #ax[1,1].legend()

  for a in ax.flat:

    a.set(xlabel='epochs', ylabel='values')

  plt.tight_layout()

  fig.subplots_adjust(top=0.85) #to prevent overlapping of title

  plt.show()
one_lstm = one_lstm_model(400)
early_stopping_monitor = EarlyStopping(patience=3)

history_one_wfi = one_lstm.fit(X_train, y_train,

                    batch_size=512, epochs=20,

                    validation_data = (X_cv, y_cv),

                    callbacks = [early_stopping_monitor])
history_one_wfi.history['accuracy']
plot_graphs(history_one_wfi)
from prettytable import PrettyTable

table = PrettyTable()

table.field_names = ['encoding','lstm layers', 'conv layers', 'train accuracy', 'test accuracy']
# lets see how our model1 performs on test data

score1_train = one_lstm.evaluate(X_train, y_train, batch_size=512)

score1_test = one_lstm.evaluate(X_test, y_test, batch_size=512)

print(score1_train)

print(score1_test)
table.add_row(['custom word frequency indexing',1, 0, 0.957, 0.922])
two_lstm = two_lstm_conv_model(400)
early_stopping_monitor = EarlyStopping(patience=3)

history_two_wfi = two_lstm.fit(X_train, y_train,

                    batch_size=512, epochs=20,

                    validation_data = (X_cv, y_cv),

                    callbacks = [early_stopping_monitor])
plot_graphs(history_two_wfi)
score2_train = two_lstm.evaluate(X_train, y_train, batch_size=512)

score2_test = two_lstm.evaluate(X_test, y_test, batch_size=512)

print(score2_train)

print(score2_test)
table.add_row(['custom word frequency indexing',2, 1, 0.9658, 0.9272])
one_lstm_w2m = one_lstm_model(600)
early_stopping_monitor = EarlyStopping(patience=3)

history_one_t2m = one_lstm_w2m.fit(tok_t2m_train_reviews, y_train,

                    batch_size=512, epochs=7,

                    validation_data = (tok_t2m_valid_reviews, y_cv),

                    callbacks = [early_stopping_monitor])
plot_graphs(history_one_t2m)
score3_train = one_lstm_w2m.evaluate(tok_t2m_train_reviews, y_train, batch_size=512)

score3_test = one_lstm_w2m.evaluate(tok_t2m_test_reviews, y_test, batch_size=512)

print(score3_train)

print(score3_test)
table.add_row(['keras text2matrix',1, 0, 0.88, 0.866])
two_lstm_w2m = two_lstm_conv_model(600)
early_stopping_monitor = EarlyStopping(patience=3)

history_two_t2m = two_lstm_w2m.fit(tok_t2m_train_reviews, y_train,

                    batch_size=512, epochs=5,

                    validation_data = (tok_t2m_valid_reviews, y_cv),

                    callbacks = [early_stopping_monitor])
plot_graphs(history_two_t2m)
score4_train = two_lstm_w2m.evaluate(tok_t2m_train_reviews, y_train, batch_size=512)

score4_test = two_lstm_w2m.evaluate(tok_t2m_test_reviews, y_test, batch_size=512)

print(score4_train)

print(score4_test)
table.add_row(['keras text2matrix',2, 1, 0.885, 0.866])
one_lstm_t2sp = one_lstm_model(400)
early_stopping_monitor = EarlyStopping(patience=4)

history_one_t2sp = one_lstm_t2sp.fit(tok_t2sp_train_reviews, y_train,

                    batch_size=1024, epochs=20,

                    validation_data = (tok_t2sp_valid_reviews, y_cv),

                    callbacks = [early_stopping_monitor])
plot_graphs(history_one_t2sp)
score5_train = one_lstm_t2sp.evaluate(tok_t2sp_train_reviews, y_train, batch_size=1024)

score5_test = one_lstm_t2sp.evaluate(tok_t2sp_test_reviews, y_test, batch_size=1024)

print(score5_train)

print(score5_test)
print(table)
table.add_row(['keras text2sequence padded',1, 0, 0.933, 0.904])
two_lstm_t2sp = two_lstm_conv_model(400)
early_stopping_monitor = EarlyStopping(patience=5)

history_two_t2sp = two_lstm_t2sp.fit(tok_t2sp_train_reviews, y_train,

                    batch_size=1024, epochs=20,

                    validation_data = (tok_t2sp_valid_reviews, y_cv),

                    callbacks = [early_stopping_monitor])
plot_graphs(history_two_t2sp)
score6_train = two_lstm_t2sp.evaluate(tok_t2sp_train_reviews, y_train, batch_size=1024)

score6_test = two_lstm_t2sp.evaluate(tok_t2sp_test_reviews, y_test, batch_size=1024)

print(score6_train)

print(score6_test)
print(table)
table.add_row(['keras text2sequence padded',2, 1, 0.9756, 0.923])
print(table)
# keras text to sequence two lstm model confusion matrix

model2_test_predict = two_lstm_t2sp.predict_classes(X_test)

skplt.metrics.plot_confusion_matrix(y_test, model2_test_predict);
# custom word frequency two lstm model confusion matrix

model1_test_predict = two_lstm.predict_classes(X_test)

skplt.metrics.plot_confusion_matrix(y_test, model1_test_predict);
print(table)
from sklearn.utils import class_weight
cls_wt_dict = dict(enumerate(class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)))

cls_wt_dict
pd.Series.value_counts(y_train, normalize=True)
two_lstm_class_wts = two_lstm_conv_model(400)

early_stopping_monitor = EarlyStopping(patience=10)

history_nine_wfi = two_lstm_class_wts.fit(X_train, y_train,

                    batch_size=1024, epochs=20,

                    validation_data = (X_cv, y_cv),

                    class_weight = cls_wt_dict,

                    callbacks = [early_stopping_monitor])

plot_graphs(history_nine_wfi)
score9_train = two_lstm_class_wts.evaluate(X_train, y_train, batch_size=1024)

score9_test = two_lstm_class_wts.evaluate(X_test, y_test, batch_size=1024)

print(score9_train)

print(score9_test)
table.add_row(['custom word frequency indexing with class wts',2, 1, 0.978, 0.908])
skplt.metrics.plot_confusion_matrix(y_test, model1_test_predict);
# custom word frequency two lstm model confusion matrix

final_model_test_predict_cls_wts = two_lstm_class_wts.predict_classes(X_test)

skplt.metrics.plot_confusion_matrix(y_test, final_model_test_predict_cls_wts);
skplt.metrics.plot_confusion_matrix(y_test, final_model_test_predict_cls_wts, title='with class weights');

skplt.metrics.plot_confusion_matrix(y_test, model1_test_predict, title='without class weights');
print(table)