# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from collections import Counter

import re

import string





import nltk

from nltk.corpus import stopwords

#nltk.download('stopwords')

from wordcloud import WordCloud



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Embedding, Input, LSTM,Bidirectional, GlobalMaxPool1D, Dropout

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint



from wordcloud import WordCloud, STOPWORDS



#settings

#start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape, test.shape
train.head()
# train.columns.values

list(train)
test.head()

# list(test)
'check for nulls'

train.isnull().any(),test.isnull().any()
x=train.iloc[:,2:].sum()

x_values = np.sort(x.values)

#plot  

plt.figure(figsize=(10,5))

ax= sns.barplot( x_values, x.index, orient='h')

plt.xlabel('# of Occurrences', fontsize=12)

plt.ylabel('Type ', fontsize=12)



# rects = ax.patches

# labels = x.values

   

plt.show()

np.sort(x.values)
columns = list(x.index)

train.groupby(columns).size().sort_values(ascending=False).reset_index().rename(columns={0: 'count'}).head(15)
print("Out of {} rows: \n {} are toxic \n {}  are severe_toxic \n {}  are obscene \n {}   are threat \n {}  are insult and \n {}  are identity_hate". \

      format(len(train),len(train[train.toxic==1]),len(train[train.severe_toxic==1]),len(train[train.obscene==1]), \

             len(train[train.threat==1]),len(train[train.insult==1]),len(train[train.identity_hate==1])))
print("toxic examples:")

train[train['toxic']==1]['comment_text'][:5]
print("severe_toxic examples:")

train[train['severe_toxic']==1]['comment_text'][:5]
print("obscene examples:")

train[train['obscene']==1]['comment_text'][:5]
print("threatthreat examples:")

train[train['threat']==1]['comment_text'][:5]
print("insult examples:")

train[train['insult']==1]['comment_text'][:5]
fig, ax = plt.subplots(figsize=(10, 6))

fig.suptitle('Correlation Matrix')

sns.heatmap(train[columns].corr(), annot=True, cmap="YlGnBu", linewidths=.5, ax=ax);
pd.set_option('display.max_colwidth', -1)



Fifth_line = train.comment_text.iloc[4]

Fifth_line,len(Fifth_line)
train_length = train.comment_text.apply(len)

train_length.head(6)
comments_max_ln  =np.max(train_length)

comments_min_ln  =np.min(train_length)

comments_mean_ln  =np.mean(train_length)

print (' comments_max_ln: {}, \n comments_min_ln: {} \n comments_mean_ln: {}'.format( comments_max_ln, 

                                                                                 comments_min_ln, comments_mean_ln ))
plt.figure(figsize = (12, 6))

plt.hist(train_length, bins = 60, alpha = 0.5, color = 'r')

plt.show()
print("max length : ", np.max(train_length))

print("min length : ", np.min(train_length))

print("mean length : ", np.mean(train_length))
'Check Missing Data'



def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        # written by MJ Bahmani

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)

    

check_missing_data(train)
print(train.comment_text.isna().sum())

print(test.comment_text.isna().sum())
test_comments = test.comment_text

test_comments[0:3]
# list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# y = train[list_classes].values

y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

y[:5]
comments_train = train.comment_text

comments_test = test.comment_text

# create the tokenizer

tok = Tokenizer()

# fit the tokenizer on the documents

tok.fit_on_texts(list(comments_train))  # train.comment_text



num_words_count = len(tok.word_index) + 1  # 210338 +1

tokenizer_all_comments = Tokenizer(num_words=num_words_count)

tokenizer_all_comments.fit_on_texts(list(comments_train))



list_tokenized_train = tokenizer_all_comments.texts_to_sequences(comments_train)

list_tokenized_test = tokenizer_all_comments.texts_to_sequences(comments_test)
list_tokenized_test[:1]
'to find max length of words in comments'

# distribution of number of words in sentence''

totalNumWords = [len(word_in_comment) for word_in_comment in list_tokenized_train]

totalNumWords.sort(reverse=True) # to find max word in list

totalNumWords[:5]
plt.hist(totalNumWords,bins = np.arange(0,400,10));
#  Most of the sentence length is about 30+. We could set the "maxlen" to about 50,

#  but I'm being paranoid so I have set to 200. 

max_len = 200

X_train = pad_sequences(list_tokenized_train, maxlen=max_len)

X_test = pad_sequences(list_tokenized_test, maxlen=max_len)
X_train[0]
# By indicating an empty space after comma, we are telling Keras to infer the number automatically.

inp = Input(shape=(max_len, )) #maxlen=200 as defined earlier
embed_size = 128

layer = Embedding(num_words_count, embed_size)(inp);   # num_words_count = 210338 +1
# layer = LSTM(60, return_sequences=True,name='lstm_layer')(layer)

# layer = GlobalMaxPool1D()(layer)

# layer = Dropout(0.1)(layer)

# layer = Dense(50, activation="relu")(layer)

# layer = Dropout(0.1)(layer)

# layer = Dense(6, activation="sigmoid")(layer)

# model = Model(inputs = inp, outputs = layer)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# batch_size = 32

# epochs = 2

# model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
def model():

    embed_size = 128

    inp = Input(shape=(max_len, ))

    layer = Embedding(num_words_count, embed_size)(inp)

    layer = LSTM(60, return_sequences=True,name='lstm_layer')(layer)

#     layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)

    layer = GlobalMaxPool1D()(layer)

    layer = Dropout(0.1)(layer)

    layer = Dense(50, activation="relu")(layer)

    layer = Dropout(0.1)(layer)

    layer = Dense(6, activation="sigmoid")(layer)

    model = Model(inputs = inp, outputs = layer)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = model()

model.summary()
file_path = 'save_analysis'

checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 1, save_best_only=True)



early_stop = EarlyStopping(monitor = 'val_loss', patience = 1)
hist = model.fit(X_train, y, batch_size = 32, epochs = 2, verbose=1, 

                 validation_split = 0.2, callbacks = [checkpoint, early_stop]);
vloss = hist.history['val_loss']

loss = hist.history['loss']



#x_len = np.arange(len(loss))

#plt.plot(x_len, vloss, marker='.', lw=2.0, c='red', label='val')

#plt.plot(x_len, loss, marker='.', lw=2.0, c='blue', label='train')

plt.figure()

plt.plot(loss, marker='.', lw=2.0, c='blue', label='train')

plt.plot(vloss, marker='.', lw=2.0, c='red', label='val')



plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Cross-Entropy Loss')

plt.grid()

plt.show()
plt.figure()

plt.plot(hist.history['val_acc'], marker='.', c='blue', label='train')

plt.plot(hist.history['acc'], marker='.', c='red', label='val')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')

plt.grid()

plt.show()
y_pred = model.predict(X_test)
submission = pd.read_csv('../input/sample_submission.csv')

list_classes=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] 

submission[list_classes] = y_pred



submission.to_csv("submission.csv", index=False)
submission.head()