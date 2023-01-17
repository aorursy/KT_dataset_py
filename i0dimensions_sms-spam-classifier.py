# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from wordcloud import WordCloud



import matplotlib.pyplot as plt

from math import log, sqrt



%matplotlib inline
import nltk

from nltk.corpus import stopwords

import string



#Need to download stopwords

nltk.download('stopwords')
with open('../input/data.txt', 'r') as f:

    msgs = f.read().splitlines()
print(msgs[10])

print(msgs[11])
msgs = pd.read_csv('../input/data.txt', sep='\t', header=None, names=['class', 'msg'])

msgs.head()
msgs.shape
# Map Class to 0/1

new_values = {'spam': 1, 'ham': 0}

msgs['class'] = msgs['class'].map(new_values)         



msgs.head()
# Display duplicate data-items

msgs[msgs.duplicated(keep=False)].sort_values("msg") 
#Checking for duplicates and removing them

msgs.drop_duplicates(inplace = True)

msgs.shape
# Spam WordCloud

spam_words = ' '.join(list(msgs[msgs['class'] == 1]['msg']))

spam_wc = WordCloud(width=512, height=512).generate(spam_words)



# Ham WordCloud

ham_words = ' '.join(list(msgs[msgs['class'] == 0]['msg']))

ham_wc = WordCloud(width=512, height=512).generate(ham_words)



# Sub-plots

fig = plt.figure(figsize = (10,8), facecolor = None)

fig.tight_layout(pad=3.0)



ax1 = fig.add_subplot(2,2,1)

ax1.imshow(spam_wc)

ax1.axis('off')

ax1.set_title('spam')



ax2 = fig.add_subplot(2,2,2)

ax2.imshow(ham_wc)

ax2.axis('off')

ax2.set_title('non-spam')
def process_text(text):

    

    #Remove punctuations

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    

    #Stop words clean-up

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    

    return clean_words
#Tokenization

msgs['msg'].head().apply(process_text)
from sklearn.feature_extraction.text import CountVectorizer



messages_bow = CountVectorizer(analyzer=process_text).fit_transform(msgs['msg'])
#Split data into 80% training & 20% testing data sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(messages_bow, msgs['class'], test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB()

classifier.fit(X_train, y_train)
#Evaluate the model on the training data set

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score, precision_score
list_alpha = np.arange(1/100000, 20, 0.11)



score_train = np.zeros(len(list_alpha))

score_test = np.zeros(len(list_alpha))

recall_test = np.zeros(len(list_alpha))

precision_test= np.zeros(len(list_alpha))



count = 0



for alpha in list_alpha:

    bayes = MultinomialNB(alpha=alpha)

    bayes.fit(X_train, y_train)

    score_train[count] = bayes.score(X_train, y_train)

    score_test[count]= bayes.score(X_test, y_test)

    recall_test[count] = recall_score(y_test, bayes.predict(X_test))

    precision_test[count] = precision_score(y_test, bayes.predict(X_test))

    count = count + 1 
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])

models = pd.DataFrame(data = matrix, columns = 

             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
plt.plot(list_alpha, precision_test, 'r', label='Test Precision')

plt.plot(list_alpha, score_test, 'b', label='Test Accuracy')

plt.title('Test Precision & Test Accuracy')

plt.xlabel('Smoothing Rate (alpha)')

plt.ylabel('Value')

plt.legend()

plt.show()
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()

bayes = MultinomialNB(alpha=list_alpha[best_index])

bayes.fit(X_train, y_train)

models.iloc[best_index, :]
m_confusion_test = confusion_matrix(y_test, bayes.predict(X_test))

pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])
from sklearn import svm
list_C = np.arange(1, 2000, 100) #100000



score_train = np.zeros(len(list_C))

score_test = np.zeros(len(list_C))

recall_test = np.zeros(len(list_C))

precision_test= np.zeros(len(list_C))



count = 0

for C in list_C:

    svc = svm.SVC(C=C)

    svc.fit(X_train, y_train)

    score_train[count] = svc.score(X_train, y_train)

    score_test[count]= svc.score(X_test, y_test)

    recall_test[count] = recall_score(y_test, svc.predict(X_test))

    precision_test[count] = precision_score(y_test, svc.predict(X_test))

    count = count + 1 
matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])

models = pd.DataFrame(data = matrix, columns = ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
plt.plot(list_C, precision_test, 'r', label='Test Precision')

plt.plot(list_C, score_test, 'b', label='Test Accuracy')

plt.title('Test Precision & Test Accuracy')

plt.xlabel('C-Regularization Parameter')

plt.ylabel('Value')

plt.legend()

plt.show()
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()

svc = svm.SVC(C=list_C[best_index])

svc.fit(X_train, y_train)

models.iloc[best_index, :]
m_confusion_test = confusion_matrix(y_test, svc.predict(X_test))

pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])
from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
X = msgs['msg']

Y = msgs['class']

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
max_words = 1000

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))