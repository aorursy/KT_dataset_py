# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import string

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))





stopword = stopwords.words("english")

lemma = WordNetLemmatizer()



# Any results you write to the current directory are saved as output.
!pip install hyperas
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print("Train dataset shape: ", train_df.shape)

print("Test dataset shape: ", test_df.shape)
train_df.isna().sum()
print(train_df.duplicated(subset=['question_text']).sum())

print(test_df.duplicated(subset=['question_text']).sum())
(train_df['target'].value_counts()/train_df.shape[0]).plot(kind='bar')
train_df['qlen'] = train_df['question_text'].apply(lambda x: len(x.split()))

test_df['qlen'] = test_df['question_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(15,8))

sns.boxplot(y='qlen',x='target', data=train_df)
#Remove the outlier and plot again to get better visualization

train_df = train_df[train_df['qlen']<100]



plt.figure(figsize=(15,8))

sns.boxplot(y='qlen',x='target', data=train_df)
from tqdm import tqdm



glove_embedding = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

file = open(glove_embedding)

embedding_index={}

for line in tqdm(file):

    split_values = line.split(" ")

    #print(line.split(" ")[0], line.split()[1:])

    embedding_index[split_values[0]] = np.asarray(split_values[1:], dtype='float32')

file.close()
def cleanup(text):

    text = "".join([char for char in text if (ord(char)>=48 and ord(char)<=57) or (ord(char)>=65 and ord(char)<=90) or (ord(char)>=97 and ord(char)<=122) or (ord(char)==32) ])

    #text = " ".join([lemma.lemmatize(word.lower()) for word in text.split() if word.lower() not in stopword if len(word)>2])

    text = " ".join(word if (word in embedding_index) and (word.lower() not in embedding_index) else word.lower() for word in text.split())

    return text



train = train_df.copy()

train['question_text'] = train['question_text'].apply(cleanup)
train['qlen_wostopwords'] = train['question_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(15,8))

sns.boxplot(y='qlen_wostopwords',x='target', data=train)
train.groupby(by=['target']).mean()
train_df.head()
train.head()
print('Max words in train dataset: ', train.qlen_wostopwords.max())

#maxlen = train.qlen_wostopwords.max()+2

maxlen=100
X_train, X_test, y_train, y_test = train_test_split(train['question_text'],train['target'], test_size=0.2, random_state=42)
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=50000)

tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train,maxlen=maxlen)

X_test = pad_sequences(X_test,maxlen=maxlen)

y_train = y_train.values

y_test = y_test.values
#np.stack(embedding_index.values()).shape

embedding_matrix = np.zeros((50000,300))

for word,i in tokenizer.word_index.items():

    if i>=50000: continue

    v = embedding_index.get(word)

    if v is not None: embedding_matrix[i]=v
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, CuDNNLSTM

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate

from keras.models import Model

#from keras import initializers, regularizers, constraints, optimizers, layers



inp = Input(shape=(maxlen,))

x = Embedding(50000, 300, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

max_pool = GlobalMaxPool1D()(x)

avg_pool = GlobalAveragePooling1D()(x)

conc = concatenate([max_pool,avg_pool])

x = Dense(16, activation="relu")(conc)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

print(model.summary())
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=1024)
pred = model.predict(X_test)



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score, recall_score



thresh=0.1

prev_score=0

for i in np.arange(0.1,0.41,0.01):

    score = f1_score(y_test, (pred>i))

    if score>prev_score: thresh=i

    prev_score=score

    print('threshold at ', i, score)

#recall_score(y_test, (pred>0.3))

print('Best threshold:  ', thresh, ' f1score: ', f1_score(y_test, (pred>thresh)))
test_df['question_text'] = test_df['question_text'].apply(cleanup)

test_X = tokenizer.texts_to_sequences(test_df['question_text'].values)

test_X = pad_sequences(test_X,maxlen=maxlen)
test_pred = model.predict(test_X)

test_pred = (test_pred>thresh).astype('int')
pd.DataFrame({'qid':test_df['qid'].values, 'prediction':np.squeeze(test_pred)}).to_csv('submission.csv',index=False)
