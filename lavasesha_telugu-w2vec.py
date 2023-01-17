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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
telugu_df=pd.read_csv("/kaggle/input/telugu-nlp/telugu_books/telugu_books.csv")

train_df=pd.read_csv("/kaggle/input/telugu-nlp/telugu_news/train_telugu_news.csv")

test_df=pd.read_csv("/kaggle/input/telugu-nlp/telugu_news/test_telugu_news.csv")

print(telugu_df.info())

print(test_df.info(10))

print(train_df.info(10))
train_df["text"]=train_df['heading']+train_df['body']

test_df['text']=test_df['heading']+test_df['body']



concat_df=pd.concat([train_df['text'],test_df['text']])

concat_df1=pd.DataFrame(concat_df,columns=['text'])

result_df=pd.concat([telugu_df['text'],concat_df1['text']])

result_df1=pd.DataFrame(result_df,columns=['text'])

result_df1.info()
import re

import re

def clean_telugu_text_vocab(str_element):

    a=str(str_element)

    

    a=a.replace("\r",'')

    a=a.replace("\n",'')

    #a=a.replace(u'xao','')

    a=a.replace("  ","")

    a=a.replace('"','')

    a=a.replace(u'xao',u'')

  

    a=a.split()

    

    a=' '.join(map(str, a))

    return a

result_df1['text2']=result_df1['text'].apply(clean_telugu_text_vocab)

def clean_telugu_text(str_element):

    a=str(str_element)

    

    a=a.replace("\r",'')

    a=a.replace("\n",'')

    #a=a.replace(u'xao','')

    a=a.replace("  ","")

    a=a.replace('"','')

    a=a.replace(u'xao',u'')

  

    a=a.split()

    

    a=' '.join(map(str, a))

    return a.split()

result_df1['text1']=result_df1['text'].apply(clean_telugu_text)
result_df1['text1'].head()
from gensim.test.utils import get_tmpfile

from gensim.models import Word2Vec

common_texts=list(result_df1['text1'])

path = get_tmpfile("word2vec.model")

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
print(model.most_similar("వుండలేరు"))

print()

print(model.most_similar("కళ్ళలో"))
print(model.most_similar("దూరం"))

print()

print(model.most_similar("భయం"))

print()

print(model.most_similar("భారీ"))
#model.save("/content/Gdrive/My Drive/Kaggle_prjs/Telugu_NLP/W2V_TE")

#model.wv.save_word2vec_format("/kaggle/working/Word2Vec_TE1.txt",binary=False)
#train_df=pd.read_csv("/content/Gdrive/My Drive/Kaggle_prjs/Telugu_NLP/train_telugu_news.csv")



train_df.head()
test_df.head()
train_df.info()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

train_df['target']=le.fit_transform(train_df['topic'])

print(train_df['target'].unique())

print(train_df['topic'].unique())
test_df['target']=le.fit_transform(test_df['topic'])

print(test_df['target'].unique())

print(test_df['topic'].unique())
print(train_df["topic"].value_counts())

sns.countplot(x='topic',data=train_df)

max_len = 500

train_df['text1']=train_df['body'].apply(clean_telugu_text)

train_df['heading1']=train_df['heading'].apply(clean_telugu_text)

train_df["concat_text"]=train_df['heading1']+train_df['text1']

train_df["concat_text"].head()

train_df['t2']=train_df['body']+train_df['heading']

train_df["text2"]=train_df["t2"].apply(clean_telugu_text_vocab)

train_df["text2"].head()
test_df['t2']=test_df['body']+test_df['heading']

test_df["text2"]=test_df["t2"].apply(clean_telugu_text_vocab)

test_df["text2"].head()
import os 

embeddings_index={}



f = open(os.path.join("/kaggle/working/Word2Vec_TE1.txt"))

for line in f:

    values=line.split()

    word=values[0]

    #print(word)

    coefs=np.asarray(values[1:])

    embeddings_index[word]=coefs

f.close()
len(embeddings_index)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(train_df['text2'])

tokenizer1=Tokenizer(oov_token="<OOV>")

tokenizer1.fit_on_texts(test_df['text2'])

word_index=tokenizer.word_index



word_index
word_index_test=tokenizer1.word_index



word_index_test
train_df["seq_train"]=tokenizer.texts_to_sequences(train_df['text2'])

padded_X=pad_sequences(list(train_df["seq_train"]),maxlen=256)

test_df["seq_train"]=tokenizer1.texts_to_sequences(test_df['text2'])

padded_Xtest=pad_sequences(list(test_df["seq_train"]),maxlen=256)
print(len(padded_X[0]))

print(len(padded_Xtest[0]))
print(len(word_index))

print(len(word_index_test))
from keras.models import Model, Input

import keras

from keras.applications.densenet import DenseNet121

from keras.layers import Input

from keras.models import Model,Sequential

from keras.layers import Dense,Conv1D,MaxPool1D,BatchNormalization,MaxPooling1D,SpatialDropout1D

from keras.optimizers import Adam,SGD

from keras.layers import LSTM, Embedding, Dense

from keras.models import Model, Input

from keras.layers.merge import add

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,Flatten
embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, i in word_index.items():

  

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

     

    

embedding_layer = Embedding(len(word_index) + 1,

                            100,

                            weights=[embedding_matrix],

                            input_length=256,

                            trainable=False)
max_len=256



sequence_input = Input(shape=(max_len,), )



emb_word = embedding_layer(sequence_input)

 

# Adding dropout layer

emb_word = Dropout(0.2)(emb_word)



x = emb_word

x = Bidirectional(LSTM(units=50, return_sequences=True,

                       recurrent_dropout=0.2, dropout=0.2))(x)

x_rnn = Bidirectional(LSTM(units=50, return_sequences=True,

                           recurrent_dropout=0.2, dropout=0.2))(x)

x = add([x, x_rnn])  # residual connection to the first biLSTM



x = (Flatten())(x)

out = (Dense(5, activation="softmax"))(x)

model_word2vec = Model(sequence_input, out)

model_word2vec.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['acc'])
model_word2vec.summary()
padded_X=np.array(padded_X).astype("float32")

print(padded_X.shape)

padded_Xtest=np.array(padded_Xtest).astype("float32")

print(padded_Xtest.shape)
ycat_tr=np.array(train_df["target"]).reshape(-1,1)

print(ycat_tr.shape)

ycat_te=np.array(test_df["target"]).reshape(-1,1)

print(ycat_te.shape)
history = model_word2vec.fit(padded_X,                    

                    ycat_tr,

                    batch_size=32, epochs=3, validation_split=0.15, verbose=1)
import matplotlib.pyplot as plt

hist = pd.DataFrame(history.history)

plt.style.use("ggplot")

plt.figure(figsize=(20,4))

plt.plot(hist["acc"])

plt.plot(hist["val_acc"])

plt.show()



hist = pd.DataFrame(history.history)
from sklearn.metrics import classification_report,confusion_matrix

y_preds = model_word2vec.predict(padded_Xtest)

y_preds_te=np.argmax(y_preds,axis=-1)

target_names=['business','editorial','entertainment','nation','sports']

print(classification_report(test_df['target'], y_preds_te, target_names=target_names))

print(confusion_matrix(test_df['target'], y_preds_te))