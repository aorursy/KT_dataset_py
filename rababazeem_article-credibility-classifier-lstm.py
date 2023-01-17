#importing libraries



import numpy as np

import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# reading the data needed to train the model

train=pd.read_csv('/kaggle/input/fake-news/train.csv')

test=pd.read_csv('/kaggle/input/fake-news/test.csv')
# deleting all the null values in the data

df=train.dropna()
# separating the text and their labels

X=df['text']

y=df['label']
#importing tensorflow

import tensorflow as tf

tf.__version__



from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.preprocessing.sequence import pad_sequences
voc_size=10000
# making all the text lowercase

X=[i.lower() for i in X]
# onehot encoding

onehot=[one_hot(words,voc_size) for words in X]
sen_len=300

embedded_doc=pad_sequences(onehot, padding='pre', maxlen=sen_len)

print(embedded_doc)
embedding_vector_feature=40

# bulding the model

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_feature, input_length=sen_len))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
# converting text and labels to numbers in arrays so the LSTM can be used on it

X_final=np.array(embedded_doc)

y_final=np.array(y)



X_final.shape
# splitting th testing and training data for validation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.33, random_state=0)
# fitting and validating the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
model.save("lstm_model.h5")
import nltk

nltk.download('punkt')



# prediciting if the current article is reliable or not

def reliable(file):

    with open(file, 'r') as file:

        text = file.read().replace('\n', ' ')

    

    text = nltk.tokenize.sent_tokenize(text)

    

    text=[token.lower() for token in text]

    

    onehot_enc=[one_hot(words,10000) for words in text]

    sen_len=300

    embedded_doc=pad_sequences(onehot_enc, padding='pre', maxlen=sen_len)

    text_final=np.array(embedded_doc)

    

    pred = model.predict_classes(text_final)

    

    pred_df = pd.DataFrame(pred)

    text_df = pd.DataFrame(text)

    result_df = pd.concat([pred_df, text_df])

    

    fake = result_df.loc[result_df["predictions"] == 0]

    

    return fake

    

reliable("/kaggle/input/cnn-article/message-2.txt")