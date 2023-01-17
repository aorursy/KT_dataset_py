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
import json



df=pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json",lines=True)

df.head()
df.shape
df.drop("article_link", axis=1, inplace=True)

df.head()
import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

stoplist = stopwords.words('english')



training_data=df.headline.to_list()
train_data=[]

for i in training_data:

    string=""

    for j in i.lower().split():

        if j not in stoplist:

            string=string+j+" "

    train_data.append(string.rstrip())
train_data[:3]
train_labels=np.array(df.is_sarcastic.to_list())

train_labels[:5]
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

oov_tok="<OOV_tok>"

padding="post"

max_length=25

trunc_type="post"

vocab_size=20000

tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_data)

word_index=tokenizer.word_index



train_sequence=tokenizer.texts_to_sequences(train_data)

train_pad_sequence=pad_sequences(train_sequence, padding=padding, maxlen=max_length, truncating=trunc_type)
max_length=[len(i.split()) for i in training_data]

max(max_length)
reverse_word_index=dict([(j, i) for i, j in word_index.items()])

reverse_word_index
len(word_index)
def decode_review(review):

    return " ".join([reverse_word_index.get(i, "?") for i in review])



print(decode_review(train_pad_sequence[1]))

print(train_data[:1])
embid_dim=16

vocab_size=20000

max_length=25

model=tf.keras.models.Sequential([

                                tf.keras.layers.Embedding(vocab_size, embid_dim, input_length=max_length),

                                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

                                tf.keras.layers.Dense(6, activation="relu"),

                                tf.keras.layers.Dense(1, activation="sigmoid")

])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

model.summary()
epochs=10

history=model.fit(train_pad_sequence, train_labels, epochs=epochs)
sentence="13,000 people receive #wildfires evacuation orders in California"

sentence_sequence = tokenizer.texts_to_sequences(sentence)

sentence_padding = pad_sequences(sentence_sequence, maxlen=max_length)

prediction=model.predict_classes(sentence_padding)

predict=np.max(prediction)

print(len(prediction))

print(predict)

if predict==1:

    print("real")

elif predict==0:

    print("fake")
df2=pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)

df2.head()
test_data=df2.headline.to_list()

test_data[:3]
test_sequence=tokenizer.texts_to_sequences(test_data)

test_pad_sequence=pad_sequences(test_sequence, maxlen=max_length, padding=padding, truncating=trunc_type)
predictions=model.predict_classes(test_pad_sequence)
from sklearn.metrics import accuracy_score

print(accuracy_score(df2.is_sarcastic, predictions))
