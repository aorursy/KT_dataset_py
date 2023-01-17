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
import numpy as np
import nltk
import tensorflow as tf
import keras
import seaborn as sns
from nltk.corpus import stopwords
print(stopwords.words('english'))
df = pd.read_csv("../input/language-identification-datasst/dataset.csv")
df.head()
df.info()
df["Text"] = df["Text"].str.lower()
df.head()
#NLP Preprocessing : Tokenization and Embeddings
max_words = 50000
max_len = 100

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(list(df['Text']))
train_df = tokenizer.texts_to_sequences(list(df['Text']))

train_df = tf.keras.preprocessing.sequence.pad_sequences(train_df,maxlen = max_len)
sequences
max_len

df.head()
len(tokenizer.word_index)
Y = df['language']
Y
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
le = preprocessing.LabelEncoder()
le.fit(Y)

list(le.classes_)
Y2 = le.fit_transform(Y)
total_languages = df['language'].nunique()
Y2 = keras.utils.to_categorical(Y2,num_classes=total_languages)

np.shape(Y2)
X_train,X_test,Y_train,Y_test = train_test_split(train_df,Y2)
embedding_dims = 500
vocab_size = len(tokenizer.word_index)+1

# Build the neural network model
model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim = vocab_size,output_dim = embedding_dims,input_length = max_len),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(total_languages,activation=tf.nn.softmax)
                            ])
model.summary()
model.compile(optimizer ='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])



model.fit(np.array(X_train),np.array(Y_train),epochs=3)
model.evaluate(np.array(X_test),np.array(Y_test))
print("English ",le.transform(['English']))
print("French ",le.transform(['French']))
print("Dutch ",le.transform(['Dutch']))
print("Swedish ",le.transform(['Swedish']))
#text = ["Once you know all the elements, it's not difficult to pull together a sentence."]
text = ["När du väl känner till alla element är det inte svårt att ta ihop en mening."] #swedish
#text = ["Als je eenmaal alle elementen kent, is het niet moeilijk om een zin samen te stellen."] # Dutch
#text =["Une fois que vous connaissez tous les éléments, il n'est pas difficile de rassembler une phrase."] #French
test_text = tokenizer.texts_to_sequences(text)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=max_len)
predictions = model.predict(test_text)
out = predictions.argmax()
print(le.inverse_transform([out]))
print(predictions)
#text = ["Once you know all the elements, it's not difficult to pull together a sentence."]
#text = ["När du väl känner till alla element är det inte svårt att ta ihop en mening."] #swedish
text = ["Als je eenmaal alle elementen kent, is het niet moeilijk om een zin samen te stellen."] # Dutch
#text =["Une fois que vous connaissez tous les éléments, il n'est pas difficile de rassembler une phrase."] #French
test_text = tokenizer.texts_to_sequences(text)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=max_len)
predictions = model.predict(test_text)
out = predictions.argmax()
print(le.inverse_transform([out]))
print(predictions)
#text = ["Once you know all the elements, it's not difficult to pull together a sentence."]
#text = ["När du väl känner till alla element är det inte svårt att ta ihop en mening."] #swedish
#text = ["Als je eenmaal alle elementen kent, is het niet moeilijk om een zin samen te stellen."] # Dutch
text =["Une fois que vous connaissez tous les éléments, il n'est pas difficile de rassembler une phrase."] #French
test_text = tokenizer.texts_to_sequences(text)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=max_len)
predictions = model.predict(test_text)
out = predictions.argmax()
print(le.inverse_transform([out]))
print(predictions)
text =["Una vez que conoces todos los elementos, no es difícil armar una oración."] #French
test_text = tokenizer.texts_to_sequences(text)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=max_len)
predictions = model.predict(test_text)
out = predictions.argmax()
print(le.inverse_transform([out]))
print(predictions)
