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
df = pd.read_csv("../input/us-consumer-finance-complaints/consumer_complaints.csv")
df.head(5)


complains = df[["product","consumer_complaint_narrative"]]
complains["consumer_complaint_narrative"].isnull().value_counts()
complains["product"].value_counts()
df_complains = complains.dropna().reset_index().drop("index",axis =1)
df_complains["product"].value_counts()
def print_plot(index):
    print(df_complains["consumer_complaint_narrative"].iloc[index])
    print("Product: ",df_complains["product"].iloc[index])
    print("\n")
print_plot(12)
print_plot(15)
text = df_complains["consumer_complaint_narrative"].iloc[15]
type(text)

text = text.lower()
pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")
text = re.sub(pattern,"",text)
# remove x from the text
text = text.replace("x","")
text = word_tokenize(text)
texts = [word for word in text if word.isalpha()]
stop_words = stopwords.words("english")
cleaned_text = [word for word in texts if word not in stop_words]
cleaned_text = " ".join(cleaned_text)
cleaned_text
a = "xxx thisx isx wroxng"
pattern = re.compile(r"x*")
re.sub(pattern,"",a)
#a.replace("x","")
complains["consumer_complaint_narrative"]
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
def clean_text(text):
    """
    Text: a string 
    
    Will apply this function to a series"""
    text = text.lower()
    pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")
    text = re.sub(pattern,"",text)
    # remove x from the text
    text = text.replace("x","")
    text = text.split()
    texts = [word for word in text if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    cleaned_text = [word for word in texts if word not in stop_words]
    cleaned_text = " ".join(cleaned_text)
    return cleaned_text

df_complains["consumer_complaint_narrative"] = df_complains["consumer_complaint_narrative"].apply(clean_text)
print_plot(15)
print_plot(19)
import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.preprocessing.text import Tokenizer

# maximum number of words in our vocabulary
max_nb_words = 50000
# maximum number of words within each complaint document 
max_sequence_length = 250

embedding_dim = 100

tokenizer= Tokenizer(num_words = max_nb_words,
                    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                    lower = True,
                    split = " ",
                    char_level = False)

tokenizer.fit_on_texts(df_complains["consumer_complaint_narrative"].to_numpy())
word_index = tokenizer.word_index
print("Found unique tokens", len(word_index))
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df_complains["consumer_complaint_narrative"].to_numpy())
X = pad_sequences(X,maxlen = max_sequence_length)
print("shape of data tensor:", X.shape)
print(X[1])
print(tokenizer.sequences_to_texts([X[1]]))

y = pd.get_dummies(df_complains["product"]).values
print("Shape of target tensor", y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                test_size = 0.1,
                                                random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(max_nb_words,embedding_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(11,activation = "softmax"))
model.compile(loss = "categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])

EPOCHS = 5
batch_size = 64

history = model.fit(X_train,y_train, epochs = EPOCHS,batch_size = batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print("Test set \n Loss: {} \n Accuracy: {}".format(accr[0],accr[1]))
history.history
import matplotlib.pyplot as plt
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();