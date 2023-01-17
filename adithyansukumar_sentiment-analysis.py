import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df_train = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/train.txt", delimiter=';', header=None, names=['sentence','label'])
df_train.head()
sns.countplot(df_train['label'])
from nltk.corpus import stopwords 
import string
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    word_seq=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return word_seq
text=df_train['sentence'].apply(text_process)

text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_len=100
max_words=20000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)

data=pad_sequences(sequences,maxlen=max_len)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encode=LabelEncoder()
y=encode.fit_transform(df_train['label'])
y_data=np_utils.to_categorical(y)
y_data.shape
y_data
len(data)
len(y_data)
x_train=data[:13000]
y_train=y_data[:13000]
x_val=data[13000:]
y_val=y_data[13000:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dropout,Dense,Bidirectional,LSTM
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Embedding(max_words,64,input_length=max_len))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dense(y_data.shape[1],activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
es=EarlyStopping(monitor='val_loss')
predictions=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=30,callbacks=[es])
losses=pd.DataFrame(model.history.history)
losses.plot()
df_test= pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/test.txt", delimiter=';', header=None, names=['sentence','label'])
df_test
testing_text=df_test['sentence'].apply(text_process)
len(testing_text)
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(testing_text)
test_sequences=tokenizer.texts_to_sequences(testing_text)
test_data=pad_sequences(test_sequences,maxlen=max_len)
testing_labels=encode.fit_transform(df_test['label'])
testing_labels=np_utils.to_categorical(testing_labels)
testing_labels.shape


test_predictions=model.fit(test_data,testing_labels,epochs=30,callbacks=[es])
encode.classes_


