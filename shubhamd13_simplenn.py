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
df = pd.read_excel('/kaggle/input/outpu.xls')

df=df.dropna()
df
X=df.comment.astype('str')
X
y=df.stars
y
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot('stars',data=df)
plt.show()
y.value_counts()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocab=20000
tokenizer=Tokenizer(vocab,oov_token="<oov>")
from keras.utils import to_categorical
cat_train=to_categorical(y_train)
cat_test=to_categorical(y_test)
cat_train.shape
tokenizer.fit_on_texts(X)
train_sequence=tokenizer.texts_to_sequences(X_train)
test_sequence=tokenizer.texts_to_sequences(X_test)

from scipy import stats 
stats.scoreatpercentile([len(i) for i in train_sequence],70)
train_sequence
padded_train=pad_sequences(train_sequence,maxlen=50)
padded_test=pad_sequences(test_sequence,maxlen=50)
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,GlobalAveragePooling1D,Dropout,Conv1D,GlobalMaxPooling1D
from keras.optimizers import Adam
model=Sequential()
model.add(Embedding(vocab,100))
# model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
# model.add(GlobalMaxPooling1D())
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(128,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer=Adam(lr=0.0005),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(padded_train,cat_train, batch_size=32,validation_split=0.25,epochs=3)

accr = model.evaluate(padded_test,cat_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import matplotlib.pyplot as plt
import seaborn as sns
plt.title('Loss')
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='test')
plt.legend()
plt.show();
plt.title('Accuracy')
plt.plot(model.history.history['accuracy'], label='train')
plt.plot(model.history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

new_complaint = ['good']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=50)
pred = model.predict(padded)
labels = ['1', '2', '3', '4', '5']
print(pred, labels[np.argmax(pred)])
new_complaint = ['not good']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=50)
pred = model.predict(padded)
labels = ['1', '2', '3', '4', '5']
print(pred, labels[np.argmax(pred)])
y_pred=model.predict_classes(padded_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
y_pred
type(y_test)
unique_elements, counts_elements = np.unique(y_pred, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
X
