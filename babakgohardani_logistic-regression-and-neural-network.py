



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report





import keras 

from keras.models import Sequential 

from keras.layers import Embedding, Dropout, Flatten, Dense



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Any results you write to the current directory are saved as output.
ad_data = pd.read_csv('../input/advertising.csv')
ad_data.columns
ad_data.head(10)
ad_data.shape
sns.pairplot(ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Ad Topic Line', 'Male','Clicked on Ad']])
plt.scatter(ad_data['Daily Time Spent on Site'], ad_data['Clicked on Ad'])
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male','Clicked on Ad']]

Y = ad_data['Clicked on Ad']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
lgmodel = LogisticRegression()

lgmodel.fit(X_train, Y_train)
preds = lgmodel.predict(X_test)
print(confusion_matrix(Y_test, preds))

print(classification_report(Y_test, preds))
classifier = Sequential()

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=6))

classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()
classifier.fit(X_train, Y_train, batch_size=10, nb_epoch=100)
# predict

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5) # true / false if statement



cm = confusion_matrix(Y_test, y_pred)

print(cm)
ad_data = pd.read_csv('../input/advertising.csv')

X_text = ad_data.iloc[:, 4]

Y = ad_data.iloc[:,9]

X_text.head(3)
print(len(X_text))
from __future__ import print_function, division

from builtins import range



print(max(len(s) for s in X_text))

print(min(len(s) for s in X_text))

s = sorted(len(s) for s in X_text)

print(s[len(s) // 2])
max_sequence_length = 40

embedding_dim = 50





tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_text)

sequences = tokenizer.texts_to_sequences(X_text)



word2index = tokenizer.word_index
sequences = pad_sequences(sequences, maxlen=max_sequence_length)

print(sequences.shape)



X1, X2, Y1, Y2 = train_test_split(sequences,  Y, test_size = 0.2, random_state= 0)
input_dim = len(word2index)+1

print(input_dim)



model = Sequential()

model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=max_sequence_length))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
batch_size = 10

epochs = 10



model.fit(X1, Y1, epochs=epochs, batch_size=batch_size, validation_split=0.1)

loss, accuracy = model.evaluate(X2, Y2, verbose=0)

print('Accuracy: %f' % (accuracy*100))