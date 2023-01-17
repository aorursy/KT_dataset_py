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
df = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')
df.shape
df.head()
df['STATE'].unique()
Telangana = df[df['STATE']=='Telangana']

Telangana['CONSTITUENCY'].unique()
df = df.dropna()
df = df[df['CRIMINAL\nCASES'] != "Not Available"]

df['CRIMINAL\nCASES'] = df['CRIMINAL\nCASES'].astype(int)
import matplotlib.pyplot as plt

import seaborn as sb
plt.figure(figsize=(20,10))

sb.set(style="darkgrid")

ax = sb.countplot(x='STATE', data=df)

plt.title('Candidate by State')

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
Telangana
df['CONSTITUENCY'].unique()
plt.figure(figsize=(20,10))

sb.set(style="darkgrid")

ax = sb.countplot(x='PARTY', data=df)

plt.title('Candidate by Party')

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure(figsize=(20,10))

sb.set(style="darkgrid")

ax = sb.distplot(df['AGE'], kde = True)

plt.title('Candidate by Age')
df.dtypes
df.columns
def categorizing(dat):

    cat = dat.astype('category').cat.codes

    return cat
df['STATE'] = categorizing(df['STATE'])

df['CONSTITUENCY'] = categorizing(df['CONSTITUENCY'])

df['NAME'] = categorizing(df['NAME'])

df['PARTY'] = categorizing(df['PARTY'])

df['SYMBOL'] = categorizing(df['SYMBOL'])

df['GENDER'] = categorizing(df['GENDER'])

df['CATEGORY'] = categorizing(df['CATEGORY'])

df['EDUCATION'] = categorizing(df['EDUCATION'])

df['ASSETS'] = categorizing(df['ASSETS'])

df['LIABILITIES'] = categorizing(df['LIABILITIES'])
df.head()
y = df['WINNER'].values

x = df.drop(columns=['WINNER']).values
from sklearn import preprocessing



minmax_scaler = preprocessing.MinMaxScaler()

X = minmax_scaler.fit_transform(x)
y.shape
X.shape
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense
model1 = Sequential()

model1.add(Dense(32,activation = 'relu', input_shape= (18,)))

model1.add(Dense(32,activation = 'relu'))

model1.add(Dense(1, activation = 'sigmoid'))



model1.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics=['accuracy'])
hist1 = model1.fit(X_train, Y_train, 

                  batch_size=32, epochs=200, validation_data=(X_val, Y_val))
score = model1.evaluate(X_test,Y_test)

print("Loss: ", score[0])

print("Accuracy: ", score[1])
plt.plot(hist1.history['loss'])

plt.plot(hist1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
model2 = Sequential()

model2.add(Dense(1000, activation='relu', input_shape=(18,)))  

model2.add(Dense(1000, activation='relu'))

model2.add(Dense(1000, activation='relu'))

model2.add(Dense(1000, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))



model2.compile(optimizer='nadam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

hist2 = model2.fit(X_train, Y_train,

          batch_size=32, epochs=200,

          validation_data=(X_val, Y_val))
plt.plot(hist2.history['loss'])

plt.plot(hist2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
model2.evaluate(X_test,Y_test)
from keras.layers import Dropout

from keras import regularizers
model3 = Sequential()

model3.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(18,)))

model3.add(Dropout(0.3))

model3.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model3.add(Dropout(0.3))

model3.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model3.add(Dropout(0.3))

model3.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model3.add(Dropout(0.3))

model3.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
model3.compile(optimizer='nadam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

hist3 = model3.fit(X_train, Y_train,

          batch_size=32, epochs=200,

          validation_data=(X_val, Y_val))
plt.plot(hist3.history['loss'])

plt.plot(hist3.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.ylim(top=1.2, bottom=0)

plt.show()
model3.evaluate(X_test,Y_test)
y_pred_class = model3.predict(X_test)
Y_test = Y_test[:, np.newaxis]