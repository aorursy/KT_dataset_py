# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from matplotlib import pyplot as plt

from keras.metrics import categorical_accuracy

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bank.csv')
scaler = MinMaxScaler()
import seaborn as sns

sns.set(style="ticks")



sns.pairplot(df, palette="Set1")

plt.show()

sns.heatmap(df.corr(), annot=True)
def oneHotEncode(df,colNames):

    for col in colNames:

        if( df[col].dtype == np.dtype('object')):

            dummies = pd.get_dummies(df[col],prefix=col)

            df = pd.concat([df,dummies],axis=1)



            #drop the encoded column

            df.drop([col],axis = 1 , inplace=True)

    return df
df.head()
#select the training columns

training_cols = ['age','job','poutcome','marital', 'education', 'balance', 'housing','loan',\

                 'duration', 'pdays', 'previous','contact','month','day']

#our predicted variable

y_col = ['deposit']

X = df[training_cols]

y = df[y_col]

#create a training/testing set 80/20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#we are one hot encoding the target variable since it containes "yes" and "no"

y_train = np.asarray(oneHotEncode(y_train,['deposit']))

y_test = np.asarray(oneHotEncode(y_test,['deposit']))

#we are also going to send all of our categorical variables through one hot encoding

X_train = oneHotEncode(X_train, training_cols)

X_test = oneHotEncode(X_test, training_cols)

#we must scale our numerical data types

X_train[['balance', 'duration', 'pdays', 'previous']] = scaler.fit_transform(X_train[['balance', 'duration', 'pdays', 'previous']])

X_test[['balance', 'duration',  'pdays', 'previous']] = scaler.fit_transform(X_test[['balance', 'duration', 'pdays', 'previous']])

colnum = X_train.shape[1]
X_train.shape
model = Sequential()

model.add(Dense(150, activation='relu', input_shape=(colnum,)))

model.add(Dense(100, activation='relu', input_shape=(colnum,)))

model.add(Dense(50, activation='relu', input_shape=(colnum,)))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adamax',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=45)

score = model.evaluate(X_test, y_test, verbose=0) 

score[1]
history = model.fit(X_train, y_train, epochs=45)
plt.plot(np.arange(0,len(history.history['loss'])), history.history['loss'])

plt.title("Loss")

plt.show()

plt.plot(np.arange(0,len(history.history['acc'])), history.history['acc'])

plt.title("Accuracy")

plt.show()
pivot = df.pivot_table(index=['job'], columns='deposit',values='age', aggfunc='count')

pivot.plot(kind='bar')

plt.setp(fig.get_xticklabels(), visible=True) 

plt.xticks(np.arange(len(pivot.index)), pivot.index)

plt.show()

pivot = df.pivot_table(index=['contact'], columns='deposit',values='age', aggfunc='count')

pivot.plot(kind='bar')

plt.setp(fig.get_xticklabels(), visible=True) 

plt.xticks(np.arange(len(pivot.index)), pivot.index)

plt.show()

pivot = df.pivot_table(index=['month'], columns='deposit',values='age', aggfunc='count')

pivot.plot(kind='bar')

plt.setp(fig.get_xticklabels(), visible=True) 

plt.xticks(np.arange(len(pivot.index)), pivot.index)

plt.show()

pivot = df.pivot_table(index=['day'], columns='deposit',values='age', aggfunc='count')

pivot.plot(kind='bar')

plt.setp(fig.get_xticklabels(), visible=True) 

plt.xticks(np.arange(len(pivot.index)), pivot.index)

plt.show()