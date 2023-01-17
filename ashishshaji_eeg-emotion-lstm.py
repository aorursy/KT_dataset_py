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
df=pd.read_csv('/kaggle/input/eeg-brainwave-dataset-feeling-emotions/emotions.csv')

df.head()
df.info()
import seaborn as sns

sns.countplot(x='label', data=df)

df.isnull().sum()

#no missing values
encode = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )

#new dataset with replaced values

df_encoded = df.replace(encode)



print(df_encoded.head())

print(df_encoded['label'].value_counts())
df_encoded['label'].unique()
df_encoded.head()
x=df_encoded.drop(["label"]  ,axis=1)

x.shape
y = df_encoded.loc[:,'label'].values

y.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

from keras.utils import to_categorical

y = to_categorical(y)

y
y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
x_train = np.reshape(x_train, (x_train.shape[0],1,x.shape[1]))

x_test = np.reshape(x_test, (x_test.shape[0],1,x.shape[1]))

import tensorflow as tf

from tensorflow.keras import Sequential



from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import LSTM

tf.keras.backend.clear_session()



model = Sequential()

model.add(LSTM(64, input_shape=(1,2548),activation="relu",return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(32,activation="sigmoid"))

model.add(Dropout(0.2))

#model.add(LSTM(100,return_sequences=True))

#model.add(Dropout(0.2))

#model.add(LSTM(50))

#model.add(Dropout(0.2))

model.add(Dense(3, activation='sigmoid'))

from keras.optimizers import SGD

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs = 100, validation_data= (x_test, y_test))

score, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score

pred = model.predict(x_test)

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Training Accuracy: {correct}")