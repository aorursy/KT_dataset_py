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
from sklearn.preprocessing import StandardScaler

import keras

import seaborn as sns

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.models import Sequential

from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import class_weight

from sklearn.pipeline import Pipeline
df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')
df['Size'] = df['Size'].replace('?', 'Medium')

df.replace('?', np.nan, inplace = True)

df = pd.get_dummies(df, columns = ['Size'])

for i in df:

    df[i] = pd.to_numeric(df[i])

for i in df:

    df[i].fillna(df[i].mean(), inplace = True)

df = df.drop(columns = ['ID'])

X = df.drop(columns =  ['Class'])

ss = StandardScaler()

ss.fit_transform(X)

y = df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size = 0.2, random_state = 42)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

y_train2 = np_utils.to_categorical(y_train)

y_test2 = np_utils.to_categorical(y_test)
# Build the architecture

model = Sequential()

model.add(Dense(13,input_dim=13, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(50, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(50, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
# Compile 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train2, validation_split=0.2, epochs=250,batch_size=64, class_weight=class_weights)
Y_pred_class=model.predict_classes(X_test, batch_size=64)
Y_pred_class
from sklearn.metrics import accuracy_score
accuracy_score(Y_pred_class, y_test)
test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')

df_submit = pd.DataFrame()

df_submit['ID'] = test['ID']
test = pd.get_dummies(test, columns = ['Size'])

X = test.drop(columns = ['ID'])

X = ss.transform(X)

Y_pred_class=model.predict_classes(X, batch_size=32)
Y_pred_class
df_submit['Class'] = Y_pred_class
df_submit.to_csv('submit3.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_submit)