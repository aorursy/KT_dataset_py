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
import pandas as pd

ref_test_x = pd.read_csv("../input/hackathon/ref_test_x.csv")

ref_train_x = pd.read_csv("../input/hackathon/ref_train_x.csv")

ref_train_y = pd.read_csv("../input/hackathon/ref_train_y.csv")
import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.utils import to_categorical

import sklearn.metrics

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras import optimizers

from keras.utils import to_categorical

from keras import models, layers

import keras

import numpy as np







def read_train_data(filepath_x, filepath_y):

    df = pd.read_csv(filepath_x, sep = ',', decimal=".")

    y = pd.read_csv(filepath_y, header = 0)

    df['y'] = y

    return df



def split_data(df):

    df = df.fillna(df.mean())

    df = df.select_dtypes(exclude=['object'])

    df = df.drop(columns=['sector'])

    df = df.sample(frac=1)

    df_train = df[0:14000]

    df_test = df[14000:]



    return df_train, df_test



df = read_train_data("../input/hackathon/ref_train_x.csv", "../input/hackathon/ref_train_y.csv")

df_train, df_test = split_data(df)





def FCNN():



    model = Sequential()

    model.add(Dense(128, input_dim=20, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(2, input_dim=20, activation='softmax'))





    return model





model = FCNN()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# prerocess dataframe



col_train_bis = list(df_train.columns)

col_train_bis.remove('y')

feature_cols = df_train[col_train_bis]



labels = df_train['y'].values



history = model.fit(np.array(feature_cols), to_categorical(np.array(labels)), epochs=5, batch_size=1, verbose=1)



model.summary()

model.save('Hackathon.h5')

col_test_bis = list(df_test.columns)

col_test_bis.remove('y')

feature_cols = df_test[col_test_bis]

labels = df_test['y'].values



y_scores = model.predict(np.array(feature_cols), batch_size=1)



print(y_scores.shape)

print(labels.shape)



score = sklearn.metrics.roc_auc_score(labels, y_scores[:, 1])

print(score)