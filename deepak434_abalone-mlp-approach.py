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
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
seed = 7

np.random.seed(seed)



filename = '/kaggle/input/abalone-uci/abalone_original.csv'

dataset = pd.read_csv(filename)

print(dataset.shape)

dataset.head(10)

dataset.isnull().sum()
dataset['sex'].value_counts()

#dataset['length'].value_counts()

#dataset['diameter'].value_counts()

#dataset['height'].value_counts()

#dataset['whole-weight'].value_counts()

#dataset['shucked-weight'].value_counts()

#dataset['viscera-weight'].value_counts()

#dataset['shell-weight'].value_counts()
# Age is equal to no. of rings + 1.5

dataset['age'] = dataset['rings'] + 1.5

dataset = dataset.drop('rings', axis=1)
dataset['length'].hist()

dataset.head(10)
#dataset['sex'] = dataset['sex'].str.get_dummies(" ")

encoder = LabelEncoder()

encoder.fit(dataset['sex'])

dataset['sex'] = encoder.transform(dataset['sex'])

dataset.head(10)
print(dataset.shape)

data = dataset.values.astype('float')

print(data.shape)

x = data[:, 0:8]

y = data[:,8]
def model():

    model = Sequential()

    model.add(Dense(20, input_dim=8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(15, kernel_initializer='normal', activation='relu'))

    model.add(Dense(10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(estimator, x, y, cv=kfold)

print("Model: {} ({})".format(results.mean(), results.std()))
print(results)