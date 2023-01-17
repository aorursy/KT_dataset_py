# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Read dataset

X_train = pd.get_dummies(train.drop('type', axis=1)).values

y_train = train['type'].values.reshape(-1, 1).ravel()

X_test = pd.get_dummies(test).values
# See relation of each features briefly.

sns.pairplot(train.drop('id', axis=1), hue="type", size=1.7)
print("dimension of X: {}".format(X_train.shape[1]))

# We have 3 types of monster

print(np.unique(y_train))
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



# String type is converted integer categorical label.

label_encoder = LabelEncoder()

labeled_y_train = label_encoder.fit_transform(y_train)

y_train = labeled_y_train.reshape(-1, 1)



# Convert type into onehot encoded data.

onehot_encoder = OneHotEncoder()

y_train = onehot_encoder.fit_transform(y_train).toarray()



# Or you can use Keras util for converting into onehot encoding label.

# from keras.utils import np_utils

# y_train = np_utils.to_categorical(labeled_y_train)
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.wrappers.scikit_learn import KerasClassifier



# Create sequential multilayer perceptron model

# input: 11 -> hidden1: 4 -> hidden2: 10 -> output: 3

def create_model():

    model = Sequential()

    model.add(Dense(4, input_dim=11, init='normal', activation='relu'))

    model.add(Dense(10, init='normal', activation='relu'))

    model.add(Dense(3, init='normal', activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# For compatibility with sklearn interface

estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=7, verbose=False)



pipeline = Pipeline([('scaler', StandardScaler()),

                     ('clf', estimator)])
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kfold = KFold(n_splits=5, shuffle=True)

scores = cross_val_score(pipeline, X_train, y_train, cv=kfold)



print("Cross validation score: {} +/- {}".format(np.mean(scores), np.std(scores)))