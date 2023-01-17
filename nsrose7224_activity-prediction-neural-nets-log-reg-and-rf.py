# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.columns
unique_activities = train_df.Activity.unique()

print("NUmber of unique activities: {}".format(len(unique_activities)))

replacer = {}

for i, activity in enumerate(unique_activities):

    replacer[activity] = i

train_df.Activity = train_df.Activity.replace(replacer)

test_df.Activity = test_df.Activity.replace(replacer)

train_df.head(10)
train_df = train_df.drop("subject", axis=1)

test_df = test_df.drop("subject", axis=1)
def get_all_data():

    train_values = train_df.values

    test_values = test_df.values

    np.random.shuffle(train_values)

    np.random.shuffle(test_values)

    X_train = train_values[:, :-1]

    X_test = test_values[:, :-1]

    y_train = train_values[:, -1]

    y_test = test_values[:, -1]

    return X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression



X_train, X_test, y_train, y_test = get_all_data()

model = LogisticRegression(C=10)

model.fit(X_train, y_train)

model.score(X_test, y_test)
# Try some transformations

from sklearn.decomposition import PCA



X_train, X_test, y_train, y_test = get_all_data()

pca = PCA(n_components=200)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)



model.fit(X_train, y_train)

model.score(X_test, y_test)

# Worse performance, but trains faster
# Scale features to be between -1 and 1

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



model.fit(X_train, y_train)

model.score(X_test, y_test)

# Better performance
# Neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils.np_utils import to_categorical



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



n_input = X_train.shape[1] # number of features

n_output = 6 # number of possible labels

n_samples = X_train.shape[0] # number of training samples

n_hidden_units = 40



Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)

print(Y_train.shape)

print(Y_test.shape)



def create_model():

    model = Sequential()

    model.add(Dense(n_hidden_units,

                    input_dim=n_input,

                    activation="relu"))

    model.add(Dense(n_hidden_units,

                    input_dim=n_input,

                    activation="relu"))

    model.add(Dense(n_output, activation="softmax"))



    # Compile model

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model





estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=10, verbose=False)

estimator.fit(X_train, Y_train)

print("Score: {}".format(estimator.score(X_test, Y_test)))
from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



model = RandomForestClassifier(n_estimators=500)

model.fit(X_train, y_train)

model.score(X_test, y_test)