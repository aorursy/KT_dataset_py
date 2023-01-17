# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from keras.utils.np_utils import to_categorical

from keras.callbacks import Callback

from keras.callbacks import EarlyStopping



import pandas as pd

import os

import numpy as np

import pdb



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
INBOUND_DATA_FOLDER = "../input"

DATASET_FILENAME = "adult-training.csv"
df_train = pd.read_csv(os.path.join(INBOUND_DATA_FOLDER, DATASET_FILENAME), header=None)

df_test = pd.read_csv(os.path.join(INBOUND_DATA_FOLDER, "adult-test.csv"), header=None, skiprows=[0])



cols = [

    'age',

    'workclass',

    'fnlwgt',

    'education',

    'education-num',

    'marital-status',

    'occupation',

    'relationship',

    'race',

    'sex',

    'capital-gain',

    'capital-loss',

    'hours-per-week',

    'native-country',

    'label']



df_train.columns = cols

df_test.columns = cols





replacer = {" <=50K": 0, " >50K" : 1, " <=50K.": 0, " >50K." : 1}

df_train.label = df_train.label.replace(replacer)

df_test.label = df_test.label.replace(replacer)



train_labels = df_train.label.fillna(df_train.label.mode()[0])

test_labels = df_test.label.fillna(train_labels.mode()[0])



df_train = df_train.drop("label", axis=1)

df_test = df_test.drop("label", axis=1)







train_rows = df_train.shape[0]



# merge dataframes

df = pd.concat([df_train, df_test], axis=0)



# drop redundant columns

df = df.drop("education", axis=1)

df = df.drop("fnlwgt", axis=1)



# replace ? with NaN

df = df.replace("?", value=np.nan)



# Impute columns by using fillna

mode_columns = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

median_columns = ["age", "education-num", "capital-gain", "capital-loss"]

mean_columns = ["hours-per-week"]



for mode_column in mode_columns:

    df[mode_column] = df[mode_column].fillna(df[mode_column].mode()[0])

for median_column in median_columns:

    df[median_column] = df[median_column].fillna(df[median_column].median())

for mean_column in mean_columns:

    df[mean_column] = df[mean_column].fillna(df[mean_column].mean())



# One hot encode

encoded = pd.get_dummies(df)



# unmerge train and test dataframes

df_train = encoded.iloc[:train_rows, :]

df_test = encoded.iloc[train_rows:, :]



# reappend label column to train

df_train = pd.concat([train_labels, df_train], axis=1)

df_test = pd.concat([test_labels, df_test], axis=1)





# Extract values

train = df_train.values

test = df_test.values
X_train = train[:, 1:]

y_train = train[:, 0]



X_test = test[:, 1:]

y_test = test[:, 0]



# scaling features

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Deep neural net

def create_model():

    model = Sequential()

    model.add(Dense(200,

                    input_dim=X_train.shape[1],

                    activation="relu"))

    model.add(Dense(100,

                    activation="relu"))

    model.add(Dense(2, activation="softmax"))



    # Compile model

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model

estimator = KerasClassifier(create_model, epochs=10, batch_size=40, verbose=False)



Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)



results = estimator.fit(X_train, Y_train, validation_data=(X_test, Y_test))

print("Score: {}".format(estimator.score(X_test, Y_test)))