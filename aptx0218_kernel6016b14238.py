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

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import normalize

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense

import keras

from keras.optimizers import *

from keras.initializers import *

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from itertools import chain
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head(10)
df.query('Fare == 0').head()
np.mean(df.Fare), np.std(df.Fare)
df.head(10)
binar = LabelBinarizer().fit(df.loc[:,'Sex'])

df['Sex'] = binar.transform(df['Sex'])

df.head(10)
df["A_Class"] = 0

df["B_Class"] = 0

df["C_Class"] = 0



df.loc[df.Pclass == 1, "A_Class"] = 1

df.loc[df.Pclass == 2, "B_Class"] = 1

df.loc[df.Pclass == 3, "C_Class"] = 1

df_Embarked = pd.get_dummies(df.Embarked)

df_Embarked.head()
plt.hist(df.query('Age>0').Age,bins=40)

plt.show()
df.Age.isna().sum()
df.Age[df.Age.notna()][df.Age[df.Age.notna()] % 1 != 0].head()
df.loc[(df.Sex == 1) & (df.Age.isna()) & (df.Pclass == 1), "Age"] = 41.2

df.loc[(df.Sex == 1) & (df.Age.isna()) & (df.Pclass == 2), "Age"] = 30.7

df.loc[(df.Sex == 1) & (df.Age.isna()) & (df.Pclass == 3), "Age"] = 26.5



df.loc[(df.Sex == 0) & (df.Age.isna()) & (df.Pclass == 1), "Age"] = 34.6

df.loc[(df.Sex == 0) & (df.Age.isna()) & (df.Pclass == 2), "Age"] = 28.7

df.loc[(df.Sex == 0) & (df.Age.isna()) & (df.Pclass == 3), "Age"] = 21.7
for i in range(len(df)):

    if df.loc[i, "SibSp"] + df.loc[i, "Parch"] == 0:

        df.loc[i, "Alone"] = 0

    else:

        df.loc[i, "Alone"] = 1



df.Alone = df.Alone.astype(int)

df.head()
df_new = pd.concat([df, df_Embarked], axis=1)

df_new.head()
#feature_name = ["Sex", "Age", "Fare", "Alone", "A_Class", "B_Class", "C_Class", "C", "Q", "S"]

feature_name = ["Sex", "Age", "Fare", "Alone", "A_Class", "B_Class", "C_Class", "C", "Q", "S", "SibSp", "Parch"]



dfX = df_new[feature_name]

dfY = df_new["Survived"]
scaler = MinMaxScaler()

scaler.fit(dfX)
dfX = scaler.transform(dfX)

dfX = pd.DataFrame(dfX,columns=feature_name)
df.Survived.sum()/len(df)
model = Sequential()

model.add(Dense(32, input_dim=12, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(0.1))



model.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(0.1))



model.add(Dense(512, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(1024, activation="elu", kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(0.2))



model.add(Dense(512, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(256, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(128, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(64, activation="elu", kernel_initializer="he_normal"))

model.add(Dense(32, activation="elu", kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(0.2))



model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=["binary_accuracy"])
model.summary()
model_result = model.fit(dfX, dfY, batch_size=100, epochs=200, validation_split= 0.2, shuffle = True)
plt.figure(figsize=(20, 10))



plt.subplot(1, 2, 1)

plt.plot(model_result.history["loss"], label="training")

plt.plot(model_result.history["val_loss"], label="validation")

plt.axhline(0.55, c="red", linestyle="--")

plt.axhline(0.35, c="yellow", linestyle="--")

plt.axhline(0.15, c="green", linestyle="--")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(model_result.history["binary_accuracy"], label="training")

plt.plot(model_result.history["val_binary_accuracy"], label="validation")

plt.axhline(0.75, c="red", linestyle="--")

plt.axhline(0.80, c="green", linestyle="--")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()



plt.show()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()

test.loc[test.Fare == 0, "Fare"] = np.mean(df.Fare)

test["Sex"] = binar.transform(test["Sex"])

test["A_Class"] = 0

test["B_Class"] = 0

test["C_Class"] = 0



test.loc[test.Pclass == 1, "A_Class"] = 1

test.loc[test.Pclass == 2, "B_Class"] = 1

test.loc[test.Pclass == 3, "C_Class"] = 1



test_Embarked = pd.get_dummies(test.Embarked)

test.groupby(["Sex", "Pclass"]).Age.mean()
test.loc[(test.Sex == 1) & (test.Age.isna()) & (test.Pclass == 1), "Age"] = 41.2

test.loc[(test.Sex == 1) & (test.Age.isna()) & (test.Pclass == 2), "Age"] = 30.7

test.loc[(test.Sex == 1) & (test.Age.isna()) & (test.Pclass == 3), "Age"] = 26.5



test.loc[(test.Sex == 0) & (test.Age.isna()) & (test.Pclass == 1), "Age"] = 34.6

test.loc[(test.Sex == 0) & (test.Age.isna()) & (test.Pclass == 2), "Age"] = 28.7

test.loc[(test.Sex == 0) & (test.Age.isna()) & (test.Pclass == 3), "Age"] = 21.7
for i in range(len(test)):

    if test.loc[i, "SibSp"] + test.loc[i, "Parch"] == 0:

        test.loc[i, "Alone"] = 0

    else:

        test.loc[i, "Alone"] = 1



test.Alone = test.Alone.astype(int)
test_new = pd.concat([test, test_Embarked], axis=1)

test_new.head()
testX = test_new[feature_name]

testX = scaler.transform(testX)

testX = pd.DataFrame(testX, columns=feature_name)

predict = model.predict_classes(testX)
predict = list(chain.from_iterable(predict))

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})

my_submission.to_csv('submission.csv', index=False)