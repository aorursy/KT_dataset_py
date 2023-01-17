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
import seaborn as sns

import re



from statistics import mode
df = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")

total = pd.concat([df, test]).reset_index(drop=True)
df.head()
df.isnull().sum()
sns.countplot(x="Survived",data=df)
sns.countplot(x="Survived", hue="Sex", data=df)
sns.countplot(x="Survived", hue="Pclass", data=df)
sns.countplot(x="Survived", hue="Embarked", data=df)
sns.countplot(x="Survived", hue="SibSp", data=df)
sns.countplot(x="Survived", hue="Parch", data=df)
df.describe()
df.loc[:, "Age"].hist(bins=80)
sns.countplot(df.loc[:, "SibSp"])
sns.countplot(df.loc[:, "Parch"])
df.loc[:, "Ticket"].value_counts().head()
df.loc[:, "Fare"].hist(bins=40)
df.loc[:, "Cabin"].value_counts().head()
sns.countplot(df.loc[:, "Embarked"])
sns.heatmap(df.corr(), annot=True)
df.loc[df.loc[:, "Age"].isnull(), "Age"] = df.groupby("Pclass")["Age"].transform('median')



df.loc[:, "Age"].isnull().sum()
# Sex

df.loc[df.loc[:, "Sex"] == "male", "Sex"] = 0

df.loc[df.loc[:, "Sex"] == "female", "Sex"] = 1
df.loc[df.loc[:, "Embarked"] == "S",  "Embarked"] = 0

df.loc[df.loc[:, "Embarked"] == "C",  "Embarked"] = 1

df.loc[df.loc[:, "Embarked"] == "Q",  "Embarked"] = 2
total.loc[:, "Embarked"] = total.loc[:, "Embarked"].fillna(mode(total.loc[:, "Embarked"]))
total.loc[total.loc[:, "Sex"] == "male", "Sex"] = 0

total.loc[total.loc[:, "Sex"] == "female", "Sex"] = 1



total.loc[:, "Sex"] = total.loc[:, "Sex"].astype(int)
total.loc[total.loc[:, "Embarked"] == "S",  "Embarked"] = 0

total.loc[total.loc[:, "Embarked"] == "C",  "Embarked"] = 1

total.loc[total.loc[:, "Embarked"] == "Q",  "Embarked"] = 2



total.loc[:, "Embarked"] = total.loc[:, "Embarked"].astype(int)
sns.heatmap(total.corr(), annot=True)
total.loc[total.loc[:, "Age"].isnull(), "Age"] = total.groupby("Pclass")["Age"].transform("median")
total.loc[total.loc[:, "Fare"].isnull(), "Fare"] = total.groupby("Pclass")["Fare"].transform("median")
total.loc[:, "Cabin"] = total.loc[:, "Cabin"].fillna("U")
total.loc[:, "Cabin"] = total.loc[:, "Cabin"].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
sorted(total.loc[:, "Cabin"].unique().tolist())
cabin_dict = {

    "A": 0,

    "B": 1, 

    "C": 2,

    "D": 3,

    "E": 4,

    "F": 5,

    "G": 6,

    "T": 7,

    "U": 8

}



total.loc[:, "Cabin"] = total.loc[:, "Cabin"].map(cabin_dict)
total.loc[:, "Cabin"].value_counts()
total.loc[:, "Name"] = total.loc[:, "Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)



total.loc[:, "Name"].value_counts()
total.loc[:, "Name"] = total.loc[:, "Name"].replace(["Rev", "Dr", "Col", "Major", "Mlle", "Ms", "Mme", "Lady", "Jonkheer", "Dona", "Capt", "Countess",

                              "Don", "Sir"], "Others")



total.loc[:, "Name"].value_counts()
name_dict = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Others": 4}

total.loc[:, "Name"] = total.loc[:, "Name"].map(name_dict)



total.loc[:, "Name"].value_counts()
total.loc[:, "FamilySize"] = total.loc[:, "SibSp"] + total.loc[:, "Parch"] + 1
train = total.loc[total.loc[:, "Survived"].notnull(), :]

test = total.loc[total.loc[:, "Survived"].isnull(), :]
features = ["Pclass", "Name", "Sex", "Age", "Fare", "Cabin", "Embarked", "FamilySize"]

target = "Survived"
from keras import Model

from keras.layers import Input, Dense

from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model



from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
train_train, train_test = train_test_split(train, test_size=0.2, random_state=4)
inputs = Input(shape=(8))

X = Dense(units=2, activation="relu")(inputs)

X = Dense(units=2, activation="relu")(X)

outputs = Dense(units=1, activation="sigmoid")(X)



model = Model(inputs=inputs, outputs=outputs)



model.compile(optimizer=Adam(learning_rate=0.0001) , loss="binary_crossentropy", metrics=["accuracy"])



training = model.fit(

    x=train_train[features],

    y=train_train[target],

    epochs=100,

    verbose=1,

    batch_size=10,

    validation_data=(train_test[features], train_test[target])

)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(training.history['loss'], color='b', label="Training loss")

ax[0].plot(training.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(training.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(training.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
inputs = Input(shape=(8))

X = Dense(units=2, activation="relu")(inputs)

X = Dense(units=2, activation="relu")(X)

outputs = Dense(units=1, activation="sigmoid")(X)



model = Model(inputs=inputs, outputs=outputs)



model.compile(optimizer=Adam(learning_rate=0.0001) , loss="binary_crossentropy", metrics=["accuracy"])



training = model.fit(

    x=train[features],

    y=train[target],

    epochs=100,

    verbose=1,

    batch_size=10

)
test.loc[:, "Survived"] = model.predict(test[features])
test.loc[:, "Survived"] = test.loc[:, "Survived"].map(lambda x: 1 if x >= 0.5 else 0).astype(int)
submission = test.loc[:, ["PassengerId", "Survived"]]

submission.to_csv("/kaggle/working/submission.csv", index=False)