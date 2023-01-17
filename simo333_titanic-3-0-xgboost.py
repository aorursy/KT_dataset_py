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
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from tensorflow.keras import layers, models, optimizers



from xgboost import XGBClassifier, plot_importance
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df.head()
train_df.drop(["Name", "Ticket", "Cabin", "Age", "Fare", "SibSp","Parch"], axis=1, inplace=True)

train_label = train_df.pop("Survived")

train_df.Sex = pd.Categorical(train_df.Sex)

train_df.Sex = train_df.Sex.cat.codes

train_df.Embarked = pd.Categorical(train_df.Embarked)

train_df.Embarked = train_df.Embarked.cat.codes

train_df = train_df.astype(float)



test_df.drop(["Name", "Ticket", "Cabin", "Age", "Fare", "SibSp", "Parch"], axis=1, inplace=True)

test_df.Sex = pd.Categorical(test_df.Sex)

test_df.Sex = test_df.Sex.cat.codes

test_df.Embarked = pd.Categorical(test_df.Embarked)

test_df.Embarked = test_df.Embarked.cat.codes



test_df = test_df.astype(float)



train_df.set_index(["PassengerId"], inplace=True)

test_df.set_index(["PassengerId"], inplace=True)



train_df.head(), test_df.head()
train_X, test_X = train_test_split(train_df, test_size=0.2, random_state=7)

train_y, test_y = train_test_split(train_label, test_size=0.2, random_state=7)

train_X.shape
SC = StandardScaler()

train_X = SC.fit_transform(train_X)

test_X = SC.transform(test_X)

test_df = SC.transform(test_df)

train_X.shape
model = tf.keras.Sequential([

    tf.keras.layers.Dense(16, activation="relu",

                          kernel_regularizer=tf.keras.regularizers.l2(0.001),

                          input_shape=(train_X.shape[1],)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(32, activation="relu",

                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, activation="relu",

                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)

    ])



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0),

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=["accuracy"])



history = model.fit(train_X, train_y, epochs=20, validation_split=0.2)



model.evaluate(test_X, test_y)
hist = pd.DataFrame(history.history)

hist["epoch"] = history.epoch





def plot_history(history):

    plt.xlabel("epochs")

    plt.ylabel("Train, Val Accuracy")

    plt.plot(hist["epoch"], hist["accuracy"], label="Train acc")

    plt.plot(hist["epoch"], hist["val_accuracy"], label="val acc")

    plt.ylim([0, 1])

    plt.legend()

    plt.show()

    

    plt.xlabel("epochs")

    plt.ylabel("Train, Val Loss")

    plt.plot(hist["epoch"], hist["loss"], label="loss")

    plt.plot(hist["epoch"], hist["val_loss"], label="val loss")

    plt.legend()

    plt.show()

    



plot_history(history)
inputs = np.concatenate([train_X, test_X], axis=0)

targets = np.concatenate((train_y, test_y), axis=0)



model1 = XGBClassifier()

model1.fit(inputs, targets)



plot_importance(model1)

plt.show()
pred = model1.predict(test_df)

pred = pred.reshape(-1, 1)

pred[:7]
sub_df = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

sub_df1 = sub_df.copy()

sub_df1.Survived = pred

sub_df1.to_csv("My_sub_titanic.csv", index=False)

sub_df1[:7]