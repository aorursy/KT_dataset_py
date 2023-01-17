# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/year_prediction.csv")

# Group release years into decades

df['label'] = df.label.apply(lambda year : year-(year%10))



train = df.iloc[:463715]

test = df.iloc[-51630:]



print( train.shape)

train_labels = train['label']

train_features = train.drop("label", axis=1)

test_labels= test['label']

test_features = test.drop("label", axis=1)

train_features.describe()
any(train_features.isna().sum() > 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_features_scaled = scaler.fit_transform(train_features.values)

test_features_scaled = scaler.transform(test_features.values)

train_features = pd.DataFrame(train_features_scaled, columns=train_features.columns, index=train_features.index)

test_features = pd.DataFrame(test_features_scaled, columns=test_features.columns, index=test_features.index)
f,ax=plt.subplots(1,2,figsize=(16,7))

sns.countplot(train_labels, ax=ax[0])

sns.countplot(test_labels, ax=ax[1])

ax[0].set_title("Train Labels Dist")

ax[1].set_title("Test Labels Dist")

plt.show()
train = train_features

train['label'] = train_labels

test = test_features

test['label'] = test_labels

print (train_features.shape, test_features.shape)

print (train_labels.shape, test_labels.shape)



train = train[train['label'] > 1940]

test = test[test['label'] > 1940]

# Borrowing Code fr//om https://www.kaggle.com/vinayshanbhag/predict-release-timeframe-from-audio-features for downsampling

min_samples = train.label.value_counts().min() 

decades = train.label.unique()

df_sampled = pd.DataFrame(columns=train.columns)

for decade in decades:

    df_sampled = df_sampled.append(train[train.label==decade].sample((min_samples)))

df_sampled.label = df_sampled.label.astype(int)



train_labels =df_sampled['label']

train_features = df_sampled.drop("label", axis=1)

test_labels= test['label']

test_features = test.drop("label", axis=1)

print (train_features.shape, test_features.shape)

print (train_labels.shape, test_labels.shape)

f,ax=plt.subplots(1,2,figsize=(16,7))

sns.countplot(train_labels, ax=ax[0])

sns.countplot(test_labels, ax=ax[1])

ax[0].set_title("Train Labels Dist")

ax[1].set_title("Test Labels Dist")

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score
lgbm = LGBMClassifier()

rf = RandomForestClassifier()

scaler = StandardScaler()

pipeline1 = Pipeline([('scaler', scaler), ('lgbm', lgbm)])

pipeline2 = Pipeline([('scaler', scaler), ('rf', rf)])

print( cross_val_score(pipeline1, train_features, train_labels, cv=5))

print( cross_val_score(pipeline2, train_features, train_labels, cv=5))
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import keras as kr

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers.core import Dense

from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedShuffleSplit
lb = LabelBinarizer()

trainY = lb.fit_transform(train_labels)

testY = lb.transform(test_labels)

train_features, trainY = shuffle(train_features, trainY)

scaler = StandardScaler()

train_standardized = scaler.fit_transform(train_features)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)

label_count = len(test_labels.unique())
def get_network():

    model = kr.models.Sequential()

    model.add(Dense(20, input_shape=(train_features.shape[1],), activation="relu"))

    model.add(Dense(20, activation="relu"))

    model.add(Dense(label_count, activation="softmax"))

    opt = "adam"

    model.compile(loss= "categorical_crossentropy", optimizer=opt, metrics=["accuracy"], )

    return model
model = get_network()

scikit_net= KerasClassifier(build_fn=get_network, epochs=10, batch_size=40)

print( cross_val_score(scikit_net, train_features, trainY, cv=5))
from sklearn.metrics import accuracy_score

print("LGB accuracy")

pipeline1.fit(train_features, train_labels)

print(accuracy_score(pipeline1.predict(test_features), test_labels))

print("RF accuracy")

pipeline2.fit(train_features, train_labels)

print(accuracy_score(pipeline2.predict(test_features), test_labels))

model.fit(train_features, trainY, batch_size=40, epochs=5)

model.evaluate(test_features, testY)
preds = model.predict(test_features)
pred_single = [i.argmax() for i in preds]

label_single = [i.argmax() for i in testY]
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(label_single, pred_single)

cm_df= pd.DataFrame(cm)

cm_df = cm_df.apply(lambda x: x/sum(x), axis ='columns')

sns.heatmap(cm_df)
cm_df