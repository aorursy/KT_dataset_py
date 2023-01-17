import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline
#データの読み込み

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

sub_df = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train_df.head()
len(train_df)
#trainデータの欠損値確認

train_df.isnull().sum()
#testデータの欠損値確認

test_df.isnull().sum()
#AgetとFareだけでやってみる

columns = ["Age","Fare"]

#null値補間

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

test_df['Fare'] = test_df['Fare'].fillna(test_df['Age'].median())

#必要なデータを取り出し、ndarrayに変換

age_fare = train_df[columns].values

survived = train_df["Survived"].values

test_test = test_df[columns].values
#学習用データと評価用データに分ける（8:2）

x_train,x_test,y_train,y_test = model_selection.train_test_split(age_fare,survived,test_size=0.2)
#ニューラルネットワークのモデル作成(層の数、ノードの数、活性化関数を決める)

model = keras.Sequential([

    keras.layers.Dense(2, activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])
#学習スタイル、損失関数、評価方法を決める

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#学習させる

model.fit(x_train,y_train,epochs=10)
#予測精度の表示

pred = model.predict(x_test).argmax(axis=1)

accuracy_score(y_test, pred)
#testデータの予測と提出用csv出力

sub_pred = model.predict(test_test).argmax(axis=1)

sub_df["Survived"] = sub_pred

sub_df.to_csv("decision_tree.csv",index=False)