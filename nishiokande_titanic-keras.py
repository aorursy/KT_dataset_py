import numpy as np

import pandas as pd
#データセットの読み込み

train_csv = pd.read_csv("../input/titanic/train.csv")

test_csv = pd.read_csv("../input/titanic/test.csv")
#trainデータの確認

train_csv.head(3)
#testデータの確認

test_csv.head(3)
#trainデータのいらない列をdropして確認

train_csv = train_csv.drop(["PassengerId", "Name", "Ticket","Cabin"], axis = 1)

train_csv.head(3)
#testデータのいらない列をdropして確認

test_csv = test_csv.drop(["Name", "Ticket","Cabin"],axis = 1)

test_csv.head(3)
#trainデータの欠損値を求める

train_csv.isnull().sum()
#trainの欠損値を埋めるためにEmbarkedの最頻値を求める

pd.value_counts(train_csv["Embarked"])
#Ageの欠損値を求める

train_csv["Age"].mean()
#Embarkedの欠損値を最頻値で埋める

_ = train_csv.fillna({"Embarked" : "S"}, inplace = True)

_ = train_csv.fillna({"Age" : 29.699}, inplace = True)
#testデータの欠損値を求める

test_csv.isnull().sum()
#testのFareの欠損値を埋めるために平均値を求める

test_csv["Fare"].mean()
test_csv["Age"].mean()
_ = test_csv.fillna({"Fare" : 35.627}, inplace = True)

_ = test_csv.fillna({"Age" : 30.272}, inplace = True)
#文字列を数値に変換するためにlabelencoderを使う

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()



labels = ["Embarked", "Sex"]

for label in labels:

    train_csv[label] = LE.fit_transform(train_csv[label])

    test_csv[label] = LE.fit_transform(test_csv[label])
#trainデータとtestデータを作成

from sklearn.model_selection import train_test_split



x = train_csv.drop(["Survived"], axis = 1)

y = train_csv["Survived"]
#色々インポート

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.layers.advanced_activations import LeakyReLU
alpha = 0.01

dropout = 0.3

epochs = 300

batch_size = 32



#モデル

model = Sequential()



#入力層

model.add(Dense(7,input_dim = 7))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))



#中間層

model.add(Dense(100))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))

model.add(Dense(50))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))

model.add(Dense(25))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))

model.add(Dense(10))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))

model.add(Dense(5))

model.add(LeakyReLU(alpha = alpha))

model.add(Dropout(dropout))



#出力層

model.add(Dense(1, activation = 'sigmoid'))



#モデルの概要

model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = ['accuracy'])

result = model.fit(x, y, epochs = epochs, batch_size = batch_size)
#正答率描画

import matplotlib.pyplot as plt



plt.plot(range(1, epochs + 1), result.history['acc'], label = 'training')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()
#学習曲線描画

plt.plot(range(1, epochs + 1), result.history['loss'], label = 'training')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()
test_data = test_csv.drop(['PassengerId'], axis = 1)

predict = np.round(model.predict(test_data))

prediction = pd.DataFrame(predict)
submission = pd.concat([test_csv[['PassengerId']],prediction],axis = 1)

submission = submission.rename(columns = {0 : 'Survived'})

_ = submission.fillna({'Survived' : 0}, inplace = True)

submission = submission.astype(np.int64)

submission.head()
submission.to_csv("submission.csv", index = False)