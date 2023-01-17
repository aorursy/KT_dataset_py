from sklearn import model_selection
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train2= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',na_filter=False)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

print(train2.iloc[:,1:11])
from sklearn.preprocessing import LabelEncoder
#LabelEncoderのインスタンスを生成
le = LabelEncoder()
#ラベルを覚えさせる
le = le.fit(train2['MSSubClass'])
#ラベルを整数に変換
train2['MSSubClass'] = le.transform(train2['MSSubClass'])

le = le.fit(train2['MSZoning'])
#ラベルを整数に変換
train2['MSZoning'] = le.transform(train2['MSZoning'])

le = le.fit(train2['Street'])
#ラベルを整数に変換
train2['Street'] = le.transform(train2['Street'])

le = le.fit(train2['Alley'])
#ラベルを整数に変換
train2['Alley'] = le.transform(train2['Alley'])

le = le.fit(train2['LotShape'])
#ラベルを整数に変換
train2['LotShape'] = le.transform(train2['LotShape'])

le = le.fit(train2['LandContour'])
#ラベルを整数に変換
train2['LandContour'] = le.transform(train2['LandContour'])

le = le.fit(train2['Utilities'])
#ラベルを整数に変換
train2['Utilities'] = le.transform(train2['Utilities'])

le = le.fit(train2['LotConfig'])
#ラベルを整数に変換
train2['LotConfig'] = le.transform(train2['LotConfig'])

le = le.fit(train2['LandSlope'])
#ラベルを整数に変換
train2['LandSlope'] = le.transform(train2['LandSlope'])

le = le.fit(train2['Neighborhood'])
#ラベルを整数に変換
train2['Neighborhood'] = le.transform(train2['Neighborhood'])

le = le.fit(train2['Condition1'])
#ラベルを整数に変換
train2['Condition1'] = le.transform(train2['Condition1'])

le = le.fit(train2['Condition2'])
#ラベルを整数に変換
train2['Condition2'] = le.transform(train2['Condition2'])
train2['LotFrontage']=train2['LotFrontage'].replace('NA','0')
train2['LotFrontage'].to_csv('frontage.csv', index = False)
train_price=round(train2['SalePrice']/50000)
plt.scatter(train_price, train['SalePrice']);
feature_sub= np.array(train2.iloc[:,1:11], dtype='int32')
train_price=np.array(train_price, dtype='int32')
print(feature_sub.dtype)
print(train_price.dtype)

train_price2=tf.convert_to_tensor(train_price)
feature=tf.convert_to_tensor(feature_sub)
print(train_price2.dtype)
print(feature.dtype)


X_train,X_test,y_train,y_test=model_selection.train_test_split(feature,train_price2,test_size=0.2)

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,activation=tf.nn.relu),
    tf.keras.layers.Dense(15,activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
             loss="mse",
             metrics=["accuracy"])
model.fit(X_train,y_train,epochs=500)
