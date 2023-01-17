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
from matplotlib import pyplot as plt

import seaborn as sns

from keras import models

from keras import layers

import keras.optimizers

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.layers import Dropout

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.describe
train.corr()
train.columns
train["SalePrice"].describe()
plt.figure(figsize=(20, 10))

sns.distplot(train["SalePrice"])
test_x = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]
plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
train = train.drop(train[(train["TotalSF"]>7500) & (train["SalePrice"]<300000)].index)
plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)
data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)
plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test_x],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test_x['Id']



all_data.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test_x],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test_x['Id']



all_data.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na
# 欠損値があるカラムをリスト化

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()



#欠損があるカラムのデータ型を確認

all_data[na_col_list].dtypes.sort_values()
#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")



#欠損値が全て置換できているか確認

all_data.isnull().sum()[all_data.isnull().sum() > 0]
#相関係数が高いリスト（検証に使うリスト）

#"SalePrice"含む

high_correlation_list = ["OverallQual", "YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", 

                          "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", "GarageArea", "SalePrice"]

#"SalePrice"含まない

high_correlation_list_0 = ["OverallQual", "YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", 

                          "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", "GarageArea"]



#相関係数が高かったものだけを抽出

train_correlation = train[high_correlation_list]

test_correlation = test[high_correlation_list_0]
#訓練データとテストデータ



x_train = train_correlation.iloc[:, :-1].values.astype("float32")

x_targets = train["SalePrice"].values.astype("float32")



y_train = test_correlation.values.astype("float32")





#正規化



#インスタンス作成

stdsc = StandardScaler()



mean_train = np.mean(x_train, axis=0)

std_train = np.std(x_train, axis=0)



for i in range(len(high_correlation_list)-1):

    y_train[:, i] = (y_train[:, i] - mean_train[i]) / std_train[i]



x_targets = x_targets.reshape(-1, 1)



x_train_ss = stdsc.fit_transform(x_train)

x_targets_ss = stdsc.fit_transform(x_targets)
#k分割交差検証



k = 4#分割数

num_val_data = len(x_train) // k#検証データ数

num_epochs = 50

batch_size = 32



mae_all_scores = []

loss_all_scores = []

val_loss_all_scores = []



for i in range(k):

    print(i+1, "回目")



    #検証データとラベル

    val_data = x_train_ss[i*num_val_data: (i+1)*num_val_data]

    val_targets = x_targets_ss[i*num_val_data: (i+1)*num_val_data]



    #訓練データとラベル

    partial_train_data = np.concatenate([x_train_ss[: i*num_val_data], x_train_ss[(i+1)*num_val_data: ]], axis = 0)

    partial_train_targets = np.concatenate([x_targets_ss[: i*num_val_data], x_targets_ss[(i+1)*num_val_data: ]], axis = 0)



    model = models.Sequential()

    model.add(layers.Dense(16, activation="relu", input_shape=(len(high_correlation_list)-1, )))

    model.add(layers.Dense(16, activation="relu"))

    model.add(layers.Dense(1))



    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    

    history = model.fit(partial_train_data, partial_train_targets, 

                        epochs=num_epochs, 

                        batch_size=batch_size, 

                        validation_data=(val_data, val_targets), 

                        verbose = 0

                       )



    mae_history = history.history["mae"]

    loss_history = history.history["loss"]

    val_loss_history = history.history["val_loss"]



    mae_all_scores.append(mae_history)

    loss_all_scores.append(loss_history)

    val_loss_all_scores.append(val_loss_history)







print("終了")
average_mae_history = [np.mean([x[i] for x in mae_all_scores]) for i in range(num_epochs)]

average_loss_history = [np.mean([x[i] for x in loss_all_scores]) for i in range(num_epochs)]

average_val_loss_history = [np.mean([x[i] for x in val_loss_all_scores]) for i in range(num_epochs)]



plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.grid(True)

plt.show()



plt.plot(range(1, len(average_loss_history) + 1), average_loss_history, color = "red", label="loss")

plt.plot(range(1, len(average_val_loss_history) + 1), average_val_loss_history, color = "blue", label="val_loss")

plt.legend()

plt.grid(True)

plt.show()
ans_ss = model.predict(y_train)

ans = stdsc.inverse_transform(ans_ss)

#ans = (model.predict(y_train) * targets_std) + targets_mean

print(ans.shape)



df_sub_pred = pd.DataFrame(ans).rename(columns={0: "SalePrice"})

df_sub_pred = pd.concat([test["Id"], df_sub_pred["SalePrice"]], axis=1)





df_sub_pred.to_csv("house_price_kekka.csv", index=False)

df_sub_pred.head()