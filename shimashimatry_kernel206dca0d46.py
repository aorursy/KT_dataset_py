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

import seaborn as sns

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#ここでpredictのために、テストデータを読み込む。

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.describe
sns.distplot(train['SalePrice']);

#以下を見ての通り正規分布ではない
train.isnull().sum()[train.isnull().sum()>0].sort_values()
#欠損値の数が多すぎるので'PoolQC', 'MiscFeature', 'Alley'を削除

#そもそも重要視されていないから、欠損値が多いとも捉えられる

drop_list='PoolQC', 'MiscFeature', 'Alley'

for name in drop_list:

    train.drop(name, axis=1, inplace=True)

    test.drop(name, axis=1, inplace=True)

#Garageに関する変数は多い。最も重要な情報は'GarageCars'である気がするので他は削除

drop_list='GarageCond','GarageQual','GarageFinish','GarageType'

for name in drop_list:

    train.drop(name, axis=1, inplace=True)

    test.drop(name, axis=1, inplace=True)

#basementに関しては、日本に住んでるせいかよく分からんが

drop_list='BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF'

for name in drop_list:

    train.drop(name, axis=1, inplace=True)

    test.drop(name, axis=1, inplace=True)

train.corr()


#個人的にベニヤ板系の情報はいらんと勝手に思っていたが、まさかのMasVnrAreaがSalePriceと相関高かったので残す

#以下は某サイトのマルパクリ

#物件の広さを合計した変数を作成

train["TotalSF"]=train["1stFlrSF"]+train["2ndFlrSF"]+train["TotalBsmtSF"]

test["TotalSF"]=test["1stFlrSF"]+test["2ndFlrSF"]+test["TotalBsmtSF"]

#特徴量に1部屋あたりの面積を追加

train["FeetPerRoom"]=train["TotalSF"]/train["TotRmsAbvGrd"]

test["FeetPerRoom"]=test["TotalSF"]/test["TotRmsAbvGrd"]

#その他有効そうなものを追加する。

#建築した年とリフォームした年の合計

train['YearBuiltAndRemod']=train['YearBuilt']+train['YearRemodAdd']

test['YearBuiltAndRemod']=test['YearBuilt']+test['YearRemodAdd']

#バスルームの合計面積

train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) +

                               train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))

test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +

                               test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

#縁側の合計面積

train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] +

                              train['EnclosedPorch'] + train['ScreenPorch'] +

                              train['WoodDeckSF'])

test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +

                              test['EnclosedPorch'] + test['ScreenPorch'] +

                              test['WoodDeckSF'])

#プールの有無

train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#2階の有無

train['has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test['has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#ガレージの有無

train['hasgarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

test['hasgarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#地下室の有無

train['hasbsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['hasbsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

#暖炉の有無

train['hasfireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test['hasfireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#ガレージ年数

def garage_age(row):

    if row['GarageYrBlt'] == 0:

        return 0

    return (row['YrSold'] - row['GarageYrBlt'])

train['GarageAge'] = train.apply(garage_age, axis=1)

test['GarageAge'] = test.apply(garage_age, axis=1)

# House Age at sale point

train['HouseAge'] = train['YrSold'] - train['YearBuilt']

test['HouseAge'] = test['YrSold'] - test['YearBuilt']

# Remodelled?

def remodelled(row):

    if row['YearRemodAdd'] > row['YearBuilt']:

        return 1

    return 0

train['Remodelled'] = train.apply(remodelled, axis=1)

test['Remodelled'] = test.apply(remodelled, axis=1)

# Modelling or remodelling age

def last_modelled(row):

    year = max(row['YearBuilt'], row['YearRemodAdd'])

    return row['YrSold'] - year

train['LastModelled'] = train.apply(last_modelled, axis=1)

test['LastModelled'] = test.apply(last_modelled, axis=1)





#drop_list='1stFlrSF','2ndFlrSF','TotalBsmtSF','YearRemodAdd','FullBath','HalfBath','BsmtFullBath'\

#    ,'BsmtHalfBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'\

#    ,'PoolArea','GarageArea','Fireplaces'

    

#for name in drop_list:

#    train.drop(name, axis=1, inplace=True)

#    test.drop(name, axis=1, inplace=True)

plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice") 

#これは住宅の広さ、自分だったら最も重要視する点だが明らかに変なデータが２つある

#事故物件かな（´・ω・｀）

train=train.drop(train[(train['TotalSF']>7500)&(train['SalePrice']<300000)].index)
#これは完全に興味

data=pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#外れ値を除外する

train=train.drop(train[(train['YearBuilt']<2000)&(train['SalePrice']>600000)].index)
#OverallQualという変数がある。これもめちゃくちゃ大事な変数でSalePriceとの相関係数も高い

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#外れ値を除外する

train=train.drop(train[(train['OverallQual']<5)&(train['SalePrice']>200000)].index)

train=train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)

na_columns = train.isnull().sum()[train.isnull().sum()>0].sort_values().index.tolist()

train[na_columns].dtypes

#MasVnrArea,GarageYrBlt,LotFrontageだけ特別なことをしないと、おかしなことになる。

#他の変数はダミー変数化するときに勝手に欠損値の情報がいい感じになるから、無視する。
train.corr()
#tempdata=train.corr()

#for i in range(len(tempdata)-2):

#    tempdata=tempdata.drop(columns=tempdata.columns[0])

#tempdata=tempdata.drop(columns=tempdata.columns[1])

#tempdata=tempdata.abs()

#tempdata=list(tempdata.query('SalePrice < 0.05').index)

#tempdata

#欠損値のせいで相関が取れていないかどうか確認

test.drop("Id", axis = 1, inplace = True)

train.drop("Id", axis = 1, inplace = True)
#相関ないデータを消す。

#train.drop(tempdata, axis=1, inplace=True)

#オブジェクトデータがどれか確認

pd.set_option('display.max_columns', 50)

objectlist=list(train.select_dtypes(include=object).columns)

objectlist
#相関ないデータを消す。

#test.drop(tempdata, axis=1, inplace=True)

#オブジェクトの変数を全て消すという手法もあるが、ここでは丁寧にトレーニングとテストデータで共通して出現しないデータを消す。

#その変数が存在するかどうかの確認のためにシーボーンで箱ひげ図を出す必要がある。
test['MSSubClass']=test['MSSubClass'].apply(str)

train['MSSubClass']=train['MSSubClass'].apply(str)



test['YrSold']=test['YrSold'].astype(str)

train['YrSold']=train['YrSold'].astype(str)



test['MoSold']=test['MoSold'].astype(str)

train['MoSold']=train['MoSold'].astype(str)
sns.boxplot(x="MSZoning", y="SalePrice", data=train)
sns.boxplot(x="MSZoning", y="OverallQual", data=test)
sns.boxplot(x="Street", y="SalePrice", data=train)
sns.boxplot(x="Street", y="OverallQual", data=test)
sns.boxplot(x="LotShape", y="SalePrice", data=train)
sns.boxplot(x="LotShape", y="OverallQual", data=test)
sns.boxplot(x="LandContour", y="SalePrice", data=train)
sns.boxplot(x="LandContour", y="OverallQual", data=test)
sns.boxplot(x="Utilities", y="SalePrice", data=train)
#ちょっとしかデータがないみたいだから消す。いわば特異点

train.drop("Utilities", axis=1, inplace=True)

test.drop("Utilities", axis=1, inplace=True)

sns.boxplot(x="LotConfig", y="SalePrice", data=train)
sns.boxplot(x="LotConfig", y="OverallQual", data=test)
sns.boxplot(x="LandSlope", y="SalePrice", data=train)
sns.boxplot(x="LandSlope", y="OverallQual", data=test)
sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
sns.boxplot(x="Neighborhood", y="OverallQual", data=test)
#まさかのトレーニングとテストデータの出現するオブジェクトが全部一致（手動で確認しました）

sns.boxplot(x="Condition1", y="SalePrice", data=train)
sns.boxplot(x="Condition1", y="OverallQual", data=test)
sns.boxplot(x="Condition2", y="SalePrice", data=train)
sns.boxplot(x="Condition2", y="OverallQual", data=test)
train.drop("Condition2", axis=1, inplace=True)

test.drop("Condition2", axis=1, inplace=True)

sns.boxplot(x="BldgType", y="SalePrice", data=train)
sns.boxplot(x="BldgType", y="OverallQual", data=test)
sns.boxplot(x="HouseStyle", y="SalePrice", data=train)
sns.boxplot(x="HouseStyle", y="OverallQual", data=test)
train.drop("HouseStyle", axis=1, inplace=True)

test.drop("HouseStyle", axis=1, inplace=True)

sns.boxplot(x="RoofStyle", y="SalePrice", data=train)
sns.boxplot(x="RoofStyle", y="OverallQual", data=test)
sns.boxplot(x="RoofMatl", y="SalePrice", data=train)
sns.boxplot(x="RoofMatl", y="OverallQual", data=test)
train.drop("RoofMatl", axis=1, inplace=True)

test.drop("RoofMatl", axis=1, inplace=True)

sns.boxplot(x="Exterior1st", y="SalePrice", data=train)
sns.boxplot(x="Exterior1st", y="OverallQual", data=test)
train.drop("Exterior1st", axis=1, inplace=True)

test.drop("Exterior1st", axis=1, inplace=True)

sns.boxplot(x="Exterior2nd", y="SalePrice", data=train)
sns.boxplot(x="Exterior2nd", y="OverallQual", data=test)
train.drop("Exterior2nd", axis=1, inplace=True)

test.drop("Exterior2nd", axis=1, inplace=True)

sns.boxplot(x="MasVnrType", y="SalePrice", data=train)
sns.boxplot(x="MasVnrType", y="OverallQual", data=test)
sns.boxplot(x="ExterQual", y="SalePrice", data=train)
sns.boxplot(x="ExterQual", y="OverallQual", data=test)
sns.boxplot(x="ExterCond", y="SalePrice", data=train)
sns.boxplot(x="ExterCond", y="OverallQual", data=test)
sns.boxplot(x="Foundation", y="SalePrice", data=train)
sns.boxplot(x="Foundation", y="OverallQual", data=test)
sns.boxplot(x="BsmtQual", y="SalePrice", data=train)
sns.boxplot(x="BsmtQual", y="OverallQual", data=test)
sns.boxplot(x="BsmtCond", y="SalePrice", data=train)
sns.boxplot(x="BsmtCond", y="OverallQual", data=test)
sns.boxplot(x="BsmtExposure", y="SalePrice", data=train)
sns.boxplot(x="BsmtExposure", y="OverallQual", data=test)
sns.boxplot(x="Heating", y="SalePrice", data=train)
sns.boxplot(x="Heating", y="OverallQual", data=test)
train.drop("Heating", axis=1, inplace=True)

test.drop("Heating", axis=1, inplace=True)

sns.boxplot(x="HeatingQC", y="SalePrice", data=train)
sns.boxplot(x="HeatingQC", y="OverallQual", data=test)
sns.boxplot(x="CentralAir", y="SalePrice", data=train)
sns.boxplot(x="CentralAir", y="OverallQual", data=test)
sns.boxplot(x="Electrical", y="SalePrice", data=train)
sns.boxplot(x="Electrical", y="OverallQual", data=test)
train.drop("Electrical", axis=1, inplace=True)

test.drop("Electrical", axis=1, inplace=True)

sns.boxplot(x="KitchenQual", y="SalePrice", data=train)
sns.boxplot(x="KitchenQual", y="OverallQual", data=test)
sns.boxplot(x="Functional", y="SalePrice", data=train)
sns.boxplot(x="Functional", y="OverallQual", data=test)
sns.boxplot(x="FireplaceQu", y="SalePrice", data=train)
sns.boxplot(x="FireplaceQu", y="OverallQual", data=test)
sns.boxplot(x="PavedDrive", y="SalePrice", data=train)
sns.boxplot(x="PavedDrive", y="OverallQual", data=test)
sns.boxplot(x="Fence", y="SalePrice", data=train)
sns.boxplot(x="Fence", y="OverallQual", data=test)
sns.boxplot(x="SaleType", y="SalePrice", data=train)
sns.boxplot(x="SaleType", y="OverallQual", data=test)
sns.boxplot(x="SaleCondition", y="SalePrice", data=train)
sns.boxplot(x="SaleCondition", y="OverallQual", data=test)
na_columns = train.isnull().sum()[train.isnull().sum()>0].sort_values().index.tolist()

train[na_columns].dtypes
#こんだけやっても消えなかった数値データで欠損値を持つ変数を処理する

train.isnull().sum()[train.isnull().sum()>0].sort_values()
#どれも欠損数の全体としての割合は５０%を超えていないので、その列を消すという手法は選択しない

na_columns = train.isnull().sum()[train.isnull().sum()>0].sort_values().index.tolist()

for name in na_columns:

    if train[name].dtypes==float:

        train[name].fillna(train[name].mean(skipna=True), inplace=True)



na_columns = test.isnull().sum()[test.isnull().sum()>0].sort_values().index.tolist()

for name in na_columns:

    if test[name].dtypes==float:

        test[name].fillna(test[name].mean(skipna=True), inplace=True)





train.isnull().sum()[train.isnull().sum()>0].sort_values()
#ディープラーニングの準備

objectlist=list(train.select_dtypes(include=object).columns)

t_train=train['SalePrice'].values

train.drop('SalePrice', axis=1, inplace=True)

x_train=pd.get_dummies(train, columns=objectlist, drop_first=True)

test=pd.get_dummies(test, columns=objectlist, drop_first=True)

name_x_train_columns=x_train.columns.values

x_train.to_csv("x_train.csv",index=False)

x_train
test.to_csv("test.csv",index=False)

test
test.drop('MSSubClass_150', axis=1, inplace=True)



x_train=x_train.values

x_test=test.values
import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.optimizers import Adam

from keras.initializers import he_normal

from keras.callbacks import EarlyStopping,ModelCheckpoint
#標準化

#入力層に入れるデータの平均値と標準偏差を調べて配列に入れる。

x_train_mean=[]

x_train_std=[]

t_train_mean=[]

t_train_std=[]

for i in range(x_train.shape[1]):

    x_train_mean.append(np.mean(x_train[:,[i]]))

    x_train_std.append(np.std(x_train[:,[i]]))

t_train_mean=np.mean(t_train)

t_train_std=np.std(t_train)

#平均0で標準偏差１のデータに変える。

x_train-=x_train_mean

x_train/=x_train_std

x_test-=x_train_mean

x_test/=x_train_std

#なぜか一気に書くとバグったのでここだけ不格好

t_train=t_train-t_train_mean

t_train=t_train/t_train_std
#ニューラルネットワークのモデル作成

train_size=x_train.shape[1]

model=Sequential()

model.add(Dense(train_size, activation='relu', input_shape=(train_size,), kernel_initializer=he_normal()))

for node_num in [250,300,300,300,200,100,50,25,6]:

    model.add(Dense(node_num, activation='relu', kernel_initializer=he_normal()))

model.add(Dense(1))

model.summary()

#MSE(コンペはこれのルート、RMSEで評価するそうなのでこれにした)、Adam、評価関数（←これはなんでもいい。結果に影響しない）

#学習率0.001は文献値

model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001))#,metrics=['accuracy'])
epochs=500

stack = model.fit(x_train, t_train, epochs=epochs, batch_size=16,verbose=2)#,validation_split=0.1)
#import matplotlib.pyplot as plt

#x = range(epochs)

#plt.plot(x, stack.history['loss'], label="loss")

#plt.plot(x, stack.history['val_loss'], label="val_loss")

#plt.title("MSE")

#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#plt.ylim([0,0.0001])

#plt.show()

#過学習現象が起きていなかった→母集団が同じだと断定！
expect=model.predict(x_test)

expect=expect*t_train_std+t_train_mean
expect
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = expect



sub[["Id","SalePrice"]].head()



sub[["Id","SalePrice"]].to_csv("submission.csv",index=False)