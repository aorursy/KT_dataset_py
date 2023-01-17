#ライブラリ

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression



%matplotlib inline
#データ読み込み

train= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')
print(f"trainデータの形状:{train.shape}")

print(f"testデータの形状:{test.shape}")
train.info()
#データの損失がある部分は白くなる

import missingno as msno

msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
#欠損値をカウント

def missing_values_table(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() /len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)

    mis_val_table_ren_columns = mis_val_table.rename(columns={0:'Missing Values', 1:'% of Total Values'})

    #欠損値の割合

    mis_val_table_ren_columns=mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1]!=

                                                        0].sort_values('% of Total Values',ascending=False).round(1)

    #カラム数を表示

    print("You selected dataframe has" + str(df.shape[1]) + "columns.\n"

         "There are" + str(mis_val_table_ren_columns.shape[0]) + 

         "columns that have missing values.")

    

    #dataframe中の欠損値の情報

    return mis_val_table_ren_columns



missing_values_table(train)
#今回予想するのは"SalePrice"である. その分布は

sns.distplot(train['SalePrice'])

#歪度(skewness) and 尖度(kurtosis)

print(f"Skewness:{train['SalePrice'].skew()}")

print(f"Kurtosis:{train['SalePrice'].kurt()}")
#正規分布に従うデータが一般的なので対数を取っておく(予測しやすくなる)

train['SalePrice_Log'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice_Log'])

#歪度(skewness) and 尖度(kurtosis)

print(f"Skewness:{train['SalePrice_Log'].skew()}")

print(f"Kurtosis:{train['SalePrice_Log'].kurt()}")

print(f"Mean:{train['SalePrice_Log'].mean()}")

print(f"Std:{train['SalePrice_Log'].std()}")

train = train.drop(['SalePrice'], axis=1)
#pandas_profileとやらを使ってみる #ピアソンの積率相関とスピアマンの順位相関

pandas_profiling.ProfileReport(train)
print(train.dtypes)

print(test.dtypes)
train.describe()
# plt.hist(train['BsmtFinSF1'], alpha=0.5, bins=20)

# plt.hist(test['BsmtFinSF1'], alpha=0.5, color='indianred', bins=20)





sns.distplot(train['BsmtFinSF1'],bins=20, label='train')

sns.distplot(test['BsmtFinSF1'], bins=20, label='test') #colorの指定も可能
#Label Encoder

from sklearn.preprocessing import LabelEncoder

print(f"Before : {train.columns[train.dtypes == object]}")

for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

        

train = train.fillna(-1)

test = test.fillna(-1)

print(f"After : {train.columns[train.dtypes == object]}")
#各説明変数に対し, 目的変数をプロット

fig = plt.figure(figsize=(20,20))

for i in np.arange(29-1):

    ax = fig.add_subplot(7,4,i+1)

    sns.regplot(x=train.drop('SalePrice_Log', axis=1).iloc[:,i],

                y=train['SalePrice_Log'])



plt.tight_layout()

plt.show()
#各特徴同士の相関(めちゃめちゃ時間かかるし見にくいけど一発)

pd.scatter_matrix(train.iloc[:,1:10])
#わからん時はFeatureImportanceを出してくれるモデルに突っ込んでみる

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=80, max_features='auto')

rf.fit(train.drop(['SalePrice_Log'], axis=1), train['SalePrice_Log'].values)

print('Training done using Random Forest')



ranking=np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11,9))

sns.barplot(x=rf.feature_importances_[ranking],

            y=train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
#データ読み込み

train= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')
#matplot & pandas

train["OverallQual"].value_counts().plot(kind="bar")
#seabornで(一行で書けるが, 順番が変わる)

sns.barplot(train.OverallQual,train.SalePrice)
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# violinplots 

sns.violinplot(data=train,x="OverallQual", y="SalePrice")
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# カテゴリカル変数

train['Street'].value_counts()
#カテゴリカル変数

train['MSSubClass'].head()