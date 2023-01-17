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
import pandas as pd

import numpy as np

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso, ElasticNet

import matplotlib.pyplot as plt



train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ
train_saleprice = train["SalePrice"].values
# サンプルから欠損値と割合、データ型を調べる関数

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'欠損値', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['欠損値'], ascending=False)

#欠損値の確認

A=Missing_table(train)

print(A)

train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)



# 訓練データ特徴量をリスト化

cat_cols = alldata.dtypes[train.dtypes=='object'].index.tolist()

num_cols = alldata.dtypes[train.dtypes!='object'].index.tolist()



other_cols = ['Id','WhatIsData']

# 余計な要素をリストから削除

cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去

num_cols.remove('Id') #Id削除



cat = pd.get_dummies(alldata[cat_cols])



# データ統合

all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)

all_data.describe()



Fig1=plt.figure(tight_layout=True)

ax1=Fig1.add_subplot(1,1,1)



ax1.hist(train_saleprice, bins=100)

#ax1.plot(z,x7,color='red')

ax1.set_title('train_SalePrice',fontsize=12)
Fig2=plt.figure(tight_layout=True)

ax2=Fig2.add_subplot(1,1,1)



ax2.hist(np.log(train['SalePrice']), bins=50)

ax2.set_title('log(train_SalePrice)',fontsize=12)

#ax2.set_xlabel("z",fontsize=16)

#ax2.set_ylabel("S(z)",fontsize=16)

#ax2.grid()

#Fig2.savefig("log(train_SalePrice_2)"+'.pdf',bbox_inches="tight")

#plt.grid()

plt.show()
# ElasticNetによる回帰計算

train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)



alpha = 0.001

l1_ratio=0.7



x_ = train_.drop('SalePrice',axis=1)

y_ = train_.loc[:, ['SalePrice']]

y_ = np.log(y_)



enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

enet.fit(x_, y_)

print(f"training dataに対しての精度: {enet.score(x_, y_):.2}")



#テストデータへの検証

test_feature = test_.drop('Id',axis=1)

prediction = np.exp(enet.predict(test_feature))



# Idを取得

Id = np.array(test["Id"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

result = pd.DataFrame(prediction, Id, columns = ["SalePrice"])

# my_tree_one.csvとして書き出し

result.to_csv("prediction_regression3.csv", index_label = ["Id"])
