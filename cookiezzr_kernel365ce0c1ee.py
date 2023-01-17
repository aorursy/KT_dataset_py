# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_info1 = pd.read_csv("../input/train.csv",index_col=0)

train_target = data_info1.loc[:,"SalePrice"]

data_info1.drop(labels="SalePrice",axis=1,inplace=True)

data_info2 = pd.read_csv("../input/test.csv",index_col=0)

data_info = pd.concat([data_info1,data_info2],axis=0)

data_info.shape
#查看train的label是甚么样

draw = pd.DataFrame( { "price":train_target,"log(price+1)":np.log1p(train_target)  }  )

draw.hist()   #绘制df列直方图
#对label标签进行映射，服从高斯分布

train_target = np.log1p(train_target)

data_info.loc[1]
data_info.isnull().sum().sort_values(ascending=False)
#先处理缺失值

data_info.isnull().sum().sort_values(ascending=False)



#drop掉  PoolQC  MiscFeature Alley Fence 值因为缺的太多了

col = ["PoolQC","MiscFeature","Alley","Fence"]

data_info.drop(labels=col, axis=1, inplace=True)



#缺省值适中，可以把NA当成一个新类别

#GarageCond GarageQual GarageFinish GarageType   车库

#BsmtCond  BsmtQual BsmtExposure BsmtFinType2 BsmtFinType1  地下室整体情况

#MasVnrType  

#FireplaceQu(壁炉)数量

col = ["GarageCond" ,"GarageQual", "GarageFinish", "GarageType",

      "BsmtCond" , "BsmtQual" ,"BsmtExposure", "BsmtFinType2" ,"BsmtFinType1",

      "MasVnrType" , "FireplaceQu"]

data_info[col] = data_info[col].fillna("N")



#进行填充离散值 按列进行填充！！！！！

col = ["BsmtFinSF1", "BsmtFinSF2","BsmtUnfSF", "TotalBsmtSF", "GarageArea"]

means = data_info[col].mean()

for k,v in zip(col,means):

    data_info.fillna({k:v},inplace=True)



#进行one-hot编码 只会对string类型数据进行

#Using the get_dummies will create a new column for every unique string in a certain column

data_info = pd.get_dummies(data_info) #第一遍系统会自己检测，需要二遍！！！

col_1 = ["BsmtFullBath", "BsmtHalfBath" , "GarageCars"]

hot_info = pd.get_dummies(data_info[col_1],columns=col_1)

data_info = pd.concat([hot_info,data_info],axis=1)

data_info.drop(labels=col_1,inplace=True,axis=1)
#连续型数据字段可以进行离散化（加step）/直接预测拟合（random forest）

#我将进行拟合  GarageYrBlt（年份） LotFrontage   MasVnrArea

def set_Feature(data):

    from sklearn.ensemble import RandomForestRegressor

    labels = ["GarageYrBlt", "LotFrontage"  , "MasVnrArea"]#回归问题

    model = RandomForestRegressor(n_estimators=300)

    #data_info.columns[data_info.dtypes!="object"]

    for label in labels:

        X = data.loc[ data[label].notnull(),:]

        X = X.drop(labels=labels,axis=1)

        y = data.loc[ data[label].notnull(),label]

        y_pre = data.loc[ data[label].isnull() ,:]

        y_pre = y_pre.drop(labels=labels,axis=1)

        model.fit(X,y)

        pred = model.predict( y_pre )

        data.loc[ data[label].isnull() ,label] = pred

    return data

data_info = set_Feature(data_info)
data_info.isnull().sum().sort_values(ascending=False)
#处理好所有缺失值数据,进行标准化数据

mean_numeric = data_info.mean()

std_numeric = data_info.std()

data_info = (data_info-mean_numeric)/std_numeric
#分离数据 

#loc行和列是基于索引的，iloc行和列是基于行号列号的 ix是可以混用(效率低)

train_target = train_target  

train_feature = data_info.loc[data_info1.index]

test_feature = data_info.loc[data_info2.index]

print(train_target.shape,train_feature.shape,test_feature.shape)

print(type(train_target))
#建模

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

# train_feature = train_feature.values  #转化为Numpy

# train_target = train_target.values   #转化为Numpy

# weight_c = np.logspace(-1,3)

scores = []

for c in weight_c:

    score = np.sqrt(-cross_val_score( LogisticRegression(max_iter=10000,C=c,solver="lbfgs",multi_class="auto"),train_feature,train_target.astype("int"),cv=5,scoring='neg_mean_squared_error'))

    scores.append(np.mean(score))

plt.plot(weight_c,scores)

plt.show()

type(train_feature)
np.logspace(-1,3) #输入y值（-1，3）计算x的值！！！