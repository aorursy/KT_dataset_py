# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import  matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import data
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head(5)
#查看总体信息（数据类型以及缺失）
train.info()
test.info()
def FeatureSelect(df):

    cols_to_drop = ['matchDuration','rankPoints','Id','matchType','groupId','numGroups','matchId', 'roadKills', 'teamKills', 'maxPlace','winPoints','killPoints']
    cols_to_fit = [cols for cols in df.columns if cols not in cols_to_drop]
    df = df[cols_to_fit]

    return df
train = FeatureSelect(train)
test= FeatureSelect(test)


corr = train.corr()
train.head()
#数据可视化:画相关系数图，直观获取数据相关性
plt.figure(figsize=(10,10))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    annot=True,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
plt.show()
from sklearn.model_selection import train_test_split
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]

X_Test = test
#使用线性模型训练并预测
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(X_Train,Y_Train)
y_pred = model.predict(X_Test)
print(y_pred.shape)
print(test_idx.shape)
#Make into submission
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["winPlacePerc"] = y_pred
test_pred.columns = ["Id", "winPlacePerc"]
test_pred.to_csv("submission.csv", index=False) # submission