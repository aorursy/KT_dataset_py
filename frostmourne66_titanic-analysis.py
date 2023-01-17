# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor 
# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')

fig = plt.figure()
fig.set(alpha = 0.2)

survived_0 = data_train.Pclass[data_train.Survived ==0].value_counts()
survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived':survived_1,'unsurvived':survived_0})
df.plot(kind='bar',stacked=True)
plt.xlabel('class of passengers')
plt.ylabel('survived or not')
plt.show()
data_train.Cabin.value_counts()
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']] #把已有的数据特征放进df中
    
    known_age = age_df[age_df.Age.notnull()].as_matrix() #Convert the frame to its Numpy-array representation.
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:,0] #目标年龄
    x = known_age[:,1:] #特征属性值
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    predictAges = rfr.predict(unknown_age[:,1::]) # 用得到的模型进行未知年龄结果预测
    df.loc[(df.Age.isnull()),'Age'] = predictAges
    
    return df,rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
