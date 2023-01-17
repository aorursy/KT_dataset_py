# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import numpy.linalg as lg
import math
import os
print(os.listdir("../input"))
import random

def changeFeature(pandasDFname,featureName,featureList):
    num=len(featureList)
    Fea=pandasDFname.loc[:,featureName]
    pandasDFname.drop(columns=featureName,inplace=True)
    for n in Fea.iteritems():
        for i in list(range(num)):
            if n[1]==featureList[i]:
                pandasDFname.loc[n[0],featureName+str(i)]=1
            else:
                pandasDFname.loc[n[0],featureName+str(i)]=0
                
def sigmod(x):
    if x>0:
        return 1
    if x<0:
        return 0
    if x==0:
        return random.randint(0,1)

def normalized(DataFrameName,featureName):
    divi=max(abs(DataFrameName.loc[:,featureName].max()),abs(DataFrameName.loc[:,featureName].min()))
    DataFrameName.loc[:,featureName]=DataFrameName.loc[:,featureName]/divi
# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
train.drop(columns=['Ticket','Name','Cabin','PassengerId'],inplace=True)

#填充年龄段缺失值，平均值
train.loc[:,'Age'].fillna(train.loc[:,'Age'].mean(),inplace=True)
train.loc[:,'Embarked'].fillna('S',inplace=True)

#归一化
for i in ['Pclass','Age','Fare']:
    normalized(train,i)

#将特征值向量化
changeFeature(train,'Sex',['male','female']) 
changeFeature(train,'Embarked',['S','C','Q'])

#设置常数特征
train.loc[:,'b']=1
train.head()
#创造训练要用的矩阵
x=train.drop(columns='Survived').as_matrix(columns=None)
y=train.loc[:,'Survived'].as_matrix(columns=None)
#计算系数矩阵
xt=np.transpose(x)
temp=np.dot(xt,x)
temp=lg.inv(temp)
temp=np.dot(temp,xt)
w=np.dot(temp,y)
sub=pd.read_csv('../input/gender_submission.csv')
sub.head()
test=pd.read_csv('../input/test.csv')
test.drop(columns=['Ticket','Name','Cabin',],inplace=True)

#填充年龄段缺失值，平均值
test.loc[:,'Age'].fillna(test.loc[:,'Age'].mean(),inplace=True)
test.loc[:,'Embarked'].fillna('S',inplace=True)

#归一化
for i in ['Pclass','Age','Fare']:
    normalized(test,i)

#将特征值向量化
changeFeature(test,'Sex',['male','female']) 
changeFeature(test,'Embarked',['S','C','Q'])

#设置常数特征
test.loc[:,'b']=1
test.head()
x=test.drop(columns='PassengerId').as_matrix(columns=None)
resultdf=test.loc[:,['PassengerId','Survived']]
for i in list(range(len(np.dot(x,w)))):
    resultdf.loc[i,'Survived']=sigmod(np.dot(x,w)[i])
test.to_csv('result.csv', index=False)