import pandas as pd

import numpy as np

import math
data = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")

data.shape
data.info()
data.head()
#将title，text填充为空，len（）=0

data['Title']=data['Title'].fillna('')

data['Review Text']=data['Review Text'].fillna('')
data.head()
#因确实数据较少，将其替换为各类中最多的数据

data['Division Name']=data['Division Name'].fillna('General')

data['Department Name']=data['Department Name'].fillna('Tops')

data['Class Name']=data['Class Name'].fillna('Dresses')

data.head()
data.info()
def get_lenth(x):

    if type(x)==str:

        

        return math.ceil(len(x)/10)

#get_lenth('qweddff df')
# title_df=pd.DataFrame()

# title_df=data['Title'].map(get_lenth)

# title_df.head()
data['Title']=data['Title'].map(get_lenth)

data['Title']
def get_text(x):

    if type(x)==str:

        return math.ceil(len(x)/50)
data['Review Text']=data['Review Text'].map(get_text)

data['Review Text']
DN_Df = pd.DataFrame()

DN_Df = pd.get_dummies(data['Division Name'] , prefix='DN' )

DN_Df.head()
data = pd.concat([data,DN_Df],axis=1)

data.drop('Division Name',axis=1,inplace=True)

data.head()
Dep_Df = pd.DataFrame()

Dep_Df = pd.get_dummies(data['Department Name'] , prefix='Dep' )

DN_Df.head()
data = pd.concat([data,Dep_Df],axis=1)

data.drop('Department Name',axis=1,inplace=True)

data.head()
CN_Df = pd.DataFrame()

CN_Df = pd.get_dummies(data['Class Name'] , prefix='CN' )

CN_Df.head()
data = pd.concat([data,CN_Df],axis=1)

data.drop('Class Name',axis=1,inplace=True)

data.head()
data.shape
#相关性矩阵

corrDf = data.corr() 

corrDf
corrDf['Recommended IND'].sort_values(ascending =False)
full_X = pd.concat([data['Rating']],axis=1)

full_X.head()
source_X=data.loc[:,'Rating']



source_y=data.loc[:,'Recommended IND']

source_X,source_y
from sklearn.model_selection import train_test_split 



#建立模型用的训练数据集和测试数据集

train_X, test_X, train_y, test_y = train_test_split(source_X ,

                                                    source_y,

                                                    train_size=.8)



#输出数据集大小

print ('原始数据集特征：',source_X.shape, 

       '训练数据集特征：',train_X.shape ,

      '测试数据集特征：',test_X.shape)



print ('原始数据集标签：',source_y.shape, 

       '训练数据集标签：',train_y.shape ,

      '测试数据集标签：',test_y.shape)
train_X=train_X.values.reshape(-1,1)

train_y=train_y.values.reshape(-1,1)

train_X,train_y
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(train_X,train_y)
test_X=test_X.values.reshape(-1,1)

test_y=test_y.values.reshape(-1,1)

test_X,test_y
model.score(test_X,test_y)