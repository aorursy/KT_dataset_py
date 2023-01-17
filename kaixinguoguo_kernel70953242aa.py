# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/wwwkagglecomkaixinguoguo/ma_resp_data_temp.csv")
df.head()
df.info()
df["KBM_INDV_ID"]=df["KBM_INDV_ID"].astype("object") 
df.describe().T
#可视化数据分布

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("seaborn")

plt.rcParams["font.sans-serif"]=["SimHei"]

plt.rcParams["axes.unicode_minus"]=False
#统计目标标签是否平衡

df.resp_flag.value_counts()[0]/df.shape[0]
df.resp_flag.value_counts()[1]/df.shape[0]
plt.figure(figsize=(10,3))  #设置画布大小

sns.countplot(y="resp_flag",data=df)#分类柱状图countplot

plt.show()
#查看总体的年龄分布情况

sns.distplot(df["age"],bins=20)
#是否购买保险年龄段

import seaborn as sns

sns.kdeplot(df.age[df.resp_flag==1],label="1",shade=True)

sns.kdeplot(df.age[df.resp_flag==0],label="0",shade=True)

plt.xlim([60,90])

plt.xlabel("Age")

plt.ylabel("Density")
#性别比例

plt.figure(figsize=(3,5))  

sns.countplot(x="GEND",data=df)

plt.show()
#学历比例

plt.figure(figsize=(10,3))

sns.countplot(y="c210mys",data=df)

plt.show()
#不同学历购买保险情况

sns.countplot(x="c210mys",hue="resp_flag",data=df)  #hue分类变量

plt.xlabel("学历")

plt.ylabel("购买数量")
#县级大小购买保险情况

sns.countplot(x="N2NCY",hue="resp_flag",data=df)

plt.xlabel("municipal")

plt.ylabel("count")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 

plt.rcParams['axes.unicode_minus'] = False 
#处理空值

NA=df.isnull().sum().reset_index()

NA.columns=["Var","NA_count"]

NA=NA[NA.NA_count>0].reset_index(drop=True)

NA.NA_count/df.shape[0]
temp=[]

for i in NA.Var:

    temp.append(df[i].dtypes)

NA["数据类型"]=temp  
#填充空值

for i in NA[NA.Var!="age"].Var:

    df[i].fillna(df[i].mode()[0],inplace=True)

df.age.fillna(df.age.mean(),inplace=True)
del df["KBM_INDV_ID"]  
#变量编码

from sklearn.preprocessing import OrdinalEncoder

for i in df.columns:

    if df[i].dtype=="object":

        df[i]=OrdinalEncoder().fit_transform(df[[i]])
from sklearn.model_selection import train_test_split

X=df.iloc[:,1:]

y=df["resp_flag"]

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=420)
from sklearn import tree

clf=tree.DecisionTreeClassifier()

clf.fit(Xtrain,ytrain)

clf.score(Xtest,ytest)
max_depth=range(2,10)
from sklearn.model_selection import cross_val_score
score=[]

for i in max_depth:

    clf=tree.DecisionTreeClassifier(max_depth=i)

    result=cross_val_score(clf,Xtrain,ytrain,cv=5)

    score.append(result.mean())

plt.plot(max_depth,score)

clf=tree.DecisionTreeClassifier(max_depth=7,min_samples_leaf=500)

clf.fit(Xtrain,ytrain)

clf.score(Xtest,ytest)