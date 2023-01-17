# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Iris.csv')
df.head()
df.tail()
df.info()
df.set_index('Id',inplace=True)
df.head(1)
df.describe()
qd=pd.DataFrame((df.describe().loc['75%']-df.describe().loc['25%'])/(df.describe().loc['75%']+df.describe().loc['25%']),columns=['cofficient of quartile deviation'])
qd
sns.pairplot(df,hue='Species',size=2.6)
sns.lmplot(data=df,x='SepalLengthCm',y='SepalWidthCm',hue='Species',size=10,fit_reg=False )
sns.lmplot(data=df,x='PetalLengthCm',y='PetalWidthCm',hue='Species',size=10,fit_reg=False ,logistic=True)
fig,axis=plt.subplots(nrows=2,ncols=2,figsize=(18,9))
sns.stripplot(y='Species',x='PetalWidthCm',data=df,ax=axis[0,1])
sns.stripplot(y='Species',x='PetalLengthCm',data=df,ax=axis[1,1])
sns.stripplot(y='Species',x='SepalWidthCm',data=df,ax=axis[0,0])
sns.stripplot(y='Species',x='SepalLengthCm',data=df,ax=axis[1,0])
plt.show()
fig,ax=plt.subplots(nrows=1 ,ncols=4)
sns.distplot(df['SepalLengthCm'],ax=ax[0])
sns.distplot(df['SepalWidthCm'],ax=ax[1])
sns.distplot(df['PetalLengthCm'],ax=ax[2])
sns.distplot(df['PetalWidthCm'],ax=ax[3])
fig.set_figwidth(30)
fig,ax=plt.subplots(nrows=1 ,ncols=4)

sns.boxplot(data=df,y='SepalLengthCm',x='Species',ax=ax[0])
sns.boxplot(data=df,y='SepalWidthCm',x='Species',ax=ax[1])
sns.boxplot(data=df,y='PetalLengthCm',x='Species',ax=ax[2])
sns.boxplot(data=df,y='PetalWidthCm',x='Species',ax=ax[3])
fig.set_figwidth(30)
fig.set_figheight(10)
fig=plt.figure(figsize=(20,7))
df.iloc[:,0].plot(label='Sepal Length')
df.iloc[:,1].plot(label='Sepal Width')
df.iloc[:,2].plot(label='Petal Length')
df.iloc[:,3].plot(label='Petal Width')
leg=plt.legend()
plt.show()

plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)
plt.title('correlation matrix')
plt.show()

x=df.iloc[:,:-1].values

pre=df.iloc[:,-1]
y=pre.replace(pre.unique(),np.arange(3))
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)

from sklearn.svm import  SVC
classifier=SVC(C=51,degree=1,gamma=10,kernel='poly')
classifier.fit(xtrain,ytrain)

from sklearn.metrics import classification_report
print(classification_report(ytest,classifier.predict(xtest)))
from sklearn.model_selection import cross_val_score
ca=cross_val_score(classifier,xtrain,ytrain,scoring='accuracy',cv=10)

ca
print(str(ca.mean()*100)+'% accuracy')#accuray after 10 cross validations

ca.std()*100#4%variance/bias on the set hence accuracy can be varied 4% in general
from sklearn.model_selection import GridSearchCV
params=[
    {
        'C':[51,0.1,100,1,10,80],'kernel':['rbf'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
    {
        'C':[51,0.1,100,1,10,80],'kernel':['poly'],'degree':[1,2,3,4],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
    {
        'C':[51,0.1,100,1,10,80],'kernel':['sigmoid'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
     {
        'C':[51,0.1,100,1,10,80],'kernel':['linear'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    }


]
gc=GridSearchCV(classifier,param_grid=params,cv=10,scoring='accuracy')
gc.fit(xtrain,ytrain)
gc.best_params_
gc.best_score_
ypred=classifier.predict(xtest)

unique=df['Species'].unique()
u=pd.Series(ypred,name='flowers').apply(lambda x:unique[x])
u.head(10)
ytest.values==ypred
plt.figure(figsize=(15,8))
sns.heatmap(pd.crosstab(ypred,ytest),annot=True,cmap='coolwarm')
