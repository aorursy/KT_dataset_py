
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('../input/cardio-data-set/cardio_train.csv')
df.head(5)
## Separate the columns

df=pd.read_csv('../input/cardio-data-set/cardio_train.csv',sep=';')
df.head(786)
#Data set Shape

df.shape
#Data Set info 
df.info()
#Check Null value

df.isnull().sum()
## To see correaltion

df.corr()
##Drop Id ..its not usefull

df=df.drop('id',axis=1)
df.head(5)
## How many people have cardio vascular Disease

df['cardio'].value_counts()
## Now visualize data who have cardio diseases and who have safe 

sns.countplot(df['cardio'],palette=['#137909','#ff0707']);

## we see that its closely 50/50
## Cardio Vascular diseases according to gender

sns.countplot(x='gender',hue='cardio',data=df,palette=['#137909','#ff0707'],edgecolor=sns.color_palette('dark',n_colors=1))
sns.countplot(x='age',hue='cardio',data=df,palette=['#137909','#ff0707'],edgecolor=sns.color_palette('dark',n_colors=1))
### The data is so noise here bcz age is given in days
##Now age convert in year

df['age']=(df['age']/365).round(0)


df.age
## Ok fine now cardio vascular detect according to age (don't worry it is converted in years now)

sns.countplot(x='age',hue='cardio',data=df,palette=['#137909','#ff0707'],edgecolor=sns.color_palette('dark',n_colors=1))
## We See diseases increased when age is increased 
## Now According to weight

sns.countplot(x='weight',hue='cardio',data=df,palette=['#137909','#ff0707'],edgecolor=sns.color_palette('dark',n_colors=1))
##Splited Features and Label

x=df.drop(['cardio'],axis=1)
x.head(5)
y=df['cardio']
#Take data for test size
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20)
##Use Random Forest bcz data set is big but we also see others classification algo

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf
rf.fit(xtrain,ytrain)
rf.score(xtest,ytest)
##Predict the value to check 

rf.predict(xtest)
ytest
##Now we check using Decission Tree

from sklearn.tree import  DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dt.score(xtest,ytest)
#Now using SVm

from sklearn.svm import SVC

sv=SVC()
sv.fit(xtrain,ytrain)
sv.score(xtest,ytest)
sv.predict(xtest)
ytest
##Logistic Regression

from sklearn.linear_model import LogisticRegression

lg= LogisticRegression()
lg.fit(xtrain,ytrain)
lg.score(xtest,ytest)
lg.predict_proba(xtest)
lg.predict(xtest)
##Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(xtrain,ytrain)
nb.score(xtest,ytest)
