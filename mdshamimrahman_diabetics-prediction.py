
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv("../input/diabetes-dataset/diabetes2.csv")
df.head(5)
df.shape

df.info()
df.isnull().sum()
df.describe()
df.corr()
import seaborn as sns
sns.countplot(df['Outcome'],palette=['#137909','#ff0707'])
df['Outcome'].value_counts()
sns.countplot(x='Age',hue='Outcome',data=df,palette=['#137909','#ff0707'])
x=df.drop('Outcome',axis=1)
y=df['Outcome']
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20)
from sklearn.ensemble import ExtraTreesClassifier
et= ExtraTreesClassifier()
et.fit(x,y)
et.feature_importances_
top=pd.Series(et.feature_importances_,index=x.columns)
result=top.nlargest(10)
result.plot(kind='bar')
result
x=df.drop(['SkinThickness','Insulin'],axis=1)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)