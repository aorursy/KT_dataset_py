
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df=pd.read_csv("../input/startup-logistic-regression/50_Startups.csv")
df.head(5)
df.isnull().sum()
df.info()
df.shape
## One Hot Encoding 

df=pd.get_dummies(df,columns=['State'])
df
df.corr()
x=df.drop(['Profit'],axis=1)
x.head(1)
y=df['Profit']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
sb=SelectKBest(k=6)
sb.fit(x,y)
score=pd.DataFrame(sb.scores_,columns=["Score"])
score
colname=pd.DataFrame(x.columns)
final=pd.concat([score,colname],axis=1)
final
final.nlargest(3,'Score')
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)
