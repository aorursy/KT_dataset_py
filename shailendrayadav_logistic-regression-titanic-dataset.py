import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/titanic.csv")

df.head()
df.isnull().sum()
#Lets do some initial analysis on the data

sns.countplot(x=df.Sex) # male and Female ratio at Ship
#Lets see the Age group of people with missing data ,Total missing value is 177.

df[df.Age.isnull()]
#lets see the  age with  High frequency in data set and mean value as well

df.Age.mode(),df.Age.mean()
#Lets fill the missing value as mean value of the people as 29

df.Age.fillna(value=29,inplace=True)

df.Age.isnull().sum() # all values are filled now 
#lets encode Sex as its a categorical value and needs to be encoded for machine learnin.

from sklearn.preprocessing import LabelEncoder

dfe=df.copy()

le=LabelEncoder()

Sex1 =le.fit_transform(dfe.Sex)

Sex1
dfe["Sex1"]=Sex1

dfe.head()

dfe.columns
#Machine learning process started

Y=dfe.Survived

my_col=['PassengerId', 'Survived', 'Name', 'Ticket',"Sex","SibSp","Parch", 'Cabin', 'Embarked']

dfe.drop(columns=my_col,axis="columns",inplace=True)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,Y_train)
len(X_test)
lr.predict(X_test)
lr.score(X_test,Y_test)