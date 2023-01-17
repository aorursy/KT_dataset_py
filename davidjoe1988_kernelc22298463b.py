import numpy as np

import pandas as pd



mydata=pd.read_csv("../input/titanic_train.csv")



# Any results you write to the current directory are saved as output.
#top 5 will be reflects and use .tail to lower data and .loc with index numbers for consideration.(basics)

mydata.head()
#info gives the missing values in our data

mydata.info()
#if required to know the True values

mydata.isnull()
import matplotlib.pyplot as plt

import seaborn as sb



#null values present in our data using data visuals

sb.heatmap(mydata.isnull())
#percentage of persons survived in disaster

sb.countplot(x='Survived',data=mydata)
#defining in sex

sb.countplot(x='Survived',hue='Sex',data=mydata)
#classifying in Pclass(L,M,H)

sb.countplot(x='Survived',hue='Pclass',data=mydata)
#defining pclass on reference with age for suvival.

plt.figure(figsize=(12,7))

sb.boxplot(x='Pclass',y='Age',data=mydata,palette='autumn')
#imputing age on null values

def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
mydata['Age']=mydata[['Age','Pclass']].apply(impute_age,axis=1)
#age null values filled using decision making

sb.heatmap(mydata.isnull())
#dropping cabin 

mydata.drop('Cabin',axis=1,inplace=True)
mydata.info()
#Dropping variables which is not required for study

sex=pd.get_dummies(mydata['Sex'],drop_first=True)

embark=pd.get_dummies(mydata['Embarked'],drop_first=True)
mydata.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#concantenate to remove the dummies

mydata=pd.concat([mydata,sex,embark],axis=1)
mydata.head()
#train_test to prediction

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(mydata.drop('Survived',axis=1),mydata['Survived'],test_size=0.20,random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression(solver='liblinear')

logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)

print(predictions[5])

y_test.head(5)

#package warning
logmodel.fit(x_train,y_train)
logmodel.score(x_test,y_test)
predictions