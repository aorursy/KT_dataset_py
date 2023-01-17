import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score
from sklearn.linear_model import LogisticRegression

s=pd.read_csv('../input/train.csv')
s.head()
s.info()
#More null values are in Cabin and some are in Age
s['Survived'].value_counts()
# Total 342 Members are survived
s.isnull().sum()
# There are total  687, 177 null values are in Age, Cabin respectively

s['Pclass'].value_counts()
#just see which class are more survived. wheather money matter when titanic is shrinking
s[(s['Pclass']==3) & (s['Survived']==1)]['Pclass'].count()
s[(s['Pclass']==2) & (s['Survived']==1)]['Pclass'].count()
#It seems Yes
s[(s['Pclass']==1) & (s['Survived']==1)]['Pclass'].count()
# which Embarked people are more survived
s['Embarked'].value_counts()
s[(s['Embarked']=='S') & (s['Survived']==1)]['Pclass'].count()
s[(s['Embarked']=='C') & (s['Survived']==1)]['Pclass'].count()
s[(s['Embarked']=='Q') & (s['Survived']==1)]['Pclass'].count()
#handling na values
s['Age'].isnull().sum() #Total 177 values are null
s['Age'].hist().plot()
from scipy import stats
stats.mode(s['Age'])
#Just replacing null values with mode i.e 24

def Datacleaning(s):
    s["Age"]=s['Age'].fillna(24)
    # filling with 'S' as there are more no of peoples are survivied 
    s['Embarked']=s['Embarked'].fillna('S')
    #PID,Name,'Ticket' attributes could not help while titanic is shrinking
    # As we are keeping Pclass removing Cabin, Fare
    s=s.drop(['PassengerId','Name','Ticket','Cabin','Fare'],axis=1)
    s=pd.get_dummies(s)
    return s
    
    
Feature=s.drop('Survived',axis=1)
Feature=Datacleaning(Feature)
Target=s['Survived']

l=LogisticRegression().fit(Feature,Target)
#Predict with train data validate with test data
TEST=pd.read_csv('../input/test.csv')
passengerID=TEST['PassengerId']
TEST.head()
TEST=Datacleaning(TEST)
#Precdiction
ypred=l.predict(TEST)
submission_df=pd.DataFrame()
submission_df['PassengerID']=pd.Series(passengerID)
submission_df['Survived']=pd.Series(ypred)
submission_df.head()
logit_data=submission_df.to_csv('LogitSurvive.csv')
