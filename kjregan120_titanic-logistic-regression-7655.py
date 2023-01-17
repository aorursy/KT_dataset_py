#import the libraries & file(s)
import numpy as np
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt
%matplotlib inline

#Import the file
data = pd.read_csv('../input/train.csv')
#reading data contents.
#data.head()
#data.tail()

#function List
#data.isnull()
#data.info()
#data.drop('Cabin',axis=1,inplace=True)
#data.dropna(inplace=True) 
#pd.get_dummies()
#pd.get_dummies(data['Sex'],drop_first=True)
#pd.get_dummies(data['Embarked'], drop_first=True)
#pd.concat([data,sex,embark], axis=1)

plt.figure(figsize=(10,7))

##################################################################
#Exploring the data; single line commands
##################################################################

#sms.distplot(data['Age'].dropna(),kde=False,bins=25)

#Countplots
#sms.countplot(x='Survived', hue='Sex', data=data)
#sms.countplot(x='Survived', hue='Pclass', data=data)
#sms.countplot(x = "Sibsp", hue="sex", data=data)

#Jointplots
#sms.jointplot(x='Age', y='Fare', data=data, kind='reg')
#sms.jointplot(x='Age', y='Fare', data=data, kind='kde')

#sms.pairplot(data,)
#sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma')

#BoxPlots

sms.boxplot(x='Pclass', y='Age',data=data)

##################################################################
#Data Cleaning, Purifying, and Imputing
##################################################################
#Here we're going to impute the average age for each class using the boxplot above to determine the values for a function we will build below. 
#this script is credited to Jose Portilla and his training book, which I highly recommend Python for Datascience and Machine Learning BootCamp (linked above)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
#using the function above the ages, 37, 29 and 24 were imputed (i.e. replaced the null values in the titanic dataset)
data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)
#test that the function worked correctly. 
#drop rows that are missing values
#drop column Cabin
data.dropna(inplace=True)
data.drop('Cabin', axis=1, inplace=True)
sms.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma')
#data.info()

#################################################################
#converting data to usable format; Essentially we need to convert Categorical values Male & Female to (1 or 0) for computational purposes.
#################################################################
sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'], drop_first=True)
##adding the new binary results for sex and adding the departure points QS from the newly created variables above into the data dataframe.
data = pd.concat([data,sex,embark], axis=1)
data.head()
data.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True)
data.head()
X = data.drop('Survived', axis=1)
y = data['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.30)
from sklearn.linear_model import LogisticRegression
logisticmodel = LogisticRegression()
logisticmodel.fit(X_train,y_train)
predictions = logisticmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
X_train