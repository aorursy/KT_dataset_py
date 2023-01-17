import pandas as pd

import numpy as np

import matplotlib as plt

%matplotlib inline

import seaborn as sns
#Fetch the csv file

df = pd.read_csv("../input/google-analytics/Google Analytics.csv")
#To Display the rows and columns

pd.set_option('display.max_rows',5000, 'display.max_columns',100)
df.head()
#Check the size of the dataframe

df.shape
#To Check the Null Values

df.isnull().sum()
#Removing the Null Column with maximum Nan Values 

df.drop(['sessionQualityDim','referralPath'],axis=1,inplace=True)
#Checking the dataframe

df.head()
#Droping single Nan value in page view column

df = df.dropna(how='any',axis=0)
df.shape
#checking the percentage of Nan values

(df.isnull().sum() * 100/ len(df))
#Information about the columns

df.info()
#Checking Unique value in column with their count.

print(df['medium'].nunique(),df['medium'].unique())

print('=========================================================')

print(df['browser'].nunique(),df['browser'].unique())

print('=========================================================')

print(df['operatingSystem'].nunique(),df['operatingSystem'].unique())

print('=========================================================')

print(df['isMobile'].nunique(),df['isMobile'].unique())
#Repacing '(none)' value of column medium with 'Unknown'

df['medium']=df['medium'].replace(to_replace ="(none)", 

                 value ="Unknown")
#Checking The Numberof Transaction with respect to browser

sns.countplot(y ="browser",data=df,hue="NumberOfTransactions")
#Checking if the user are using Mobile Browser or not

sns.countplot(y ="browser",data=df,hue="isMobile")
#Customer LastvisitedDays from which Medium

sns.countplot(x ="LastVisitedDays",data=df,hue="medium")
#Continent with highest Number of Transaction 

sns.barplot(x='continent',y='NumberOfTransactions',data=df)
df.head()
#Creating a new column 'Chances' of customer buying a product or not based on RFMScore

Chances = []

for i in df['RFMScore']:

    if i >= 11 and i<=15:

        Chances.append('Good Chance')

    elif i >= 9 and i<11:

        Chances.append('Possible Chance')

    else:

        Chances.append('Impossible Chance')

df['Chances'] = Chances
#Checking People with Chances of buying wrt their RFM Score

sns.barplot(x='Chances',y='RFMScore',data=df)
#Label Encoder for converting values in Numeric Form

from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['fullVisitorId'] = labelEncoder_X.fit_transform(df['fullVisitorId'])

df['medium'] = labelEncoder_X.fit_transform(df['medium'])

df['browser'] = labelEncoder_X.fit_transform(df['browser'])

df['operatingSystem'] = labelEncoder_X.fit_transform(df['operatingSystem'])

df['isMobile'] = labelEncoder_X.fit_transform(df['isMobile'])

df['continent'] = labelEncoder_X.fit_transform(df['continent'])

df['Chances'] = labelEncoder_X.fit_transform(df['Chances'])
df.head()
#Preparing for Training and Testing

X=df.drop('Chances',axis=1)

y=df['Chances']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# Standardization of the data

from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
#Applying Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc__pred = rfc.predict(X_test)
#Checking the correctness of data

from sklearn.metrics import confusion_matrix, accuracy_score

print(accuracy_score(y_test,rfc__pred))

print(confusion_matrix(y_test,rfc__pred))