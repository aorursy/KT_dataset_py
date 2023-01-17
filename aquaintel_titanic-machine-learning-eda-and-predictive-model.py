# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf  #Interactive plots
cf.go_offline()
%matplotlib inline

# Predictive model 
# Importing required ML packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.describe(include=['O']) # categorical columns
train_df.describe() # all numeric columns
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_df,palette='winter')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_df,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass', data=train_df, palette='winter')
sns.set_style('whitegrid')
sns.distplot(train_df['Age'].dropna(),kde=False,color='darkblue',bins=30)
sns.countplot(x='SibSp',data=train_df, palette='winter')
train_df['Fare'].iplot(kind='hist',bins=50,color='lightblue',xTitle='Number of Tickets', yTitle='Fare Cost ($)')
print('Highest Fare:',round(train_df['Fare'].max(),0))
print('Lowest Fare:',round(train_df['Fare'].min(),0))
print('Average Fare:',round(train_df['Fare'].mean(),0))
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='GnBu')
round(train_df.isnull().sum()*100/891,1) #Percentage of null values
round(test_df.isnull().sum()*100/891,1) #Percentage of null values
# Checking Age as a function of Passenger Class
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_df,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        return round(train_df[train_df['Pclass']==Pclass]['Age'].mean(),0)
    else:
        return Age
# Impute Age based on mean age per Passenger class
train_df['Age'] = train_df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='GnBu')
train_df.drop('Cabin',axis=1,inplace=True)
train_df.head()
train_df.dropna(inplace=True)
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='GnBu')
# Impute Age based on mean age per Passenger class
test_df['Age'] = test_df[['Age','Pclass']].apply(impute_age,axis=1)

# Drop Cabin data because we dropped in the training data set
test_df.drop('Cabin',axis=1,inplace=True)

#Check missing values in test data set using a heat map
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='GnBu')
sex = pd.get_dummies(train_df['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train_df['Embarked'],drop_first=True)
embark.head()
train_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train_df = pd.concat([train_df,sex,embark],axis=1)
train_df.head()
# Convert categorical features in test data set
sex = []
embark = []
sex = pd.get_dummies(test_df['Sex'],drop_first=True)
embark = pd.get_dummies(test_df['Embarked'],drop_first=True)
test_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test_df = pd.concat([test_df,sex,embark],axis=1)
# Complete the single missing Fare feature in test dataset using the median of the Fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df.head()
test_df.head()
logmodel = LogisticRegression()

# Train and Test Data
X_train = train_df.drop(['Survived','PassengerId'], axis=1)
y_train = train_df['Survived']
X_test  = test_df.drop('PassengerId', axis=1).copy()
# training the model
logmodel.fit(X_train,y_train)
# Predicting survival for test passengers
predictions_01 = logmodel.predict(X_test)  
predictions = predictions_01
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)