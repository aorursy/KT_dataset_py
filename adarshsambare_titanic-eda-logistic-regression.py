# Let's Get Started by importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing

# Visulation libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Reading the data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# Checking the data
train.head()
# Null values are empty data points
# checking the data for Null values
train.isnull()
# we do have null values in some columns
# creating heat map based on null values on the data 
fig, ax = plt.subplots(figsize=(12,12)) # Increasing the fig size

sns.heatmap(train.isnull(),cmap='viridis')

# Cabin & Age have most missing values (Null values)
# and Embarked seem to have some missing values
# We Will deal with the missing values later
# Let's checked the survived & not survived population as per gender

# Count plot is the basic measure for such plots
sns.countplot(x='Survived',data=train, hue='Sex', palette= 'RdBu_r')
# Count Plot as per Passenger Class
sns.countplot(x='Survived',data= train, hue='Pclass')
# Age of People on the Titanic 
# simply checking the distribution plot
sns.distplot(train['Age'].dropna(),bins=30, kde =False)
# Checking the siblings columns
sns.countplot(x='SibSp',data = train)
# Seems like People with no siblings are more
# Fare Distribution
sns.distplot(train['Fare'].dropna(),kde=False)
# High people Fare lies between 0-50
# More inovative plots can we done with cufflinks
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins = 50)
# Mean and Median Imputation are the most commonly used imputation tech
# Rather than just Imputating Mean of Age
# we will find out Mean of Age as per Passenger class
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y="Age",data=train)
# Finding out the Mean as Per Passenger class
#train.groupby('Pclass')['Age'].median()
# as median takes care of outliers(both are same in our case)
train.groupby('Pclass')['Age'].mean()

# Class 1 Mean : 38
# Class 2 Mean : 30
# Class 3 Mean : 25
# Writing a function for changing null values as per median
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
# Applying above impute on age columns
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
# Checking if null are imputed or not
sum(train['Age'].isnull())
# Dropping the Cabin as too many null values
train.drop('Cabin', axis = 1, inplace = True)
# checking if Cabin column is dropped or not
train.columns
# we can either impute or drop the data row where Embarked is missing
train.dropna(inplace=True)
# checking the data for missing values
# creating heat map based on null values on the data 
fig, ax = plt.subplots(figsize=(12,12)) # Increasing the fig size

sns.heatmap(train.isnull(),cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
pclass = pd.get_dummies(train['Pclass'],drop_first = True)
# adding the dummies in the data frame
train = pd.concat([train,sex,embark,pclass],axis = 1)
train.head()
# Dropping the unnecessary columns
train.drop(['PassengerId','Pclass','Sex','Embarked','Name','Ticket'],axis = 1,inplace =True)
train.head()
# splitting the train data for cross validations
x = train.drop('Survived',axis = 1)
y= train['Survived']
# Splitting using sklearn
from sklearn.model_selection import train_test_split
# Actual splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)
# Model Building
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
# Fitting the Logistic Regression on train data 
logmodel.fit(x_train,y_train)
# Prediting on the test data using our model
pred = logmodel.predict(x_test)
# for classification We can directly have % from
from sklearn.metrics import classification_report
# Accuracy is given by
print(classification_report (y_test,pred))
# Predicting on actual test data
Actual_pred = logmodel.predict(test)
