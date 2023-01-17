# Import all required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
titanic = pd.read_csv('../input/train.csv') # read the Train.csv for Titanic Data set
titanic.head()
titanic.info()
titanic['Pclass'].plot.hist()
titanic.plot.scatter(x='Age',y='Pclass')
#sns.pairplot(titanic)
titanic.isnull().sum() # Check for the Null Values
titanic['Pclass'].unique()
titanic[titanic['Pclass']==1]['Age'].mean()
titanic.groupby('Pclass')['Age'].mean()# checking for mean of Age based on Pclass
titanic[titanic['Pclass']==1& titanic['Age'].isnull()] # looking at data where Pclass == 1 and Age is Null
# Function to populate the Age based on Mean of Age in each Pclass
def Age_Fill(df):
    Age = df[0]
    Pclass = df[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38.23
        elif Pclass ==2:
            return 29.878
        else:
            return 25.14
    else:
        return Age
titanic['Age'] = titanic[['Age','Pclass']].apply(Age_Fill,axis=1)# filling Null Values for Age
titanic.isnull().sum()# all Null Values for Age are resolved.
titanic.drop('Cabin',axis=1,inplace=True) #  Cabin has mostly Null Values so can drop the Column 
titanic.isnull().sum()# Cabin is Dropped and only Embarked Column has 2 Null Values
titanic.dropna(inplace=True)# dropping 2 rows where Embarked is Null.
titanic.isnull().sum()# All Null Values are resolved now
titanic.count() # Total Number of Rows after Cleaning the Train Data.
sex= pd.get_dummies(titanic['Sex'],drop_first=True)# creating Dummy Variable for Sex Categorical Variable
embark = pd.get_dummies(titanic['Embarked']) # creating Dummy Variable for Embarked Categorical Variable
pclass = pd.get_dummies(titanic['Pclass']) # creating Dummy Variable for Pclass Categorical Variable
titanic = pd.concat([titanic,sex,embark,pclass],axis=1) # Merging the Dummy Variables in Titanic Data Set.
titanic.head() # Checking Head of Titanic Data Set
titanic.drop(['Sex','Ticket','Embarked','Name','Pclass'],inplace=True,axis=1) # Dropping Categorical Columns for which Dummy Variables are already created.
titanic.head() # Checking Head of DataSet after Dropping the Columns
titanic.drop(['PassengerId'],axis=1,inplace=True) # Dropping Passenger Id as well as it just like an Index Numbering.
titanic.head() # Final Data from Titanic DataSet
X_train = titanic.drop(['Survived'],axis=1)
y_train = titanic['Survived']
# Keeping on Survived Column in y_train and all other Features in X_train
from sklearn.linear_model import LogisticRegression # Importing Logistic Regression
logModel = LogisticRegression()
logModel.fit(X_train,y_train) # Fitting the Model on Training Data
titanic_test = pd.read_csv('../input/test.csv') # Importing the Test Data to create Test Data Set
titanic_test.head()# Checking the Head of Titanic Test Data Set
titanic_test.isnull().sum()# Checking the Null Values
titanic_test.groupby('Pclass')['Age'].mean() # Checking the Mean age for each Pclass
#Function to fill Null Values in Age for each Pclass
def Impute_Age(df):
    Age = df[0]
    Pclass = df[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 40.92
        elif Pclass ==2:
            return 28.78
        else:
            return 24.03
    else:
        return Age
titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(Impute_Age,axis=1)# Populating Null Values of Age
titanic_test.isnull().sum()# Checking the Reaming Null Values. Age is Fixed now.
titanic_test.head() # Checking head of Titanic_test Data Set
PassengerId = titanic_test['PassengerId'] # Getting the Passenger Id
type(PassengerId)
# Checking the mean of Fare for Pclass to be able to populate the missing Value of Fare appropriately
titanic_test.groupby('Pclass')['Fare'].mean()
titanic_test[titanic_test['Fare'].isnull()] # checking the record where Fare is NaN
titanic_test.plot.scatter('Pclass','Fare')
# Checking the Head of Titanic Test Data Set sorted by Ticket to see any possibility of 
# identifying the right Fare based on ticket.
titanic_test[titanic_test['Pclass'] == 3].sort_values('Ticket').head(10) 
#populating the Nan Fare as Mean of Pclass == 3, since it belongs to pClass = 3
titanic_test.loc[titanic_test['Fare'].isnull(),'Fare'] = titanic_test[titanic_test['Pclass']== 3]['Fare'].mean()
titanic_test[titanic_test['Pclass']== 3]['Fare'].mean()
titanic_test[titanic_test['Ticket']=='3701']# Populated the the Missing value for Fare
titanic_test.isnull().sum()# Checking the Null Values
titanic_test.drop('Cabin',axis=1,inplace=True)# Dropping Cabin from Data Set since its most of the Values are Null
titanic_test.head()
sex_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)# creating Dummy Variable for Sex Categorical Variable
embark_test=pd.get_dummies(titanic_test['Embarked'])# creating Dummy Variable for Embarked Categorical Variable
pclass_test = pd.get_dummies(titanic_test['Pclass'])# creating Dummy Variable for Pclass Categorical Variable
# Conactenating the dummy Variables to Test Data Set
titanic_test = pd.concat([titanic_test,sex_test,embark_test,pclass_test],axis=1)
titanic_test.head()# Checking the Head of Test Data Set after Merge.
titanic_test.drop(['Sex','Embarked','Ticket','Name','Pclass'],axis=1,inplace=True)# Dropping Categorical Columns
titanic_test.head()
X_test = titanic_test.drop(['PassengerId'],axis=1)# Creating X_test.
#y_test = titanic_test['Survived']

Predictions = logModel.predict(X_test)# Predicting on X_test
Predictions
# Putting the Predictions and PassengerId in the Output Csv
data1 = pd.Series(Predictions)
data2 = pd.concat([PassengerId,data1],axis=1)
data2.columns = ['PassengerId','Survived']
data2.set_index('PassengerId',inplace=True)
data2.to_csv('Titanic_Predictions.csv')

