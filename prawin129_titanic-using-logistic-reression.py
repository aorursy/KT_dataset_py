import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#import warnings

#warnings.filterwarnings('ignore')
titanic_train=pd.read_csv("../input/train.csv")

titanic_train.head(2)
print('Total Number of records in Titanic Test data----->'+ str(len(titanic_train)))
sns.set_style('darkgrid')

sns.countplot(x='Survived',data=titanic_train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=titanic_train)
sns.countplot(x='Survived',hue='Pclass',data=titanic_train)
sns.countplot(x='SibSp',hue='Survived',data=titanic_train)
sns.set_style('dark')

titanic_train['Age'].plot.hist()
titanic_train['Fare'].plot.hist(bins=20,figsize=(10,5))
titanic_train.info()
titanic_train.isnull()
titanic_train.isnull().sum()
plt.figure(figsize=(18,13)) # In order to view the chart in zoom for Embark column



sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap='viridis', cbar=False) 



# yticklabels = False --> Is used to disable the y-axis lables

# cbar = False ---------> Is used to disable the bar on right hand side
plt.figure(figsize=(10,8))

sns.boxplot(x='Pclass',y='Age',data=titanic_train)
titanic_train.head()
titanic_train.drop('Cabin', axis=1, inplace=True) # to the Cabin column
titanic_train.head(7)
def inpute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 36

        elif Pclass == 2:

            return 28

        else: return 25

    else: return Age

    

titanic_train['Age']=titanic_train[['Age','Pclass']].apply(inpute_age, axis=1)
titanic_train.dropna(inplace=True)
sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap='viridis', cbar=False) 



# yticklabels = False --> Is used to disable the y-axis lables

# cbar = False ---------> Is used to disable the bar on right hand side
titanic_train.isnull().sum()
titanic_train.info()
titanic_train.head(3)
sex = pd.get_dummies(titanic_train['Sex'],drop_first=True)

sex.head()
embarked = pd.get_dummies(titanic_train['Embarked'],drop_first=True)

embarked.head()
pclass = pd.get_dummies(titanic_train['Pclass'],drop_first=True)

pclass.head()
titanic_train = pd.concat([titanic_train,sex,embarked,pclass],axis = 1)
titanic_train.head()
titanic_train.drop(['Pclass','Name','Sex','Embarked','PassengerId','Ticket'],axis=1,inplace=True)
titanic_train.head()
X = titanic_train.drop('Survived', axis=1)



y = titanic_train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
from sklearn.linear_model import LogisticRegression



log = LogisticRegression(solver='liblinear')

log.fit(X, y)
y_pred = log.predict(X_test) # should be y_pred
from sklearn import metrics

print('The Accuracy of Logistic Regression for Titanic Train Data ---->', metrics.accuracy_score(y_test,y_pred)*100)
#Read in the test data

titanic_test = pd.read_csv('../input/test.csv')

titanic_test.head(3)
#Clean the test data the same way we did the training data



titanic_test.drop('Cabin', axis=1, inplace=True) # to the Cabin column



titanic_test['Age']=titanic_test[['Age','Pclass']].apply(inpute_age, axis=1)



titanic_test.dropna(inplace=True)



sex = pd.get_dummies(titanic_test['Sex'],drop_first=True)



embarked = pd.get_dummies(titanic_test['Embarked'],drop_first=True)



pclass = pd.get_dummies(titanic_test['Pclass'],drop_first=True)



titanic_test = pd.concat([titanic_test,sex,embarked,pclass],axis = 1)



passenger_ids = titanic_test['PassengerId']



titanic_test.drop(['Pclass','Name','Sex','Embarked','PassengerId','Ticket'],axis=1,inplace=True)



titanic_test.tail()


predictions = log.predict(titanic_test)
submission = pd.DataFrame({

        "PassengerId": passenger_ids,

        "Survived": predictions    

    })



submission.to_csv('titanic.csv', index=False)