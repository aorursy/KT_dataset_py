#Data Analysis libraries

import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns







# Model development libraries

from sklearn.linear_model import BayesianRidge

from fancyimpute import IterativeImputer as MICE

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



%matplotlib inline
titanic = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
combine = [titanic, test]
titanic.head()
test.head()
titanic.info()



print('-'*45)



test.info()
pp.ProfileReport(titanic)
pp.ProfileReport(test)
#Box plot

sns.boxplot(data = titanic)
#Box plot

sns.boxplot(data = test)
# first lets delete unnecessary columns



titanic.drop(['PassengerId', 'Ticket', 'Cabin','Name'],axis =1 , inplace = True)



test.drop(['Ticket', 'Cabin','Name'],axis = 1 , inplace = True)
# Let's create a variable that shows the family one has on the ship



titanic['Family']= titanic['SibSp'] + titanic['Parch']



test['Family']= test['SibSp'] + test['Parch']
# Check whether family has importance on whether someone survived or not



sns.countplot('Family', hue = 'Survived', data = titanic)
for dataset in combine:

    dataset.loc[dataset['Family'] == 0, 'Family']=0

    dataset.loc[(dataset ['Family']>0 ) & (dataset['Family']<= 3) , 'Family'] = 1

    dataset.loc[dataset['Family'] > 3 , 'Family']=2
test.head()
titanic.head()
#Delete the SibSp and Parch

for dataset in combine:

    dataset.drop(['SibSp', 'Parch'], axis = 1 , inplace = True)
# Let's now work on Sex and Embarked



#For sex



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'male':1 , 'female':0})

    

# For Embarked



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0 ,'C':1 , 'Q':2})
#Now let's deal with Fare.Three classes. Fare for third class, fare for second class and fare for first class



for dataset in combine:

    dataset.loc[dataset['Fare']<40 ,'Fare']=3

    dataset.loc[(dataset['Fare']>40) & (dataset['Fare']<150) , 'Fare']= 2

    dataset.loc[dataset['Fare']>150 ,'Fare']=1

titanic.info()



print('-'*45)



test.info()
#Let's impute the missing values in fare embarked and age



#Embarked

titanic['Embarked'].fillna(0 , inplace = True)



#Fare 

test['Fare'].fillna(3)



#Tells us the position of the null value

np.nonzero(pd.isnull(test['Fare']))
# From this we can  know how to impute the data

test.loc[152,:]
titanic.info()



print('-'*45)



test.info()
# Now let's impute the missing data in age

# Let's impute the missing values using the iterative imputer



#Let's start with the training set

cols = list(titanic)

titanic_new = MICE(sample_posterior = True).fit_transform(titanic)

titanic_new = pd.DataFrame(titanic_new)

titanic_new.columns = cols
titanic_new.info()
titanic_new['Pclass'] =titanic_new['Pclass'].astype(np.int64)

titanic_new['Age'] = titanic_new['Age'].astype(np.int64)

titanic_new['Family'] = titanic_new['Family'].astype(np.int64)

titanic_new['Sex'] = titanic_new['Sex'].astype(np.int64)

titanic_new['Embarked'] = titanic_new['Embarked'].astype(np.int64)

titanic_new['Survived'] = titanic_new['Survived'].astype(np.int64)

#Convert Fare to float

titanic_new['Fare'] = titanic_new['Fare'].astype(np.int64)

    

    

# For the test data

cols = list(test)

test_new = MICE(sample_posterior = True).fit_transform(test)

test_new = pd.DataFrame(test_new)

test_new.columns = cols
test_new['Pclass'] =test_new['Pclass'].astype(np.int64)

test_new['Age'] = test_new['Age'].astype(np.int64)

test_new['Family'] = test_new['Family'].astype(np.int64)

test_new['Sex'] = test_new['Sex'].astype(np.int64)

test_new['Embarked'] = test_new['Embarked'].astype(np.int64)

test_new['PassengerId'] = test_new['PassengerId'].astype(np.int64)

#Convert Fare to float

test_new['Fare'] = test_new['Fare'].astype(np.int64)
titanic_new.info()



print('-'*45)



test_new.info()
# Now let's check for  outliers

sns.boxplot(data = titanic_new)
sns.boxplot(data = test_new)
test_new.Fare.value_counts()
titanic_new.Age.value_counts()
test_new.Age.value_counts()
# Make new list cotaining the modified datasets

combine_new = [titanic_new,test_new]



# Let's place some categories in the age



for dataset in combine_new:

    dataset.loc[(dataset['Age']>=0) & (dataset['Age']<=14) , 'Age']= 0

    dataset.loc[(dataset['Age']>=15) & (dataset['Age']<=24) , 'Age']= 1

    dataset.loc[(dataset['Age']>=25) & (dataset['Age']<=64) , 'Age']= 2

    dataset.loc[dataset['Age']<0 , 'Age']= 2

    dataset.loc[dataset['Age']>=65 ,'Age']=3



titanic_new.Age.value_counts()
test_new.Age.value_counts()
titanic_new.drop_duplicates()



test_new.drop_duplicates()
titanic_new.head()
test_new.head()
test_new['Age'].astype(int)
titanic_new['Age'].astype(int)
# Split the Data

x = titanic_new.drop(['Survived','Fare'], axis =1)

y = titanic_new['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 7)
# Logistic regression



lr = LogisticRegression()



#Train the model

lr.fit(x_train,y_train)



#predict

lr_pred = lr.predict(x_test)



#Classification report

print(classification_report(y_test,lr_pred))
# Decision Tree

dt = DecisionTreeClassifier()



#Train

dt.fit(x_train,y_train)



#Predict

dt_pred = dt.predict(x_test)



#Classification report

print(classification_report(y_test,dt_pred))
#random Forest



rf = RandomForestClassifier(n_estimators = 250)



#Fit

rf.fit(x_train,y_train)



#Predict

rf_pred = rf.predict(x_test)



#classification report



print(classification_report(y_test,rf_pred))
# fit on the test dataset



#Predict

x= test_new.drop(['PassengerId', 'Fare'],axis = 1)

test_new['Survived'] = rf.predict(x)
test_new.head()
submission = test_new.drop(['Pclass','Sex','Age','Fare', 'Embarked','Family'],axis=1)
submission.head()
submission = submission.to_csv('submit.csv',index=False)
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")