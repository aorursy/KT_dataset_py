# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# get data to dataframe variables

train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# data visualization

train_df

#inspecting train dataframe

train_df.shape

train_df.info()

train_df.isnull().sum()
#inspecting test dataframe

test_df.shape

test_df.info()

test_df.isnull().sum()
train_df.columns
# drop unnecessary columns: PassengerId, Name, Ticket, Cabin from train and test data

#Note: Survived is not present in the test data



train_df = train_df.drop(['PassengerId','Cabin','Ticket','Name'],axis=1)



test_PassengerId = test_df['PassengerId']

test_df = test_df.drop(['PassengerId','Cabin','Ticket','Name'],axis=1)

print(test_df.columns)

train_df.columns

#checking the null value in the train and test data

print(train_df.isnull().sum())

print(test_df.isnull().sum())
# replacing the null age by median

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Age'].fillna(test_df['Age'].median(), inplace = True)



print(train_df.isnull().sum())

print(test_df.isnull().sum())
train_df.Embarked.value_counts()
# Imputing the empty value with S in the embarked column

train_df.loc[train_df.Embarked.isnull(),['Embarked']] ='S'

print(train_df.isnull().sum())
train_df.Embarked.value_counts()
# replacing the null fare by mean

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)



print(test_df.isnull().sum())
#Inspecting the datatype of the data

print(train_df.dtypes)

print(test_df.dtypes)
# Looking at the data distribution

train_df.describe()
# Looking at the data distribution

test_df.describe()
# checking the distribution of the data

train_df.hist()

#Age

train_df['Age'].hist()

train_df['Age'].describe()
train_df.groupby('Sex')['Sex'].count()

sns.countplot('Embarked', data=train_df)

plt.show()
train_df.groupby('Pclass')['Pclass'].count()
def class_imput(x):

    if(x==1):

        return 'First'

    elif(x==2):

        return 'Second'

    else:

        return 'Third'

    

train_df.Pclass = train_df.Pclass.apply(class_imput)

test_df.Pclass = test_df.Pclass.apply(class_imput)

train_df.head()                 
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy_train = pd.get_dummies(train_df[['Pclass', 'Sex', 'Embarked']], drop_first=True)

dummy_test = pd.get_dummies(test_df[['Pclass', 'Sex', 'Embarked']], drop_first=True)

# Adding the results to the master dataframe

train_df = pd.concat([train_df, dummy_train], axis=1)

    

test_df = pd.concat([test_df, dummy_test], axis=1)
# drop the original columns

train_df = train_df.drop(['Pclass', 'Sex', 'Embarked'],axis=1)

test_df = test_df.drop(['Pclass', 'Sex', 'Embarked'],axis=1)
train_df.head()

test_df.head()
train_df.groupby('SibSp')['SibSp'].count()
train_df.groupby('Parch')['Parch'].count()
# combining the columns SibSp and Parch and making one column family.

train_df['Family'] = train_df['SibSp'] + train_df['Parch']

train_df[train_df['Family'] > 0]
train_df.groupby('Family')['Family'].count()
# drop unnecessary columns SibSp and Parch now

train_df = train_df.drop(['SibSp','Parch'], axis =1)
train_df.info()
# replacing non zero value by 1

def decode_family(x):

    if x == 0:

        return 0

    else:

        return 1

    





train_df['Family'] = train_df.Family.apply(decode_family)
train_df.groupby('Family')['Family'].count()
# Do same operation on test_df



# combining the columns SibSp and Parch and making one column family.

test_df['Family'] = test_df['SibSp'] + test_df['Parch']



# drop unnecessary columns SibSp and Parch now

test_df = test_df.drop(['SibSp','Parch'], axis =1)



test_df['Family'] = test_df.Family.apply(decode_family)
test_df.info()

test_df.shape

test_df.groupby('Family')['Family'].count()
train_df.head()
train_df.Survived.value_counts()
# separating dependent and independent variables

y_train = train_df['Survived']

X_train = train_df.drop('Survived',axis=1)



X_test = test_df
from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import RandomForestClassifier
# Test options and evaluation metric

seed = 7

scoring = 'accuracy'
# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Make predictions on validation dataset

lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)

predictions = lda.predict(X_test)



predictions