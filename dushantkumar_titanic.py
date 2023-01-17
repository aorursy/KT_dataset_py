# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
train_data= pd.read_csv("../input/titanic/train.csv").drop(['Name','Ticket','Cabin'],axis=1)

test_data = pd.read_csv("../input/titanic/test.csv").drop(['Name','Ticket','Embarked','Cabin'],axis=1)
import pandas_profiling 

train_data.profile_report()
train_data.shape
train_data.columns
train_data.head()
#finding the total number unquie values in the data set 

train_data.nunique()
#Getting some info from the data

train_data.describe()
#preparing the data- Deleting the column we don't need and fillinh up the missing value



train_data.isnull().sum()



#from here we can see that age and cabin column has missing values
train_data.isnull().sum()
train_data.describe(include=['O'])
train_data['Survived'].value_counts()
#checking how many people survived by graph

sns.countplot( train_data['Survived'])
s=sns.pairplot(train_data)
cols= ["Sex","Pclass","SibSp","Parch", "Embarked","Survived"]

n_rows= 2

n_col= 3

fig,axs= plt.subplots(2,3,figsize=(9,6))



for r in range(0,n_rows):

    for c in range(0,n_col):

        i= r*n_col + c

        ax=axs[r][c]

        sns.countplot(train_data[cols[i]],hue= train_data['Survived'],ax=ax)

        

        
train_data.isnull().sum()
train_data.head()
#look at survival rate of sex

train_data.groupby('Sex')['Survived'].mean()
#look at survival rate by sex and class

train_data.pivot_table("Survived",index='Sex',columns='Pclass')
#look at survival rate by sex and class visulize



train_data.pivot_table("Survived",index='Sex',columns='Pclass').plot()
train_data.columns
sns.barplot(x="Pclass",y="Fare",data=train_data)
#plot price fare in each class



sns.scatterplot(x="Fare", y="Pclass", data=train_data)
train_data.isnull().sum()
train_data.head()
train_data["Age"]= train_data['Age'].astype(float)

train_data['Age']

train_data.dtypes
train_data.isnull().sum()
new_data= train_data.copy()
new_data
new_data['Age']=new_data['Age'].replace(np.NaN,new_data['Age'].mean())

new_data['Embarked']= new_data['Embarked'].fillna(method='bfill',inplace=True)
new_data.isnull().sum()
new_data.dtypes
#test Data



test_data['Age']=test_data['Age'].replace(np.NaN,test_data['Age'].mean())

test_data['Fare']=test_data['Fare'].replace(np.NaN,test_data['Fare'].mean())
test_data.head()
test_data.isnull().sum()
#Label endoder 

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

test_data['Sex']= le.fit_transform(test_data['Sex'])







X = new_data.drop(['Survived','PassengerId','Embarked'],axis=1)

X['Sex'] = le.fit_transform(X['Sex'])

Y = new_data['Survived']
X.dtypes
#spliting the data

from sklearn.model_selection import train_test_split



#predictors = X.drop(['Survived', 'PassengerId','Embarked'], axis=1)

#target = Y["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)
 #Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_svc)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gaussian)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_test)

acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_knn)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes','Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_decisiontree]})

models.sort_values(by='Score', ascending=False)
#Submisiion

#Highest accuracy rate is in Random Forest



id= test_data['PassengerId']

predictions = randomforest.predict(test_data.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions })

output.to_csv('submission.csv', index=False)