# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



#Plotting Libraries

import matplotlib.pyplot as plt

import seaborn as sns

# To filter warnings shown 

import warnings

warnings.filterwarnings('ignore') 



#Have inliine graphs i.e. within notebook

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing the Training Data

data_train=pd.read_csv("../input/train.csv")

data_train.head(3)
data_train.drop(['Name', 'Ticket', 'Cabin'],axis=1,inplace=True)
#Importing the Testing Data

data_test=pd.read_csv("../input/test.csv")
data_test.drop(['Name', 'Ticket', 'Cabin'],axis=1,inplace=True)
full_data = [data_train, data_test]

# data['Age']=data['Age'].apply(np.ceil)
print("Total no of train samples {}".format(data_train.shape[0]))
#Checking for Null - Values Age,Cabin and Embarked have Null Values

data_train.isnull().any()
def clean_age(dataFrmae):

    age_avg = dataFrmae['Age'].mean()

    age_std = dataFrmae['Age'].std()

    age_null_count = dataFrmae['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataFrmae['Age'][np.isnan(dataFrmae['Age'])] = age_null_random_list

    dataFrmae['Age'] = dataFrmae['Age'].astype(int)
def clean_embark(dataFrame):

    dataFrame['Embarked'] = dataFrame['Embarked'].fillna('S') #Replace empty values in Embark with 'S' 
# Mapping type of Sex

def clean_sexType(dataFrame):

    dataFrame['Sex'] = dataFrame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
def clean_fare(dataFrame):

    dataFrame['Fare'] = dataFrame['Fare'].fillna(dataFrame['Fare'].median())

# Now its time to remove Null values from Data

for dataset in full_data:

    clean_age(dataset)

    clean_embark(dataset)

    clean_fare(dataset)

data_train.isnull().any()
data_test.isnull().any()
data_train['Survived'].value_counts().plot(kind='bar')
lived=data_train[data_train['Survived']==1]['Sex'].value_counts()

died=data_train[data_train['Survived']==0]['Sex'].value_counts()

total_people=pd.DataFrame([lived,died],index=['Survived','Died'])
ax = plt.subplot()

ax.set_ylabel('No of Lives\n')

total_people.plot(kind='bar',figsize=(10,7),ax=ax)
data_groupBy_Class = data_train.groupby([ "Pclass",'Sex']);

data_groupBy_Class['Survived'].mean().plot(kind='bar',color = 'g')
data_groupBy_Class.mean()
figure = plt.figure(figsize=(15,8))

plt.hist([data_train[data_train['Survived']==1]['Fare'],data_train[data_train['Survived']==0]['Fare']], 

         color = ['g','r'],bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()

data_train.isnull().sum() #Count no of missing values
#Now its time to clean some Data

#map sex type of sex to 0 or 1 for better learningof Model 

clean_sexType(data_train)

clean_sexType(data_test)
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1

data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch'] + 1
data_train['Embarked'] = data_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_test['Embarked'] = data_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_train.head()
data_test.head()
plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_train.corr(),linewidths=.5,vmax=1.0, cmap=plt.cm.viridis, linecolor='white', annot=True)
from sklearn.model_selection import train_test_split



X = data_train.drop('Survived',axis=1)

y = data_train.Survived

X_train, X_test, Y_train, Y_test = train_test_split(X, y,  test_size=0.20, random_state=0)

# X_train = data_train.drop("Survived",axis=1)

# Y_train = data_train["Survived"]

# X_test  = data_test

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_test, Y_test)
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_test, Y_test)
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_test, Y_test)
#KNN-Classifier



knn = KNeighborsClassifier(n_neighbors = 5,weights='distance')

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_test, Y_test)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_test, Y_test)
# Using Random Forest Classifier

X_test  = data_test

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)
submission