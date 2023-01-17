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
train_df  = pd.read_csv("/kaggle/input/titanic/train.csv")

print ("*"*10, "Dataset information", "*"*10)

print (train_df.info())
print ("*"*10, "First 5 Train File Rows", "*"*10)

train_df.head(5)
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

print(test_df.info())
train_df.drop(['Name', 'Ticket','Cabin'], inplace= True, axis = 1)

test_df.drop(['Name', 'Ticket', 'Cabin'], inplace= True, axis = 1)
print(train_df.info(), test_df.info())
train_df.isnull().sum()
test_df.isnull().sum()
train_df.Embarked.value_counts()
#Embarked null fix

data = [train_df, test_df]



for dataset in data:

    dataset.Embarked = dataset.Embarked.fillna('S')
train_df.Fare.value_counts()
train_df.Age.value_counts()
#Age, Fare null fix

data = [train_df, test_df]



for dataset in data:

    dataset.Fare = dataset.Fare.fillna(dataset.Fare.mean())

    dataset.Age = dataset.Age.fillna(dataset.Age.mean())
train_df.isnull().sum()
print(train_df.info(), test_df.info())
train_df.Age.value_counts()
train_df[['Age', 'Survived']].groupby(['Age'], as_index = False).mean().sort_values(by = "Survived", ascending = False)
tempFare = train_df.Fare

tempFare = pd.qcut(tempFare, 5)

tempFare.value_counts()
data = [train_df, test_df]



for dataset in data:

    dataset.loc[(dataset['Fare'] <= 7.854), 'Fare'] = 0

    dataset.loc[(dataset['Fare'] >= 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] >= 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] >= 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] >= 39.688), 'Fare'] = 4

    dataset.Fare = dataset['Fare'].astype(int)

    

train_df.Fare.value_counts()
tempAge = train_df.Age

tempAge = pd.qcut(tempAge, 5)

tempAge.value_counts()
data = [train_df, test_df]



for dataset in data:

    dataset.loc[(dataset['Age'] <= 20.0), 'Age'] = 0

    dataset.loc[(dataset['Age'] >=20.0) & (dataset['Age'] <= 28.0), 'Age'] = 1

    dataset.loc[(dataset['Age'] >= 28.0) & (dataset['Age'] <= 29.699), 'Age'] = 2

    dataset.loc[(dataset['Age'] >= 29.699) & (dataset['Age'] <= 38.0 ), 'Age'] = 3

    dataset.loc[(dataset['Age'] >= 38.0), 'Age'] = 4

    dataset.Fare = dataset['Age'].astype(int)

    

train_df.Age.value_counts()
print(train_df.info(), test_df.info())
print(train_df['Sex'].value_counts())

print(train_df['Embarked'].value_counts())
genderMap = {"male": 0, "female": 1}

embarkedMap = {"S": 0, "C": 1, "Q":2}



data = [train_df, test_df] 



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genderMap)

    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)
print(train_df.info(), test_df.info())
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)

Y_train = train_df['Survived']



X_test = test_df.drop("PassengerId", axis=1)
#1. Logistic Regression



from sklearn.linear_model import LogisticRegression



clf_lr = LogisticRegression(random_state = 0) 

clf_lr.fit(X_train, Y_train) 



acc_logistic = round(clf_lr.score(X_train, Y_train)*100, 2)



print (acc_logistic)
#2. SVM

from sklearn.svm import SVC



clf_svm = SVC() 

clf_svm.fit(X_train, Y_train) 



acc_svm = round(clf_svm.score(X_train, Y_train)*100, 2)



print (acc_svm)
#3. Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



clf_gnb = GaussianNB()

clf_gnb.fit(X_train, Y_train) 



acc_gnb = round(clf_gnb.score(X_train, Y_train)*100, 2)



print (acc_gnb)
#4. Ridge Classifier

from sklearn.linear_model import RidgeClassifier



clf_rc = RidgeClassifier()

clf_rc.fit(X_train, Y_train) 



acc_rc = round(clf_rc.score(X_train, Y_train)*100, 2)



print (acc_rc)
#5. K Nearest Neighbours Classifier (KNN)

from sklearn.neighbors import KNeighborsClassifier



clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, Y_train)



acc_knn = round(clf_knn.score(X_train, Y_train)*100, 2)



print (acc_knn)
### Best score submission to the leaderboard
Y_pred  = clf_knn.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")