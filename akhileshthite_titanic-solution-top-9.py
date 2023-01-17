import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
import pandas_profiling as pp
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
pp.ProfileReport(train_df, title = 'Pandas Profiling report of "Train" set', html = {'style':{'full_width': True}})
pp.ProfileReport(test_df, title = 'Pandas Profiling report of "Test" set', html = {'style':{'full_width': True}})
train_df=train_df.drop("PassengerId",axis=1)
train_df=train_df.drop("Name",axis=1)
train_df=train_df.drop("Ticket",axis=1)
train_df=train_df.drop("Cabin",axis=1)
train_df.head()
test_df=test_df.drop("Name",axis=1)
test_df=test_df.drop("Ticket",axis=1)
test_df=test_df.drop("Cabin",axis=1)
test_df.head()
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()
common_value = 'S'
train_df["Embarked"] = train_df["Embarked"].fillna(common_value)
test_df = test_df.fillna(test_df['Fare'].mean())
train_df.info()
test_df.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df["Sex"]= le.fit_transform(train_df["Sex"])
print(train_df["Sex"])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_df["Sex"]= le.fit_transform(test_df["Sex"])
print(test_df["Sex"])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df["Embarked"]= le.fit_transform(train_df["Embarked"])
print(train_df["Embarked"])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_df["Embarked"]= le.fit_transform(test_df["Embarked"])
print(test_df["Embarked"])
train_df.head()
test_df.head()
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
''' OR
X_train = train_df[:, 0:-1]
Y_train = train_df[:, -1]
X_test  = test_df[:, 1:]
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(Y_train)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
''' 
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

'''
from sklearn.metrics import accuracy_score
classifier.score(X_train, Y_train)
classifier = round(classifier.score(X_train, Y_train) * 100, 2)
classifier
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)