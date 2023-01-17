# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.info()
train_data[train_data.Survived == 1]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
df_pclass = train_data.groupby(['Pclass', 'Survived'], as_index=False).count()
sns.barplot(x = 'Pclass', y='PassengerId', hue='Survived', data=df_pclass, ci="sd", ax=ax1)

# For sex
df_sex = train_data.groupby(['Sex', 'Survived'], as_index=False).count()
sns.barplot(x = 'Sex', y='PassengerId', hue='Survived', data=df_sex, ci="sd", ax=ax2)

#SibSp
df_sib = train_data.groupby(['SibSp', 'Survived'], as_index=False).count()
sns.barplot(x = 'SibSp', y='PassengerId', hue='Survived', data=df_sib, ci="sd", ax=ax3)

#Parch
df_parch = train_data.groupby(['Parch', 'Survived'], as_index=False).count()
sns.barplot(x = 'Parch', y='PassengerId', hue='Survived', data=df_parch, ci="sd", ax=ax4)
result = pd.DataFrame(columns=['PassengerId', 'Survived'])
result.PassengerId = test_data.PassengerId
train_data = train_data.drop('Cabin', axis=1)
test_data = test_data.drop('Cabin', axis=1)
train_data = train_data.drop(['PassengerId','Name','Ticket'], axis=1)
test_data = test_data.drop(['PassengerId','Name','Ticket'], axis=1)
train_data.Sex = train_data.Sex.map({'female': 0, 'male': 1})
test_data.Sex = test_data.Sex.map({'female': 0, 'male': 1})
avg_train_age = train_data.Age.mean()
train_data.Age = np.where(train_data.Age != train_data.Age, avg_train_age, train_data.Age)
avg_test_age = test_data.Age.mean()
test_data.Age = np.where(test_data.Age != test_data.Age, avg_test_age, test_data.Age)
avg_train_fare = train_data.Fare.mean()
train_data.Fare = np.where(train_data.Fare != train_data.Fare, avg_train_fare, train_data.Fare)
avg_test_fare = test_data.Fare.mean()
test_data.Fare = np.where(test_data.Fare != test_data.Fare, avg_test_fare, test_data.Fare)
train_data.Embarked = np.where(train_data.Embarked != train_data.Embarked, pd.value_counts(train_data.Embarked, sort=True, ascending=False).index[0], train_data.Embarked)
test_data.Embarked = np.where(test_data.Embarked != test_data.Embarked, pd.value_counts(test_data.Embarked, sort=True, ascending=False).index[0], test_data.Embarked)
train_data.Embarked = train_data.Embarked.map({'S': 1, 'C': 2, 'Q':3})
test_data.Embarked = test_data.Embarked.map({'S': 1, 'C': 2, 'Q':3})
trainY = train_data.Survived
trainX = train_data.drop('Survived', axis=1)
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(trainX)  
trainX = scaler.transform(trainX)  
test_data = scaler.transform(test_data) 
from sklearn.ensemble import BaggingRegressor
#lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#svc = SVC(C=10, gamma=0.1)
clf = BaggingRegressor(n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
clf.fit(trainX, trainY)

result.Survived = clf.predict(test_data).astype(np.int32)
submission = pd.DataFrame({
        "PassengerId": result["PassengerId"],
        "Survived": result.Survived
})
submission.to_csv('submission.csv', index=False)