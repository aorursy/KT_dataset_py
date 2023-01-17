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
dt_train = pd.read_csv("/kaggle/input/titanic/train.csv")
dt_test = pd.read_csv("/kaggle/input/titanic/test.csv")
print(dt_train.dtypes)
dt_train.head(7)
dt_test.head(7)
dt_train.describe()
dt_train.info()
print(dt_train.columns.values)
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style="ticks", color_codes=True)

# concat train and test data
data = [dt_train, dt_test]
data = pd.concat(data)
data.shape
sns.relplot(x="PassengerId", y="Age",hue="Sex", data=data);
sns.relplot(x="PassengerId", y="Age",hue="Sex",kind="line" , data=data);
sns.relplot(x="PassengerId", y="Age",hue="Sex",col="Pclass" , data=data);

sns.catplot(x="Pclass", y="Survived", hue="Sex", data=data);
g = sns.FacetGrid(data, col="Sex")
g.map(plt.hist, "Survived");
g = sns.FacetGrid(data, col="Pclass")
g.map(plt.hist, "Survived");
X_train = dt_train.drop(["PassengerId", "Name"], axis=1)
X_test = dt_test.drop(["PassengerId", "Name"], axis=1)
X_train.head()
from sklearn.preprocessing import OneHotEncoder
import numpy as np
enc = OneHotEncoder()

enc.fit(X_train[['Sex']])
Sex = enc.transform(X_train[['Sex']]).toarray()
Sex
#COLUNMS: FEMALE MALE
X_train.insert(2,"Female", Sex[:,0],True)
X_train.insert(3,"Male", Sex[:,1],True)
X_train.head()
X_train.drop("Sex", axis=1, inplace=True)
X_train.head()
em = X_train[['Embarked']]
em.fillna('A', inplace= True)
em.isnull().sum()
enc2 = OneHotEncoder()
enc2.fit(em)
Embarked = enc2.transform(em).toarray()
test = [['S'], ['C'], ['Q'], ['A']]
enc2.transform(test).toarray()
X_train['A'] = Embarked[:,0]
X_train['C'] = Embarked[:,1]
X_train['Q'] = Embarked[:,2]
X_train['S'] = Embarked[:,3]


X_train.drop("Embarked", axis=1, inplace=True)
X_train.head(10)
X_train.drop("Cabin", axis=1, inplace=True)
X_train.head(10)
Y_train = X_train['Survived']
X_train.drop("Survived", axis=1, inplace=True)
print(Y_train)
print(X_train)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
Y_train.size
X_train.Ticket
X_train.drop("Ticket", axis=1, inplace=True)
X_train.head()
X_train.info()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train[['Age','Fare']])

age = imp.transform(X_train[['Age','Fare']])
X_train.Age  =  age
X_train.info()
X_train.head()
X_test.drop('Cabin', inplace=True, axis=1)
X_test.head()
X_test.drop('Ticket', inplace=True, axis=1)
X_test.head()
emt = X_test[['Embarked']]
Embarked = enc2.transform(emt).toarray()
Embarked
X_test['A'] = Embarked[:,0]
X_test['C'] = Embarked[:,1]
X_test['Q'] = Embarked[:,2]
X_test['S'] = Embarked[:,3]
X_test.drop("Embarked", axis=1, inplace=True)
X_test.head(10)
Sex = enc.transform(X_test[['Sex']]).toarray()
X_test.insert(2,"Female", Sex[:,0],True)
X_test.insert(3,"Male", Sex[:,1],True)
X_test.drop("Sex", axis=1, inplace=True)

X_test.info()
X_train.to_csv('X_df_train.csv')
X_test.to_csv('X_df_test.csv')
Y_train = Y_train.reshape((891,1))
X_train.shape, Y_train.shape, X_test.shape
X_train = pd.read_csv('/kaggle/working/X_df_train.csv')
X_test = pd.read_csv('/kaggle/working/X_df_test.csv')
X_test.Age.fillna(0, inplace= True)
X_test.Fare.fillna(0, inplace= True)
from sklearn.linear_model import LogisticRegression
lo_reg = LogisticRegression()
lo_reg.fit(X_train, Y_train)
Y_pred = lo_reg.predict(X_test)
acc_log = round(lo_reg.score(X_train, Y_train) * 100, 2)
acc_log
from joblib import dump
dump(lo_reg, 'flo_reg.joblib') 
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
dump(random_forest, 'random_forest.joblib') 
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
dump(svc, 'svc.joblib') 
# we can use another algorithme
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier