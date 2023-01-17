import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
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
df=pd.read_csv("/kaggle/input/titanic/train.csv")
df2=pd.read_csv("/kaggle/input/titanic/test.csv")
#df=df1+df2
#df1
df.describe(include=['O'])
df.columns.values
df.groupby(["Pclass"])["Survived"].mean()
df.groupby(["Sex"])["Survived"].mean()
df["Age"].mode()
df["Age"]=df["Age"].replace(np.NaN, df["Age"].mean())
df["Age"]
df[df["Survived"]==0]
from matplotlib import pyplot as plt

fig,ax = plt.subplots(1,2,figsize=(5, 7))
#plt.figure(figsize=(10,10))
age0 = df[df["Survived"]==0].Age
age1= df[df["Survived"]==1].Age
ax[0].hist(age0,bins=20)
ax[1].hist(age1,bins=20)
ax[0].set_title("(survivor =0)")
ax[1].set_title("(survivor =1)")
#ax.set_xticks([0,25,50,75,100])
ax[0].set_xlabel('age')
ax[1].set_xlabel('age')
ax[0].set_ylabel('no. of passengers')
ax[1].set_ylabel('no. of passengers')
plt.show()
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

df["Age_cat"]=pd.cut(df["Age"], 6)
df.groupby(["Age_cat"])["Survived"].mean()
df.loc[ df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[ df['Age'] > 64, 'Age'] = 5

df2.loc[ df['Age'] <= 16, 'Age'] = 0
df2.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df2.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df2.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df2.loc[ df['Age'] > 64, 'Age'] = 5
df.head()
df["Name"]
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df2['FamilySize'] = df2['SibSp'] + df2['Parch'] + 1

df.groupby("FamilySize")["Survived"].mean().sort_values(ascending=False)
df.groupby("Embarked")["Survived"].mean()
df['Sex'] = df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
df2['Sex'] = df2['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

df2["Fare"]=df2["Fare"].replace(np.NaN, df2["Fare"].mean())
df2[df2.Fare.isna()==True]
#df["Fare_group"]=pd.qcut(df["Fare"],4)
pd.qcut(df["Fare"],4)
df["Fare_group"]=pd.cut(df["Fare"],5)

df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
df.loc[ df['Fare'] > 31, 'Fare'] = 3
df['Fare'] = df['Fare'].astype(int)
df2.loc[ df2['Fare'] <= 7.91, 'Fare'] = 0
df2.loc[(df2['Fare'] > 7.91) & (df2['Fare'] <= 14.454), 'Fare'] = 1
df2.loc[(df2['Fare'] > 14.454) & (df2['Fare'] <= 31), 'Fare']   = 2
df2.loc[ df2['Fare'] > 31, 'Fare'] = 3
df2['Fare'] = df2['Fare'].astype(int)

df = df.drop(['Ticket', 'Cabin','SibSp','Parch','PassengerId','Embarked','Age_cat',"Name","Fare_group"], axis=1)
df2 = df2.drop(['Ticket', 'Cabin','SibSp','Parch','PassengerId','Embarked','Age_cat','Name',"Fare_group"], axis=1)

df
train=df.iloc[:,1:]
#train=train.drop(["Name"], axis=1)
train
df.iloc[:,0]
test=df2
df2 = df2.drop([ 'SibSp','Parch','PassengerId','Embarked','Name','Cabin'], axis=1)

df2=df2.drop(["Ticket","Cabin"],axis=1)
#df["Fare_group"]=pd.cut(df["Fare"],5)
#df.groupby("Fare_group")["Survived"].mean()
df
df=df.drop(["Fare_group"],axis=1)
test=df2
test
test=test.drop(["Ticket"],axis=1)
test
train.shape
train=train.drop(["Ticket"],axis=1)
test.shape
from sklearn.linear_model import LogisticRegression

X_train = train
Y_train = df.iloc[:,0]
X_test  = test
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
df3=pd.read_csv("/kaggle/input/titanic/test.csv")
test_df=df3
test_df["PassengerId"]
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission
