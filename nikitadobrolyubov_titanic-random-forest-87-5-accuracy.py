import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train['Survived'].replace(1, "Yes",inplace=True)
train['Survived'].replace(0, "No",inplace=True)
train['Survived']
plt.subplots(figsize=(10,10))
sns.countplot('Sex',hue='Survived',data=train, palette='RdBu_r')
plt.show()
plt.figure(figsize=[10,10])
sns.distplot(train['Age'].dropna().values, bins=range(0,17), kde=False, color="#007598")
sns.distplot(train['Age'].dropna().values, bins=range(16, 33), kde=False, color="#7B97A0")
sns.distplot(train['Age'].dropna().values, bins=range(32, 49), kde=False, color="#06319B")
sns.distplot(train['Age'].dropna().values, bins=range(48,65), kde=False, color="#007598")
sns.distplot(train['Age'].dropna().values, bins=range(64,81), kde=False, color="#000000",
            axlabel='Age')
plt.show()
train['Age_Category'] = pd.cut(train['Age'],
                        bins=[0,16,32,48,64,81])
plt.subplots(figsize=(10,10))
sns.countplot('Age_Category',hue='Survived',data=train, palette='RdBu_r')
plt.show()
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
    
train.head()
train.head()
train['Family'] = train['SibSp'] + train['Parch'] + 1
train['Alone'] = 0
train.loc[train['Family'] == 1, 'Alone'] = 1
train.head()
train['Survived'].replace("Yes", 1,inplace=True)
train['Survived'].replace("No", 0, inplace=True)
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]
sns.barplot(x='Pclass', y='Survived', data=train, palette='RdBu_r');
train['Sex'].replace("male", 0, inplace=True)
train['Sex'].replace("female", 1, inplace=True)
train.head()
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)

train.head()
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'PassengerId', 'Age_Category', 'FareBand'], axis=1)
train['Age'] = train['Age'].fillna(2)
train['Age'] = train['Age'].astype(int)
train.head()
training, testing = train_test_split(train, test_size=0.2, random_state=0)
cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family', 'Alone']
tcols = np.append(['Survived'],cols)
df = training.loc[:,tcols].dropna()

X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
y_red_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X, y)*100, 2)
acc_random_forest
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X, y) * 100, 2)
acc_log_reg
from sklearn.svm import SVC, LinearSVC
clf = SVC()
clf.fit(X, y)
y_pred_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X, y) * 100, 2)
print (acc_linear_svc)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Linear SVC', 'Random Forest'],
    'Score': [acc_log_reg, acc_linear_svc, acc_random_forest]})

models.sort_values(by='Score', ascending=False)
