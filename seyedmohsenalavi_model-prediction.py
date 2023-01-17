#For data processing

import pandas as pd

import numpy as np



#For visualizations

import matplotlib.pyplot as plt

import seaborn as sns



#For ignoring warnings

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

result = pd.read_csv("../input/titanic/gender_submission.csv")
train.tail()
test.head()
train_copy = train.copy()

train_copy.describe()
test_copy = test.copy()

test_copy.describe()
print('train Shape : ',train_copy.shape,'\ntest shape : ',test_copy.shape)
train_copy.dtypes
test_copy.dtypes
from IPython.display import Markdown, display

def printmd(a,b,c,d):

    display(Markdown(a))

    display(b)

    display(Markdown(c))

    display(d)

printmd ( '**sum of null data in train set :**\n',train_copy.isnull().sum(),'\n**sum of null data in test set :**\n',test_copy.isnull().sum())
print('Number of Survived[0] and unsurvived[1] passengers : \n',train_copy['Survived'].value_counts())
final = pd.concat([train_copy,test_copy],axis = 0)

final.drop(['Survived'],axis = 1,inplace = True)
final.tail()
final.shape
final['Age'].fillna(final['Age'].mean(), inplace=True)

final['Age'].isnull().sum()
final["Fare"] = final["Fare"].fillna(final["Fare"].median())

final['Fare'].isnull().sum()
final['Fare'].dtypes
final["Fare"] = final["Fare"].map(lambda n: np.log(n) if n > 0 else 0)

final['Fare'].head()
new = final['Name'].str.split('.', n=1, expand = True)

final['First'] = new[0]

final['Last'] = new[1]

new1 = final['First'].str.split(',', n=1, expand = True)

final['Last Name'] = new1[0]

final['Title'] = new1[1]

new2 = final['Title'].str.split('', n=1, expand = True)
final['Title'].value_counts()
final.drop(['First','Last','Name','Last Name'],axis = 1,inplace = True)
final.replace(to_replace = [ ' Don', ' Rev', ' Dr', ' Mme',

        ' Major', ' Sir', ' Col', ' Capt',' Jonkheer'], value = ' Honorary(M)', inplace = True)

final.replace(to_replace = [ ' Ms', ' Lady', ' Mlle',' the Countess', ' Dona'], value = ' Honorary(F)', inplace = True)
final['Title'].value_counts()
final = pd.get_dummies(final, columns = ["Title"])

final.head()
final["Family"] = final["SibSp"] + final["Parch"] + 1
final['Single'] = final['Family'].map(lambda s: 1 if s == 1 else 0)

final['SmallF'] = final['Family'].map(lambda s: 1 if  s == 2  else 0)

final['MedF'] = final['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)

final['LargeF'] = final['Family'].map(lambda s: 1 if s >= 5 else 0)
final['Embarked'].fillna("S",inplace = True)
final = pd.get_dummies(final, columns = ["Embarked"], prefix="Embarked_from_")

final.head()
final.Cabin.isnull().sum()
final.Cabin.value_counts()
final['Cabin_final'] = final['Cabin'].str[0]
final['Cabin_final'].fillna('Unknown',inplace = True)
final['Cabin_final'].value_counts()
final.drop(['Cabin'],axis = 1,inplace = True)
final = pd.get_dummies(final, columns = ["Cabin_final"],prefix="Cabin_")
final.head()
final.drop(['Ticket'],axis = 1,inplace = True)
final.head()
final = pd.get_dummies(final, columns = ["Sex"],prefix="Gender_")
final.head()
final.drop(['PassengerId'],axis = 1,inplace = True)

final.drop(['SibSp','Parch','Family'],axis = 1,inplace = True)
final.dtypes
final.isnull().sum()
final_train = final.copy()

final_train =  final_train[:891]

final_train = pd.concat([final_train,train_copy['Survived']],axis = 1)

final_train.head()
sns.countplot(x = 'Survived', data = final_train)
sns.countplot(x = 'Pclass', data = final_train)
sns.countplot(x = 'Sex', data = train_copy)
plt.figure(figsize=(30,10))

sns.countplot(x = 'Age', data = train_copy)
plt.figure(figsize=(15,8))

sns.distplot(train_copy['Fare'], hist=True, rug=True)
sns.distplot(final['Fare'], hist=True, rug=True)
sns.catplot(x ='Survived', y ='SibSp', data = train_copy)
sns.catplot(x ='Survived', y ='Parch', data = train_copy)
sns.catplot(x = 'Sex',y='Survived',hue = 'Pclass', kind = 'bar', data = train_copy, col = 'Pclass', color = 'red')
sns.catplot(x ='Survived', y ='Age', hue = 'Pclass',kind='violin',data = train_copy)
sns.catplot(x = 'SibSp',y='Survived',hue = 'Pclass',kind = 'violin', data = train_copy, palette = 'BuGn_r', col = 'Pclass')
sns.catplot(x = 'Parch',y='Survived',hue = 'Pclass',kind = 'violin', data = train_copy, palette = 'cubehelix', col = 'Pclass')
sns.catplot(x = 'Embarked',y='Survived',kind = 'point', data = train_copy, hue = 'Pclass', col = 'Pclass')
correlation = final.copy()

sur = pd.concat([train['Survived'],result['Survived']],axis = 0)

correlation = pd.concat([correlation,sur],axis = 1)
plt.figure(figsize=(30,30))

sns.heatmap(correlation.corr(), annot=True, linewidth=0.6, cmap='coolwarm')
#The models trained

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import BernoulliNB



#For Scaling and Hyperparameter Tuning

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn import metrics



#Voting Classifier

from sklearn.ensemble import VotingClassifier 
x_train = final[:891]

feature_scaler = MinMaxScaler()

x_train = feature_scaler.fit_transform(x_train)

x_train
x_test = final[891:]

feature_scaler = MinMaxScaler()

x_test = feature_scaler.fit_transform(x_test)

x_test
y_train = train['Survived']
y_test = result['Survived']
Log=LogisticRegression()

Log.fit(x_train,y_train.values.ravel())

model1pred = Log.predict(x_test)
accuracy=Log.score(x_test,y_test)

accuracy
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': model1pred})

output.to_csv('submission.csv', index=False)