import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(2)
test.head(2)
data = train.copy()
data.head()
data.isnull().sum()
sns.barplot(x='Sex',y='Survived',data=data,hue='Pclass')
data['FmemOnboard'] = data['SibSp'] + data['Parch']
sns.barplot(x='FmemOnboard',y='Survived',data=data)
sns.distplot(data[(data['Age'].isnull()==False)]['Age'])
sns.countplot(x='Sex',data=data)
sns.countplot(x='Sex',data=data[data['Survived']==1])
sns.barplot(x=data['Embarked'],y=data['Survived'])
passengerId = test['PassengerId']

titanic = train.append(test,ignore_index=True)

df = titanic.copy()
df.info()
train_count = train.shape[0]

test_count = test.shape[0]
df.head()
embarked_freq = df['Embarked'].value_counts()
most_freq_embarked = embarked_freq[embarked_freq==embarked_freq.max()].index[0]
most_freq_embarked
df['Embarked'] = df['Embarked'].fillna(most_freq_embarked)
df.info()
x = df.groupby(['Pclass'])
x.mean()
class_of_missing_fare = df[df['Fare'].isnull()]['Pclass'].iloc[0]

class_of_missing_fare
x.mean()
to_fill_in_missing_fare = x.mean().loc[class_of_missing_fare,'Fare']

to_fill_in_missing_fare
df['Fare'].fillna(to_fill_in_missing_fare,inplace=True)
df.info()
age_dict = dict(df.groupby(['Pclass','Sex']).mean()['Age'])
age_dict[(1,'female')]
age_dict
type(list(age_dict.keys())[0])
type((df.loc[0]['Pclass'],df.loc[0]['Sex']))
def fill_age(row):

    if np.isnan(row['Age']):

        tp = (row['Pclass'],row['Sex'])

        return age_dict[tp]

    else:

        return row['Age']
np.isnan(df.iloc[0]['Age'])
df['Age'] = df.apply(fill_age,axis=1)
df.info()
df['Age'] = df['Age'].apply(round)
df.head()
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df.head()
df['Cabin'].nunique()
type(df['Cabin'][0])
df['Cabin'].value_counts()
df['Cabin'].apply(lambda x : 'Unknown' if type(x)==float else x).astype(str)
df['Cabin'] = (df['Cabin'].apply(lambda x : 'Unknown' if type(x)==float else x).astype(str)).apply(lambda x : x.split()[0][0])
df.head()
df.drop(['Ticket'],axis=1,inplace=True)
df.head()
df.info()
cabin_dummies = pd.get_dummies(df['Cabin'])
cabin_dummies.drop('U',axis=1,inplace=True)
embarked_dummies = pd.get_dummies(df['Embarked'])
embarked_dummies.drop('S',axis=1,inplace=True)
pclass_dummies = pd.get_dummies(df['Pclass'])

pclass_dummies
pclass_dummies.drop(3,axis=1,inplace=True)
pclass_dummies.head()
cabin_dummies.columns = 'Cabin ' + cabin_dummies.columns
cabin_dummies.head()
embarked_dummies.columns = 'Embark ' + embarked_dummies.columns
embarked_dummies.head()
(list(pclass_dummies.columns))
pclass_dummies.columns = ['Pclass 1','Pclass 2']
pclass_dummies.head()
df.head()
df.drop(['Cabin','Embarked'],axis=1,inplace=True)
df.head()
df.drop(['PassengerId','Name','Pclass'],axis=1,inplace=True)
df.head()
df_dummies = pd.concat([df, pclass_dummies, cabin_dummies, embarked_dummies], axis=1)
df_dummies.head()
train = df_dummies.iloc[:train_count]

test = df_dummies.iloc[train_count:]
train.info()
test.info()
X = train.drop('Survived',axis=1)

y = train['Survived'].astype(int)
X_to_predict = test.drop('Survived',axis=1)

X_to_predict.head()
X_to_predict.shape[0]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
from sklearn.metrics import classification_report,confusion_matrix
print('For logistic regression model')

print('Confusion Matrix')

print(confusion_matrix(y_test,logreg.predict(X_test)))
print('For logistic regression model')

print('Classification Report')

print(classification_report(y_test,logreg.predict(X_test)))
logreg_predictions = logreg.predict(X_to_predict)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=4)
clf.fit(X_train,y_train)
print('For decision tree classifier')

print('Confusion Matrix')

print(confusion_matrix(y_test,clf.predict(X_test)))
print('For decision tree classifier')

print('Classification Report')

print(classification_report(y_test,clf.predict(X_test)))
kaggle = pd.DataFrame({'PassengerId':passengerId,'Survived':logreg.predict(X_to_predict)})
kaggle.to_csv('submit.csv',index=False)