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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.isna().sum()
df['Survived'].value_counts(normalize=True)
feat = df.copy()
feat.set_index('PassengerId',inplace=True)

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') 
feat['Age'] = imp_mean.fit_transform(feat[['Age']]) #Age missing values will go with the mean. Depending on model performance we may change this strategy later
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent') 
feat['Embarked'] = imp_mode.fit_transform(feat[['Embarked']])
feat = feat[feat['Cabin'].isnull()]
feat.drop(columns=['Cabin','Name','Ticket'], inplace=True)

feat.isna().sum()
df_cabin = df.copy()
df_cabin.set_index('PassengerId',inplace=True)
df_cabin['Age'] = imp_mean.fit_transform(df_cabin[['Age']]) #Age missing values will go with the mean. Depending on model performance we may change this strategy later
df_cabin['Embarked'] = imp_mode.fit_transform(df_cabin[['Embarked']])
df_cabin.dropna(inplace=True)
df_cabin.drop(columns=['Name','Ticket'], inplace=True)
df_cabin.isna().sum()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_cabin['Cabin'] = df_cabin['Cabin'].str[0]
df_cabin['Sex'] = le.fit_transform(df_cabin['Sex'])
df_cabin['Embarked'] = le.fit_transform(df_cabin['Embarked'])
df_cabin['Cabin'] = le.fit_transform(df_cabin['Cabin'])

df_cabin.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y_cabin = df_cabin['Cabin']
X_cabin = StandardScaler().fit_transform(df_cabin.drop(columns=['Cabin','Survived']))

X_cabin_train, X_cabin_test, y_cabin_train, y_cabin_test = train_test_split(X_cabin, y_cabin, random_state=7)
from sklearn.ensemble import GradientBoostingClassifier

cabinGBC = GradientBoostingClassifier(random_state=7, max_depth=5, n_estimators = 1000)
cabinGBC.fit(X_cabin_train, y_cabin_train)
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_cabin_hat = hummerGBC.predict(X_cabin_test)
confusion = confusion_matrix(y_cabin_test, y_cabin_hat)
sns.heatmap(confusion, annot=True)
from sklearn.metrics import classification_report

print(classification_report(y_cabin_test, y_cabin_hat, zero_division=0))
feat['Sex'] = le.fit_transform(feat['Sex'])
feat['Embarked'] = le.fit_transform(feat['Embarked'])

df_cabin.head()
feat.head()
X_feat = feat.drop(columns='Survived')
feat['Cabin'] = cabinGBC.predict(X_feat)
feat.head()
df1 = feat.append(df_cabin)
df1.sort_index(inplace=True)
df1.head()
from sklearn.preprocessing import KBinsDiscretizer

df1['Child'] = df1['Age']<17 #Creating Child Variable. Will define as younger than 17
df1['Child'] = le.fit_transform(df1['Child'])
df1['Family'] = df1['SibSp'] + df1['Parch']
df1['Alone'] = df1['Family'] ==0
df1['Alone'] = le.fit_transform(df1['Alone'])

est = KBinsDiscretizer(n_bins=5, encode = 'ordinal', strategy='uniform')
df1['Age_bin'] = est.fit_transform(df1[['Age']])
df1.head()
sns.heatmap(df1.corr())
y = df1['Survived']
X = StandardScaler().fit_transform(df1.drop(columns=['Survived','Child','SibSp','Parch','Age', 'Alone']))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
#from sklearn.model_selection import GridSearchCV

"""param_grid = {'learning_rate' = [0.01,0.1,0.5,1,2],
             'n_estimators' = [50,100,500,1000,2000],
             }
"""

hummerGBC = GradientBoostingClassifier(random_state=7, max_depth=5, n_estimators = 1000)
hummerGBC.fit(X_train, y_train)
print('GBC Score:', hummerGBC.score(X_test, y_test))
      
from sklearn.ensemble import RandomForestClassifier
hummerRFC = RandomForestClassifier(random_state=7, max_depth=7, n_estimators = 1000)
hummerRFC.fit(X_train, y_train)
print('RFC Score:', hummerRFC.score(X_test, y_test))

from sklearn.ensemble import AdaBoostClassifier
hummerADA = AdaBoostClassifier(random_state=7, n_estimators = 7000, learning_rate = 0.1)
hummerADA.fit(X_train, y_train)
print('ADA Score:', hummerADA.score(X_test, y_test))
from sklearn.metrics import confusion_matrix

y_hat = hummerGBC.predict(X_test)
confusion = confusion_matrix(y_test, y_hat)
sns.heatmap(confusion, annot=True)
from sklearn.metrics import f1_score

f1_score(y_test, y_hat, average='binary')
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(hummerGBC, random_state=7).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = df1.drop(columns=['Survived','Child','SibSp','Parch','Age','Alone']).columns.tolist())
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
df2 = df1.drop(columns=['Survived','Child','SibSp','Parch','Age', 'Alone', 'Fare'])
df2 = enc.fit_transform(df2).toarray()

y_NB = df1['Survived']
X_NB = df2

X_NB_train, X_NB_test, y_NB_train, y_NB_test = train_test_split(X_NB, y_NB, random_state=7)

from sklearn.naive_bayes import CategoricalNB

hummerNB = CategoricalNB()
hummerNB.fit(X_NB_train, y_NB_train)
hummerNB.score(X_NB_test, y_NB_test)
y_NB_hat = hummerNB.predict(X_NB_test)
confusion = confusion_matrix(y_NB_test, y_NB_hat)
sns.heatmap(confusion, annot=True)
f1_score(y_NB_test, y_NB_hat, average='binary')
df_pred = pd.read_csv('/kaggle/input/titanic/test.csv')
df_pred.isna().sum()
feat_pred = df_pred.copy()
feat_pred.set_index('PassengerId',inplace=True)

feat_pred['Age'] = imp_mean.fit_transform(feat_pred[['Age']]) #Age missing values will go with the mean. Depending on model performance we may change this strategy later
feat_pred['Fare'] = imp_mean.fit_transform(feat_pred[['Fare']])
feat_pred['Embarked'] = imp_mode.fit_transform(feat_pred[['Embarked']])
feat_pred = feat_pred[feat_pred['Cabin'].isnull()]
feat_pred.drop(columns=['Cabin','Name','Ticket'], inplace=True)

feat_pred.isna().sum()
le.fit(feat_pred['Sex'])
feat_pred['Sex'] = le.transform(feat_pred['Sex'])
le.fit(feat_pred['Embarked'])
feat_pred['Embarked'] = le.transform(feat_pred['Embarked'])
X_feat_pred = feat_pred
feat_pred['Cabin'] = cabinGBC.predict(X_feat_pred)

feat_pred.head()
df_cabin_pred = df_pred.copy()
df_cabin_pred.set_index('PassengerId',inplace=True)
df_cabin_pred['Age'] = imp_mean.fit_transform(df_cabin_pred[['Age']]) #Age missing values will go with the mean. Depending on model performance we may change this strategy later
df_cabin_pred['Embarked'] = imp_mode.fit_transform(df_cabin_pred[['Embarked']])
df_cabin_pred.dropna(inplace=True)
df_cabin_pred.drop(columns=['Name','Ticket'], inplace=True)
df_cabin_pred.isna().sum()
df_cabin_pred['Cabin'] = df_cabin_pred['Cabin'].str[0]
df_cabin_pred['Sex'] = le.fit_transform(df_cabin_pred['Sex'])
df_cabin_pred['Embarked'] = le.fit_transform(df_cabin_pred['Embarked'])
df_cabin_pred['Cabin'] = le.fit_transform(df_cabin_pred['Cabin'])

df_cabin_pred.head()
df_pred1 = feat_pred.append(df_cabin_pred)
df_pred1.sort_index(inplace=True)
df_pred1.head()
df_pred1['Child'] = df_pred1['Age']<17 #Creating Child Variable. Will define as younger than 17
df_pred1['Child'] = le.fit_transform(df_pred1['Child'])
df_pred1['Family'] = df_pred1['SibSp'] + df_pred1['Parch']
df_pred1['Alone'] = df_pred1['Family'] ==0
df_pred1['Alone'] = le.fit_transform(df_pred1['Alone'])

est = KBinsDiscretizer(n_bins=5, encode = 'ordinal', strategy='uniform')
df_pred1['Age_bin'] = est.fit_transform(df_pred1[['Age']])
df_pred1.head()
X_pred = StandardScaler().fit_transform(df_pred1.drop(columns=['Child','SibSp','Parch','Age', 'Alone']))
X_pred = StandardScaler().fit_transform(X_pred)
y_hat = hummerGBC.predict(X_pred)
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_hat})
submission.to_csv('hummer1.csv', index=False)
