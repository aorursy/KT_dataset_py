# For linear algebra
import numpy as np  

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# For building a model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
import xgboost
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.shape
train_df.info()
train_df.describe()
train_df.isnull().sum()
# replacing null values with median in Age column
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# replacing null values with mode in Embarked column
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# we will drop Cabin column from our data
train_df.drop('Cabin', axis=1, inplace=True)
print("Columns with their count of null values: ")
train_df.isnull().sum()
train_df.head()
print(train_df.Survived.value_counts())
sns.countplot(x='Survived', data=train_df, palette='rainbow')
sns.countplot(x='Pclass',hue='Survived',data=train_df)
sns.barplot(x='Sex', y='Survived', data=train_df)
sns.violinplot(x='Age', data=train_df, palette='Greens_r')
sns.barplot(x='Survived', y='Age', data=train_df, palette='rocket_r')
sns.countplot(x='SibSp', hue='Survived',data=train_df, palette='binary_r')
sns.catplot(x='Parch', data=train_df, kind='count', col='Survived')
sns.barplot(x='Embarked', y='Survived', data=train_df, palette='Spectral')
train_df.head()
train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train_df.head()
lb = LabelEncoder()
lb.fit(train_df.Embarked)
train_df['Embarked'] = lb.transform(train_df.Embarked)
train_df.head()
lb.classes_
train_df = pd.get_dummies(train_df) # One-Hot Encoding is also called dummy encoding, we use pd.get_dummies func
train_df.head()
# X contains all the columns except the Survived columns, becuase predictions will be made on Survived column
# Y contains only the Survived column
# Note: the column we are going to predict is also called target

X = train_df.drop('Survived', axis=1)
y = train_df.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"Training size: {X_train.shape[0]}")
print(f"Testing size: {X_test.shape[0]}")
gbm = GradientBoostingClassifier(n_estimators=1000)
gbm.fit(X_train, y_train)
gbm_preds = gbm.predict(X_test)
metrics.accuracy_score(y_test, gbm_preds)
rfc = RandomForestClassifier(n_jobs=2, n_estimators=500, oob_score=True)
rfc.fit(X_train, y_train)
rfc_preds = rfc.predict(X_test)
metrics.accuracy_score(y_test, rfc_preds)
lgbm = lightgbm.LGBMClassifier()
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
metrics.accuracy_score(y_test, lgbm_preds)
xgb = xgboost.XGBClassifier(n_jobs=2, n_estimators=500, base_score=0.7)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
metrics.accuracy_score(y_test, xgb_preds)
etc = ExtraTreesClassifier(n_jobs=2, bootstrap=True, oob_score=True, verbose=2, n_estimators=1000)
etc.fit(X_train, y_train)
etc_preds = etc.predict(X_test)
metrics.accuracy_score(y_test, etc_preds)
adbc = AdaBoostClassifier(n_estimators=500, learning_rate=0.04)
adbc.fit(X_train, y_train)
adbc_preds = adbc.predict(X_test)
metrics.accuracy_score(y_test, adbc_preds)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_preds = dtc.predict(X_test)
metrics.accuracy_score(y_test, dtc_preds)
lg = LogisticRegression(max_iter=1000, verbose=4, n_jobs=3, dual=True)
lg.fit(X_train, y_train)
lg_preds = lg.predict(X_test)
metrics.accuracy_score(y_test, lg_preds)
test_df = pd.read_csv('../input/test.csv')
test_df.head()
PassengerId = test_df.PassengerId
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.head()
# replacing null values with median in Age column
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# replacing null values with mode in Embarked column
test_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
lb.fit(test_df.Embarked)
test_df.Embarked = lb.transform(test_df.Embarked)
test_df = pd.get_dummies(test_df)
test_df.head()
predictions = lgbm.predict(test_df)
predictions
submit_df = pd.DataFrame()
submit_df['PassengerId'] = PassengerId
submit_df['Survived'] = predictions
submit_df.head()
submit_df.to_csv('submission.csv', index=False)