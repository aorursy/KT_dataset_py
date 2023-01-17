# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.DataFrame(pd.read_csv('/kaggle/input/titanic/train.csv'))
test_df = pd.DataFrame(pd.read_csv('/kaggle/input/titanic/test.csv'))
gender_df =  pd.DataFrame(pd.read_csv('/kaggle/input/titanic/gender_submission.csv'))

train_df.head()
train_df.describe()
train_df.isnull().sum()
train_df = train_df.drop(["Cabin"], axis=1)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

train_df.info()
train_df = train_df.drop(["PassengerId","Fare", "Ticket", "Name"], axis = 1)
cat_col= train_df.drop(train_df.select_dtypes(exclude=['object']), axis=1).columns
print(cat_col)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df[cat_col[0]] = le.fit_transform(train_df[cat_col[0]].astype('str'))

train_df[cat_col[1]] = le.fit_transform(train_df[cat_col[1]].astype('str'))
train_df.head()
# Pclass

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Pclass')
# Sex

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Sex')
# Age

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Age')
# SibSp

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'SibSp')
# Parch

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Parch')
# Embarked

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Embarked')
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']
#now lets split data in test train pairs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# model training 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()   # Here kernel used is RBF (Radial Basis Function)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_df.head()
# To check Accuracy

from sklearn import metrics

# Generate the roc curve using scikit-learn.
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))

# Measure the Accuracy Score
print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_pred, y_test)))
test_df.head()
test_df.isnull().sum()
test_df = test_df.drop(["Cabin"], axis=1)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df.isnull().sum()
test_df.info()
PassengerId = test_df['PassengerId']
test_df = test_df.drop(["PassengerId","Fare", "Ticket", "Name"], axis = 1)
test_df.info()
cat_col
test_df[cat_col[0]] = le.fit_transform(test_df[cat_col[0]].astype('str'))

test_df[cat_col[1]] = le.fit_transform(test_df[cat_col[1]].astype('str'))
test_df.head()
y_test_pred = model.predict(test_df)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_test_pred
    })

submission.to_csv('submission.csv', index=False)
