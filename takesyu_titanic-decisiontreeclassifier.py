# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print(train_data.corr())
%matplotlib inline

fig, ax = plt.subplots(figsize=(12, 9)) 

sns.heatmap(train_data.corr(), square=True, vmax=1, vmin=-1, center=0)

plt.show()
print(train_data.isnull().sum())
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())
print(test_data.isnull().sum())
train_data["FamilySize"] = 1 + train_data["SibSp"] + train_data["Parch"] 

test_data["FamilySize"]  = 1 + test_data["SibSp"] + test_data["Parch"]  
test_data.head()
print(train_data.corr())
y_train = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Fare","FamilySize"]

X_train = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



train_age = train_data["Age"]

test_age = test_data["Age"]



X_train = pd.concat([train_age, X_train],axis=1)

X_test = pd.concat([test_age, X_test],axis=1)
print(X_test)
clf = tree.DecisionTreeClassifier(criterion='gini',

  splitter='best', max_depth=3, min_samples_split=2,

  min_samples_leaf=1, min_weight_fraction_leaf=0.0,

  max_features=None, random_state=1,

 )
clf = clf.fit(X_train, y_train)
ranking = np.argsort(-clf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=clf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12,8))

for i in np.arange(8):

    ax = fig.add_subplot(5,6,i+1)

    sns.regplot(x=X_train.iloc[:,i], y=y_train)



plt.tight_layout()

plt.show()
predict_clftree = clf.predict(X_test)
print(predict_clftree)
pred_df = pd.DataFrame(predict_clftree)
pred_df= pred_df.rename(columns={0: 'Survived'})

pred_df = pred_df.astype(int)

pred_df = pd.concat([test_data["PassengerId"], pred_df],axis=1)
pred_df.head()
pred_df.to_csv('my_submission.csv', index=False)