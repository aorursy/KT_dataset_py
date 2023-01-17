import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
%cd /kaggle
!ls input
# inputフォルダのファイル

import glob

for f in glob.glob("./input/*"):

    print(f)
df = pd.read_csv("./input/train.csv")
df
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df.Embarked = df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

#df.Cabin = df.Cabin.replace('NaN', 0)

df.Sex = df.Sex.replace(['male', 'female'], [0, 1])

df.Age = df.Age.replace('NaN', 0)
df.columns
df
corrmat = df.corr()

corrmat
f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
sns.set()

cols = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']

sns.pairplot(df[cols], size = 2.5)

plt.show()
train_labels = df['Survived'].values

train_features = df

train_features.drop('Survived', axis=1, inplace=True)

train_features = train_features.values.astype(np.int64)
from sklearn import svm



#Standard = svm.LinearSVC(C=1.0, intercept_scaling=1, multi_class=False , loss="l1", penalty="l2", dual=True)

svm = svm.LinearSVC()

svm.fit(train_features, train_labels)
df_test = pd.read_csv("./input/test.csv")
df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.Embarked = df_test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

#df.Cabin = df.Cabin.replace('NaN', 0)

df_test.Sex = df_test.Sex.replace(['male', 'female'], [0, 1])

df_test.Age = df_test.Age.replace('NaN', 0)
test_features = df_test.values.astype(np.int64)
 # 各点を分類する

y_test_pred = svm.predict(test_features)
df_out = pd.read_csv("./input/test.csv")

df_out["Survived"] = y_test_pred
submission = df_out[["PassengerId","Survived"]]

submission.to_csv("./working/submission.csv",index=False)
!ls

!ls working