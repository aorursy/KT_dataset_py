# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv', index_col=0)
df_train
df_test
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.countplot(x='Diabetes', data=df_train)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

# Class count

count_class_0, count_class_1 = df_train.Diabetes.value_counts()

 

# Divide by class

df_class_0 = df_train[df_train['Diabetes'] == 0]

df_class_1 = df_train[df_train['Diabetes'] == 1]



df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)



print(df_train_over.Diabetes.value_counts())



%matplotlib inline

sns.countplot(x='Diabetes', data=df_train_over)

plt.show()
df_train_dummies = pd.get_dummies(df_train_over, columns=['Gender'], drop_first=True)

df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)
from sklearn.ensemble import RandomForestClassifier



X_train_dummies = df_train_dummies.drop(columns='Diabetes').values  # Numpy行列を取り出す

y_train_dummies = df_train_dummies['Diabetes'].values  # Numpy行列を取り出す



rfc = RandomForestClassifier(max_depth=4, n_estimators=50, criterion='gini')

rfc.fit(X_train_dummies, y_train_dummies)  # 学習
X_test = df_test_dummies.values  # Numpy行列を取り出す

y_pred = rfc.predict_proba(X_test)[:, 1]  # 予測
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = y_pred

submit.to_csv('submission.csv', index=False)