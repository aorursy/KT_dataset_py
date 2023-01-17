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
df_train_dummies = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)

df_train_dummies
df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)

df_test_dummies
X_train_dummies = df_train_dummies.drop(columns='Gender_male')

Gm = df_train_dummies[['Gender_male', 'Diabetes']] / 2.

y_train_dummies = df_train_dummies['Diabetes']  # Numpy行列を取り出す

X_train_dummies = (X_train_dummies - X_train_dummies.mean()) / X_train_dummies.std(ddof=0)

X_train_dummies = X_train_dummies.merge(Gm * 2, on='Patient number')

X_train_dummies = X_train_dummies.drop(columns='Diabetes_x')

X_train_dummies = X_train_dummies.drop(columns='Diabetes_y')

X_train_dummies
X_train_dummies = X_train_dummies.drop(columns='Waist/hip ratio')

X_train_dummies = X_train_dummies.drop(columns='Chol/HDL ratio')

X_train_dummies = X_train_dummies.drop(columns='Height')
X_test_dummies = df_test_dummies.drop(columns='Gender_male')

Gm = df_test_dummies[['Gender_male']] / 2.

X_test_dummies = (X_test_dummies - X_test_dummies.mean()) / X_test_dummies.std(ddof=0)

X_test_dummies = X_test_dummies.merge(Gm * 2, on='Patient number')

X_test_dummies
X_test_dummies = X_test_dummies.drop(columns='Waist/hip ratio')

X_test_dummies = X_test_dummies.drop(columns='Chol/HDL ratio')

X_test_dummies = X_test_dummies.drop(columns='Height')
from collections import Counter

print('Original dataset shape %s' % Counter(y_train_dummies))
# ライブラリ

from imblearn.under_sampling import RandomUnderSampler



# 正例の数を保存

positive_count_train = y_train_dummies.sum()



rus = RandomUnderSampler({0:positive_count_train, 1:positive_count_train}, random_state=0)



# 学習用データに反映

X_train_dummies, y_train_dummies = rus.fit_resample(X_train_dummies, y_train_dummies)
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=0)



# 学習用データに反映

X_train_dummies, y_train_dummies = ros.fit_resample(X_train_dummies, y_train_dummies)
from collections import Counter

print('Original dataset shape %s' % Counter(y_train_dummies))
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

X_train, X_valid, y_train, y_valid = train_test_split(X_train_dummies, y_train_dummies, test_size=0.3, random_state=0) 
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgb

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import StackingClassifier



estimators = [

        ('svc', SVC(max_iter=500, random_state=0)),

        ('lgb', lgb.LGBMClassifier(learning_rate=0.005, max_depth=2,

                                   n_estimators=300, num_leaves=4, 

                                   random_state=0)),

        ]



clf = StackingClassifier(

    estimators=estimators,

    final_estimator=LogisticRegression(max_iter=300),

)

clf.fit(X_train, y_train)
from sklearn.metrics import roc_curve, auc



y_pred = clf.predict(X_valid)  # 予測

fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める

auc(fpr, tpr)  # 評価
y_pred = clf.predict_proba(X_test_dummies)[:, 1]  # 予測
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = y_pred

submit.to_csv('submission13.csv', index=False)