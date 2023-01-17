# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

pd.set_option('display.max_columns', 100)

import numpy as np

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from warnings import filterwarnings

filterwarnings(action='ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")

data.shape
data.head()

print(data.describe())

print(data.info())
# Understand the correlation between differnt variables in the dataset

corr = data.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(corr, cmap="coolwarm", annot=True)
data['fetal_health'] = data['fetal_health'].astype('int')

data['fetal_health'].value_counts()
''' Creating Bar plot for our target variable '''

plt.figure(figsize=(15,5))

data.fetal_health.value_counts().plot(kind='bar')

plt.xlabel('Health Status')

plt.ylabel('Count')

plt.legend()

plt.show()
''' Visualising through Pie chart as well '''

data.fetal_health.value_counts().plot(kind='pie')
""" Before Over Sampling """

X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values

print(X.shape,y.shape)

print(Counter(y))
''' After Over Sampling '''

# !pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X, y = oversample.fit_resample(X, y)

print(X.shape,y.shape)

print(Counter(y))
from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = MinMaxScaler()



X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve,roc_auc_score,f1_score,precision_recall_curve,confusion_matrix,classification_report
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.20,shuffle=True,stratify=y,random_state = 42)
lr_model = LogisticRegression(penalty='l2',dual=False,

    tol=0.0001,

    C=1.0,

    fit_intercept=True,

    intercept_scaling=1,

    class_weight=None,

    random_state=42,

    solver='lbfgs',

    max_iter=100,

    multi_class='ovr',

    verbose=0,

    warm_start=False,

    n_jobs=None,

    l1_ratio=None,)
y_pred_train = lr_model.fit(x_train,y_train)



y_pred_test = lr_model.predict(x_test)
cm = confusion_matrix(y_test,y_pred_test)

cm
report = classification_report(y_test,y_pred_test)

print(report)