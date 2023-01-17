!pip install ppscore
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pylab import rcParams

import ppscore as pps



%matplotlib inline



rcParams['figure.figsize'] = 15, 8

pd.options.display.max_columns = None
data = pd.read_csv("/kaggle/input/fetal-health-classification/fetal_health.csv")
data.head(10)
data.isna().sum()
data.info()
data['fetal_health'].value_counts()
sns.countplot(data['fetal_health'])

plt.show()
data.describe()
sns.heatmap(data.corr(), annot=True, fmt='.1f')

plt.show()
f, axes = plt.subplots(2, 3)

sns.barplot(x='fetal_health', y='abnormal_short_term_variability', data=data, ax=axes[0][0])

sns.barplot(x='fetal_health', y='prolongued_decelerations', data=data, ax=axes[0][1])

sns.barplot(x='fetal_health', y='accelerations', data=data, ax=axes[0][2])

sns.barplot(x='fetal_health', y='percentage_of_time_with_abnormal_long_term_variability', data=data, ax=axes[1][0])

sns.barplot(x='fetal_health', y='histogram_mode', data=data, ax=axes[1][1])

sns.barplot(x='fetal_health', y='uterine_contractions', data=data, ax=axes[1][2])

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, f1_score
X = data.drop(['fetal_health'], axis=1)

y = data['fetal_health']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

X_train
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
GBM_params =  { 

    'n_estimators': [200, 500, 800, 1000],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,10],

    'criterion' :['gini', 'entropy']

}
GBM_model = GridSearchCV(rfc, GBM_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
GBM_model.best_params_
upgraded_rfc = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt', n_estimators=500)

upgraded_rfc.fit(X_train, y_train)

up_y_pred = upgraded_rfc.predict(X_test)
print(classification_report(y_test, y_pred))

print(classification_report(y_test, up_y_pred))