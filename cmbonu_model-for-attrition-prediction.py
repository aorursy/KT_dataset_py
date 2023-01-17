import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
hr_raw = pd.read_csv("../input/HR_comma_sep.csv")

hr_raw.head()
hr_raw.corr().sort_values(by='left')['left']
hr_raw.columns
dept_ohc=pd.get_dummies(hr_raw[['sales','salary']],prefix=['dept','salary'])

hr_raw_w = pd.concat((hr_raw,dept_ohc),axis=1).drop(['sales','salary','left'],axis=1)

hr_raw_std = StandardScaler().fit_transform(hr_raw_w)



hr_raw_w.shape
from sklearn.metrics import accuracy_score, log_loss,classification_report

from sklearn.cross_validation import train_test_split
train_d,test_d,train_l,test_l = train_test_split(hr_raw_std,hr_raw['left'],train_size= 0.80,random_state=540)
clf = RandomForestClassifier()

clf.fit(train_d,train_l)
print(classification_report(test_l,clf.predict(test_d)))
accuracy_score(clf.predict(train_d),train_l)
accuracy_score(clf.predict(test_d),test_l)
plt.figure(figsize=(10,3))

plt.scatter(range(len(hr_raw_w.columns.values)), clf.feature_importances_)

plt.xticks(range(len(hr_raw_w.columns.values)), hr_raw_w.columns.values,rotation=90)