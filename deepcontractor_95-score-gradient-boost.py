import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.describe()
data.head()
sns.distplot(data['age'])
sns.distplot(data['anaemia'])
sns.distplot(data['creatinine_phosphokinase'])
sns.distplot(data['ejection_fraction'])
sns.distplot(data['platelets'])
sns.distplot(data['serum_sodium'])
sns.countplot(data['sex'])
sns.countplot(data['smoking'])
x = data.loc[:,'age':'time']

y = data.loc[:,["DEATH_EVENT"]]



x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
gradientboost_clf = GradientBoostingClassifier(max_depth=2)

gradientboost_clf.fit(x_train,y_train)

gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_clf.score(x_test,y_test)
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, gradientboost_pred)
sns.heatmap(cf_matrix, cmap='Blues',annot = True)