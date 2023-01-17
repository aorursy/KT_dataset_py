import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/student-mat.csv')
data.school.replace(['GP','MS'],[0,1],inplace=True)

data.sex.replace(['F','M'],[0,1],inplace=True)

data.address.replace(['U', 'R'],[0,1],inplace=True)

data.famsize.replace(['GT3', 'LE3'],[0,1],inplace=True)

data.Pstatus.replace(['A', 'T'],[0,1],inplace=True)

data.Mjob.replace(['at_home', 'health', 'other', 'services', 'teacher'],\

                 [0,1,2,3,4],inplace=True)

data.Fjob.replace(['teacher', 'other', 'services', 'health', 'at_home'],\

                 [0,1,2,3,4],inplace=True)

data.reason.replace(['course', 'other', 'home', 'reputation'],[0,1,2,3],inplace=True)

data.guardian.replace(['mother', 'father', 'other'],[0,1,2],inplace=True)

data.schoolsup.replace(['yes', 'no'],[0,1],inplace=True)

data.famsup.replace(['yes', 'no'],[0,1],inplace=True)

data.paid.replace(['yes', 'no'],[0,1],inplace=True)

data.activities.replace(['yes', 'no'],[0,1],inplace=True)

data.nursery.replace(['yes', 'no'],[0,1],inplace=True)

data.higher.replace(['yes', 'no'],[0,1],inplace=True)

data.internet.replace(['yes', 'no'],[0,1],inplace=True)

data.romantic.replace(['yes', 'no'],[0,1],inplace=True)
print(data.shape)

data.head(5)

data.info()
plt.figure(figsize=(20,7))

sns.set(font_scale=2)

sns.countplot(data.age);
plt.figure(figsize=(20,7))

sns.set(font_scale=2)

sns.countplot(x=data.age,hue=data.sex);
plt.figure(figsize=(20,10))

sns.set(font_scale=1.5)

pd.crosstab(data.school,data.sex).plot(kind='barh');
plt.figure(figsize=(20,20))

sns.heatmap(data.corr(),cmap='hot');
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
data_train, data_test, targ_train, targ_test = train_test_split(\

            data.drop(['Walc','Dalc'],axis=1), data.Dalc, test_size=0.2)
forest = RandomForestClassifier(criterion='gini', n_estimators=200, max_depth=4)

forest.fit(data_train, targ_train)
scores = cross_val_score(forest, data.drop(['Walc','Dalc'],axis=1), data.Dalc, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
forest.feature_importances_
ss = pd.DataFrame(np.hstack((np.array(list(data.drop(['Walc','Dalc'],axis=1).columns)).reshape(31,1),forest.feature_importances_.reshape(31,1))))
features = ss[0]

importances = forest.feature_importances_

indices = np.argsort(importances)



plt.figure(figsize=(15,15))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center');

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance');
data = pd.read_csv('../input/student-mat.csv')
plt.figure(figsize=(10,7))

sns.set(font_scale=2)

sns.countplot(x=data.sex,hue=data.Dalc);
plt.figure(figsize=(10,7))

sns.set(font_scale=2)

sns.countplot(x=data.goout,hue=data.Dalc);