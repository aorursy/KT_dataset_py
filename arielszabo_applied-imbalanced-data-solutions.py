import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

np.random.seed(64) # initialize a random seed, this will help us make the random stuff reproducible. 

import warnings

warnings.filterwarnings('ignore') # ignore jupyter's warnings, this is only used for the purpose of the blog post
data = pd.read_csv(os.path.join('..', 'input', 'KaggleV2-May-2016.csv'), parse_dates=['AppointmentDay', 'ScheduledDay'],

                   dtype={

                       'Scholarship': bool,

                       'Hipertension': bool,

                       'Diabetes': bool,

                       'Alcoholism': bool

                        },

                   index_col='AppointmentID'

)
data.head()
sns.countplot(data['No-show']);
data['IsMale'] = data['Gender'] == 'M'
drop_columns = ['ScheduledDay', 'AppointmentDay', 'PatientId', 'Gender', 'Neighbourhood']

data.drop(drop_columns, inplace=True, axis=1)
data.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
X = data.drop('No-show', axis=1)

y = data['No-show']

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
model = LogisticRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on training set:', model.score(x_train, y_train))

print('score on test set:', model.score(x_test, y_test))
print(metrics.classification_report(y_true=y_test, y_pred=pred))
y_yes = y_train[y_train == 'Yes']

x_yes = x_train.loc[y_yes.index]



y_no = y_train[y_train == 'No']

x_no = x_train.loc[y_no.index]
oversample_X = pd.concat([x_no, x_yes, x_yes, x_yes, x_yes])

oversample_y =  pd.concat([y_no, y_yes, y_yes, y_yes, y_yes])
model = LogisticRegression()

model.fit(oversample_X, oversample_y)

pred = model.predict(x_test)

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
y_yes = y_train[y_train == 'Yes']

x_yes = x_train.loc[y_yes.index]
y_no = y_train[y_train == 'No']

undersample_y_no = y_no.sample(y_yes.shape[0])



undersample_x_no = x_train.loc[undersample_y_no.index]
undersample_y = pd.concat([undersample_y_no, y_yes])

undersample_X = pd.concat([undersample_x_no, x_yes])
model = LogisticRegression()

model.fit(undersample_X, undersample_y)

pred = model.predict(x_test)

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
from imblearn.over_sampling import SMOTE



x_train_smote, y_train_smote = SMOTE(ratio='auto', k_neighbors=5, m_neighbors=10,

      out_step=0.5, kind='regular', svm_estimator=None, n_jobs=-1).fit_sample(x_train, y_train)
from collections import Counter



print('The original class distribution: {},'.format(Counter(y_train)))

print('After SMOTE class distribution:  {}'.format(Counter(y_train_smote)))
model = LogisticRegression()

model.fit(x_train_smote, y_train_smote)

pred = model.predict(x_test)

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
from sklearn.utils import class_weight

train_weights = class_weight.compute_sample_weight('balanced', y=y_train)
model = LogisticRegression()

model.fit(x_train, y_train, sample_weight=train_weights)

pred = model.predict(x_test)

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))
model = LogisticRegression(class_weight='balanced')

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('score on test set:', model.score(x_test, y_test))

print(metrics.classification_report(y_true=y_test, y_pred=pred))