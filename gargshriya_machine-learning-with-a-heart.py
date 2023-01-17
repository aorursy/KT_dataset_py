from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# for training our model

train_values = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_values.csv', index_col='patient_id')

train_labels = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_labels.csv', index_col='patient_id')
train_values.head()
train_values.dtypes
train_labels.head()
train_labels.heart_disease_present.value_counts().plot.bar(title='Number with Heart Disease')
selected_features = ['age', 

                     'sex', 

                     'max_heart_rate_achieved', 

                     'resting_blood_pressure']

train_values_subset = train_values[selected_features]
sns.pairplot(train_values.join(train_labels), 

             hue='heart_disease_present', 

             vars=selected_features)
# for preprocessing the data

from sklearn.preprocessing import StandardScaler



# the model

from sklearn.linear_model import LogisticRegression



# for combining the preprocess with model training

from sklearn.pipeline import Pipeline



# for optimizing parameters of the pipeline

from sklearn.model_selection import GridSearchCV
pipe = Pipeline(steps=[('scale', StandardScaler()), 

                       ('logistic', LogisticRegression())])

pipe
param_grid = {'logistic__C': [0.0001, 0.001, 0.01, 1, 10], 

              'logistic__penalty': ['l1', 'l2']}

gs = GridSearchCV(estimator=pipe, 

                  param_grid=param_grid, 

                  cv=3)
gs.fit(train_values_subset, train_labels.heart_disease_present)

gs.best_params_
from sklearn.metrics import log_loss



in_sample_preds = gs.predict_proba(train_values[selected_features])

log_loss(train_labels.heart_disease_present, in_sample_preds)
test_values = pd.read_csv('../input/warm-up-machine-learning-with-a-heart/test_values.csv', index_col='patient_id')
test_values_subset = test_values[selected_features]

predictions = gs.predict_proba(test_values_subset)[:, 1]

submission_format = pd.read_csv('../input/format/submission_format.csv', index_col='patient_id')

my_submission = pd.DataFrame(data=predictions,

                             columns=submission_format.columns,

                             index=submission_format.index)
my_submission.head()

my_submission.to_csv('../input/solution.csv')