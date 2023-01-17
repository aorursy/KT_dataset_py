# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.ensemble
import sklearn_pandas
import sklearn.preprocessing, sklearn.decomposition
import sklearn.linear_model, sklearn.pipeline, sklearn.metrics
import matplotlib.pyplot as plt
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
all_data_df = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')
# Shuffl everything
all_data_df = all_data_df.sample(frac=1).reset_index(drop=True)
# The times seem meaningless so we discard them and just keep the date.
all_data_df['ScheduledDay'] = all_data_df['ScheduledDay'].apply(
    lambda x: datetime.datetime.strptime(x[0:11], '%Y-%m-%dT'))
all_data_df['AppointmentDay'] = all_data_df['AppointmentDay'].apply(
    lambda x: datetime.datetime.strptime(x[0:11], '%Y-%m-%dT'))
all_data_df['AppointmentDayOfTheWeek'] = all_data_df['AppointmentDay'].apply(lambda x: x.weekday())
all_data_df['DaysBetweenBookingAndAttending'] = all_data_df['AppointmentDay'] - all_data_df['ScheduledDay']
all_data_df['DaysBetweenBookingAndAttending'] = all_data_df['DaysBetweenBookingAndAttending'].apply(lambda x: x.days)
all_data_df
# see https://github.com/scikit-learn-contrib/sklearn-pandas for exampels of data transforms

mapper = sklearn_pandas.DataFrameMapper([
    ('Gender', sklearn.preprocessing.LabelBinarizer()),
    ('Age', sklearn.preprocessing.FunctionTransformer()),
    ('Scholarship', sklearn.preprocessing.FunctionTransformer()),
    ('Hipertension', sklearn.preprocessing.FunctionTransformer()),
    ('Diabetes', sklearn.preprocessing.FunctionTransformer()),
    ('Alcoholism', sklearn.preprocessing.FunctionTransformer()),
    ('Handcap', sklearn.preprocessing.FunctionTransformer()),
    ('SMS_received', sklearn.preprocessing.FunctionTransformer()),
    ('Neighbourhood', sklearn.preprocessing.LabelBinarizer()),
    ('DaysBetweenBookingAndAttending', sklearn.preprocessing.FunctionTransformer()),
    ('AppointmentDayOfTheWeek', sklearn.preprocessing.LabelBinarizer()),
    ('No-show', sklearn.preprocessing.LabelBinarizer()),
])
all_data = mapper.fit_transform(all_data_df.copy())

val_X = all_data[0:10000, 0:-1]
val_Y = all_data[0:10000, -1]
train_X = all_data[10000:, 0:-1]
train_Y = all_data[10000:, -1]

rf_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
rf_clf.fit(train_X, train_Y)

baseline_clf = sklearn.dummy.DummyClassifier('prior')
baseline_clf.fit(train_X, train_Y)
ax = plt.gca()

svc_disp = sklearn.metrics.plot_roc_curve(rf_clf, val_X, val_Y, ax=ax)
sklearn.metrics.plot_roc_curve(baseline_clf, val_X, val_Y, ax=ax)
