import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import category_encoders as ce

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df.head(3)
from sklearn.model_selection import GroupShuffleSplit



train_inds, val_inds = next(GroupShuffleSplit(test_size=.35, n_splits=2, random_state = 42).split(df, groups=df['Patient']))

train = df.iloc[train_inds]

val = df.iloc[val_inds]



col = 'Patient'



cardinality = len(pd.Index(df[col]).value_counts())

print("Number of " + df[col].name + "s in original DataFrame df: " + str(cardinality) + '\n')   

cardinality = len(pd.Index(train[col]).value_counts())

print("Number of " + train[col].name + "s in train: " + str(cardinality) + '\n')

cardinality = len(pd.Index(val[col]).value_counts())

print("Number of " + val[col].name + "s in val: " + str(cardinality))



target = 'FVC'

features = train.drop(columns=[target, 'Patient']).columns.tolist()



X_train = train[features]

y_train = train[target]

X_val = val[features]

y_val = val[target]
model = make_pipeline(ce.OrdinalEncoder(),

                       StandardScaler(),

                       LinearRegression())

model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)

y_val_pred = model.predict(X_val)



print("Val mean squared error",mean_squared_error(y_val,y_val_pred))

sns.distplot(y_val, label='Actual')

sns.distplot(y_val_pred, label='Validation Prediction')

plt.legend();
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = pd.merge(sub, test, on = ['Patient'], how='outer')

test.drop(columns = ['Patient_Week', 'Patient','FVC_x','Confidence','Weeks_y','FVC_y'], inplace=True)



test.head(3)
s = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



y_pred = model.predict(test)

sub = s[['Patient_Week']]

sub['FVC'] = y_pred

sub['Confidence'] = sub['FVC'].std()

sub['FVC'] = sub['FVC'].astype(int)

sub['Confidence'] = sub['Confidence'].astype(int)

sub.head(3)
sub.to_csv("submission.csv", index=False)

a = pd.read_csv('./submission.csv')

a.head(2)