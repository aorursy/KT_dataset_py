import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, classification_report, confusion_matrix, auc, roc_auc_score, f1_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import tensorflow as tf

from tensorflow import keras

from keras import optimizers

from keras.models import Model, Sequential

from keras.layers import Dense, Activation, Dropout

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
train = pd.read_csv('../input/hr-analytics/HR Analytics/train.csv')

test = pd.read_csv('../input/hr-analytics/HR Analytics/test.csv')

sample = pd.read_csv('../input/hr-analytics/HR Analytics/sample_submission.csv')
train.head()
test.head()
train.info()
train.isna().sum()
test.info()
test.isna().sum()
plt.figure(figsize=(9,5))

sns.countplot(y=train.is_promoted)
dept = pd.DataFrame(train['department'].value_counts())

plt.figure(figsize=(10,6))

sns.barplot(x=dept.department, y=dept.index)

plt.xlabel('Employee count')

plt.ylabel('Department')                

plt.title('No. of employees by department')
plt.figure(figsize=(10,6))

dept_wise_promo = train['department'][train.is_promoted==1]

sns.countplot(y=dept_wise_promo, order=dept_wise_promo.value_counts().index)

plt.title('No. of promotions by department')

plt.xlabel('No. of promotions')
edu = pd.DataFrame(train.education.value_counts())

fig = px.pie(edu, values = edu.education, names = edu.index, title = 'Break up of education')

fig.show()
gender = pd.DataFrame(train.gender.value_counts())

fig = px.pie(gender, values = gender.gender, names = gender.index, title = 'Break up of gender')

fig.show()
channel = pd.DataFrame(train.recruitment_channel.value_counts())

fig = px.pie(channel, values = channel.recruitment_channel, names = channel.index, title = 'Recruitment channel')

fig.show()
plt.figure(figsize=(12,6))

sns.distplot(train['age'])

plt.title('Age distribution')
plt.figure(figsize=(12,6))

sns.distplot(train['length_of_service'], bins=40)

plt.title('Length of service distribution')
plt.figure(figsize=(12,7))

sns.scatterplot(x=train.avg_training_score, y=train.no_of_trainings, hue=train.is_promoted)
plt.figure(figsize=(12,7))

sns.scatterplot(x=train.avg_training_score, y=train.length_of_service, hue=train.is_promoted)
promoted = train[train.is_promoted==1]

not_promoted = train[train.is_promoted==0]
fig,ax = plt.subplots(1,2)

fig.set_size_inches(22,7)



plt.subplot(121)

sns.distplot(promoted['avg_training_score'])

plt.title('Promoted Group')



plt.subplot(122)

sns.distplot(not_promoted['avg_training_score'], color='red')

plt.title('Not promoted group')
fig,ax = plt.subplots(1,2)

fig.set_size_inches(18,6)



plt.subplot(121)

sns.countplot(promoted['KPIs_met >80%'])

plt.title('Promoted Group')



plt.subplot(122)

sns.countplot(not_promoted['KPIs_met >80%'])

plt.title('Not promoted group')
fig, axes = plt.subplots(1,2)

fig.set_size_inches(20,6)



plt.subplot(121)

sns.countplot(train.previous_year_rating)

plt.title('Prev yr rating for all employees')



plt.subplot(122)

sns.countplot(promoted.previous_year_rating)

plt.title('Prev yr rating for promoted employees')
promo_by_gender = pd.DataFrame(promoted.gender.value_counts())

fig = px.pie(promo_by_gender, values = promo_by_gender.gender, names = promo_by_gender.index, title = 'Promotion by gender')

fig.show()
fig,(ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(18,5)



secondary = train['is_promoted'][train.education=='Below Secondary']

ax1.pie(secondary.value_counts())

ax1.set_title('Below secondary')

ax1.legend(['not_promoted', 'promoted'])



bachelors = train['is_promoted'][train.education=='Bachelor\'s']

ax2.pie(bachelors.value_counts())

ax2.set_title('Bachelor\'s')

ax2.legend(['not_promoted', 'promoted'])



masters = train['is_promoted'][train.education=='Master\'s & above']

ax3.pie(masters.value_counts())

ax3.set_title('Master\'s & above')

ax3.legend(['not_promoted', 'promoted'])



plt.show()
train['education'].fillna('Bacherlor\'s', inplace=True)

test['education'].fillna('Bachelor\'s', inplace=True)
le = LabelEncoder()

train['department'] = le.fit_transform(train.department)

test['department'] = le.transform(test.department)



train['region'] = le.fit_transform(train.region)

test['region'] = le.transform(test.region)



train['education'] = le.fit_transform(train.education)

test['education'] = le.transform(test.education)



train['gender'] = le.fit_transform(train.gender)

test['gender'] = le.transform(test.gender)



train['recruitment_channel'] = le.fit_transform(train.recruitment_channel)

test['recruitment_channel'] = le.transform(test.recruitment_channel)



train.drop('employee_id', axis=1, inplace=True)

test_ids = test.pop('employee_id')

y = train.pop('is_promoted')
imp = IterativeImputer(max_iter=3, random_state=0)

imp.fit(train)

X_train = imp.transform(train)

test_data = imp.transform(test)



df_train = pd.DataFrame(X_train, columns = train.columns).astype('int32')

df_test = pd.DataFrame(test_data, columns = test.columns).astype('int32')
print(f'Original train prev yr rating:\n{train.previous_year_rating.value_counts()} \n\nNew train prev yr rating:\n{df_train.previous_year_rating.value_counts()}')
np.random.seed(5)

x_tr_idx, x_val_idx, x_test_idx = np.split(df_train.index, [int(.7*len(df_train)), int(.85*len(df_train))])

x_train, y_train = df_train.iloc[x_tr_idx, :], y[x_tr_idx]

x_val, y_val = df_train.iloc[x_val_idx, :], y[x_val_idx]

x_test, y_test = df_train.iloc[x_test_idx, :], y[x_test_idx]



print(f' x_train shape: {x_train.shape} \n x_val shape: {x_val.shape} \n x_test shape: {x_test.shape}')
xgb = XGBClassifier(max_depth=8, reg_lambda=5)

xgb.fit(x_train, y_train)

xgb_val_pred = xgb.predict(x_val)

xgb_train_pred = xgb.predict(x_train)

print(f'Train f1-score: {f1_score(y_train, xgb_train_pred):.4f}')

print(f'Validation data f1-score: {f1_score(y_val, xgb_val_pred):.4f}')

xgb_test_pred = xgb.predict(x_test)

print(f'Test data f1-score: {f1_score(y_test, xgb_test_pred):.4f}')
lgb = LGBMClassifier(max_depth=10)

lgb.fit(x_train, y_train)

lgb_val_pred = lgb.predict(x_val)

lgb_train_pred = lgb.predict(x_train)

print(f'Train f1-score: {f1_score(y_train, lgb_train_pred):.4f}')

print(f'Validation data f1-score: {f1_score(y_val, lgb_val_pred):.4f}')

lgb_test_pred = lgb.predict(x_test)

print(f'Test data f1-score: {f1_score(y_test, lgb_test_pred):.4f}')
cbc = CatBoostClassifier(eval_metric='F1', bagging_temperature=20, verbose=0)

cbc.fit(x_train, y_train)

cbc_val_pred = cbc.predict(x_val)

cbc_train_pred = cbc.predict(x_train)

print(f'Train f1-score: {f1_score(y_train, cbc_train_pred):.4f}')

print(f'Validation data f1-score: {f1_score(y_val, cbc_val_pred):.4f}')

cbc_test_pred = cbc.predict(x_test)

print(f'Test data f1-score: {f1_score(y_test, cbc_test_pred):.4f}')
test_predictions = pd.DataFrame(xgb.predict(df_test))

test_predictions.insert(0, column = 'employee_id', value = test_ids)

test_predictions.rename(columns={0: 'is_promoted'})

test_predictions.to_csv('my_submission.csv', index=False)