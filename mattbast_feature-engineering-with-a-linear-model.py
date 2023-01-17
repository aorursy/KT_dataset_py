import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn import linear_model, ensemble

from sklearn.metrics import mean_squared_error, mean_absolute_error



import tensorflow as tf



from tqdm.notebook import tqdm



import os

from PIL import Image
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
train.head()
train.info()
test.head()
test.info()
train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
submission['Patient'] = (

    submission['Patient_Week']

    .apply(

        lambda x:x.split('_')[0]

    )

)



submission['Weeks'] = (

    submission['Patient_Week']

    .apply(

        lambda x: int(x.split('_')[-1])

    )

)



submission =  submission[['Patient','Weeks', 'Confidence','Patient_Week']]



submission = submission.merge(test.drop('Weeks', axis=1), on="Patient")
submission.head()
train['Dataset'] = 'train'

test['Dataset'] = 'test'

submission['Dataset'] = 'submission'
all_data = train.append([test, submission])



all_data = all_data.reset_index()

all_data = all_data.drop(columns=['index'])
all_data.head()
train_patients = train.Patient.unique()
fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = train[train['Patient'] == train_patients[i]]



    ax[i].set_title(train_patients[i])

    ax[i].plot(patient_log['Weeks'], patient_log['FVC'])
all_data['FirstWeek'] = all_data['Weeks']

all_data.loc[all_data.Dataset=='submission','FirstWeek'] = np.nan

all_data['FirstWeek'] = all_data.groupby('Patient')['FirstWeek'].transform('min')
first_fvc = (

    all_data

    .loc[all_data.Weeks == all_data.FirstWeek][['Patient','FVC']]

    .rename({'FVC': 'FirstFVC'}, axis=1)

    .groupby('Patient')

    .first()

    .reset_index()

)



all_data = all_data.merge(first_fvc, on='Patient', how='left')
all_data.head()
all_data['WeeksPassed'] = all_data['Weeks'] - all_data['FirstWeek']
all_data.head()
def calculate_height(row):

    if row['Sex'] == 'Male':

        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])

    else:

        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])



all_data['Height'] = all_data.apply(calculate_height, axis=1)
all_data.head()
all_data = pd.concat([

    all_data,

    pd.get_dummies(all_data.Sex),

    pd.get_dummies(all_data.SmokingStatus)

], axis=1)



all_data = all_data.drop(columns=['Sex', 'SmokingStatus'])
all_data.head()
def scale_feature(series):

    return (series - series.min()) / (series.max() - series.min())



all_data['Weeks'] = scale_feature(all_data['Weeks'])

all_data['Percent'] = scale_feature(all_data['Percent'])

all_data['Age'] = scale_feature(all_data['Age'])

all_data['FirstWeek'] = scale_feature(all_data['FirstWeek'])

all_data['FirstFVC'] = scale_feature(all_data['FirstFVC'])

all_data['WeeksPassed'] = scale_feature(all_data['WeeksPassed'])

all_data['Height'] = scale_feature(all_data['Height'])
feature_columns = [

    'Percent',

    'Age',

    'FirstWeek',

    'FirstFVC',

    'WeeksPassed',

    'Height',

    'Female',

    'Male', 

    'Currently smokes',

    'Ex-smoker',

    'Never smoked',

]
train = all_data.loc[all_data.Dataset == 'train']

test = all_data.loc[all_data.Dataset == 'test']

submission = all_data.loc[all_data.Dataset == 'submission']
train[feature_columns].head()
model = linear_model.HuberRegressor(max_iter=200)
model.fit(train[feature_columns], train['FVC'])
predictions = model.predict(train[feature_columns])
plt.bar(train[feature_columns].columns.values, model.coef_)

plt.xticks(rotation=90)

plt.show()
mse = mean_squared_error(

    train['FVC'],

    predictions,

    squared=False

)



mae = mean_absolute_error(

    train['FVC'],

    predictions

)



print('MSE Loss: {0:.2f}'.format(mse))

print('MAE Loss: {0:.2f}'.format(mae))
def competition_metric(trueFVC, predFVC, predSTD):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    



print(

    'Competition metric: ', 

    competition_metric(train['FVC'].values, predictions, 285) 

)
train['prediction'] = predictions
plt.scatter(predictions, train['FVC'])



plt.xlabel('predictions')

plt.ylabel('FVC (labels)')

plt.show()
delta = predictions - train['FVC']

plt.hist(delta, bins=20)

plt.show()
fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = train[train['Patient'] == train_patients[i]]



    ax[i].set_title(train_patients[i])

    ax[i].plot(patient_log['WeeksPassed'], patient_log['FVC'], label='truth')

    ax[i].plot(patient_log['WeeksPassed'], patient_log['prediction'], label='prediction')

    ax[i].legend()
submission[feature_columns].head()
sub_predictions = model.predict(submission[feature_columns])

submission['FVC'] = sub_predictions
test_patients = list(submission.Patient.unique())

fig, ax = plt.subplots(5, 1, figsize=(10, 20))



for i in range(5):

    patient_log = submission[submission['Patient'] == test_patients[i]]



    ax[i].set_title(test_patients[i])

    ax[i].plot(patient_log['WeeksPassed'], patient_log['FVC'])
submission = submission[['Patient_Week', 'FVC']]



submission['Confidence'] = 285
submission.to_csv('submission.csv', index=False)
submission.head()