import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
train.head()
train.info()
test.head()
train.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
submission.head(10)
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split("_")[0])
submission['Weeks'] = submission['Patient_Week'].apply(lambda x: x.split("_")[-1])
submission
submission = submission[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
submission = submission.merge(test.drop('Weeks', axis=1), on='Patient')
submission
train['Dataset'] = 'train'

test['Dataset'] = 'test'

submission['Dataset'] = 'submission'
all_data = train.append([test, submission])
all_data = all_data.reset_index()
all_data = all_data.drop(columns=['index'])
all_data
all_data = pd.concat([

    all_data,

    pd.get_dummies(all_data.Sex),

    pd.get_dummies(all_data.SmokingStatus)

], axis=1)
all_data = all_data.drop(columns=['Sex', 'SmokingStatus'])
all_data
def scale_feature(series):

    return (series - series.min()) / (series.max() - series.min())



all_data['Percent'] = scale_feature(all_data['Percent'])

all_data['Age'] = scale_feature(all_data['Age'])
feature_columns = [

    'Percent',

    'Age',

    'Female',

    'Male', 

    'Currently smokes',

    'Ex-smoker',

    'Never smoked',

]
train = all_data.loc[all_data.Dataset == 'train']

test = all_data.loc[all_data.Dataset == 'test']

submission = all_data.loc[all_data.Dataset == 'submission']
feature_columns
train[feature_columns].head()
train['FVC']
import sklearn

from sklearn import linear_model
model = linear_model.ARDRegression()
model.fit(train[feature_columns],train['FVC'])
plt.bar(train[feature_columns].columns.values, model.coef_) 

plt.xticks(rotation=90)
predictions = model.predict(train[feature_columns])

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(train['FVC'], predictions) 

print(mae)
train['prediction'] = predictions
plt.scatter(predictions, train['FVC'])
submission[feature_columns].head()
sub_predictions = model.predict(submission[feature_columns])

submission['FVC'] = sub_predictions

submission = submission[['Patient_Week', 'FVC']]



submission['Confidence'] = 285
submission.to_csv('submission11.csv', index=False)