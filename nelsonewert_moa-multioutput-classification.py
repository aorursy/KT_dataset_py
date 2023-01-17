# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the files

train_data = pd.read_csv('../input/lish-moa/train_features.csv')

test_data = pd.read_csv('../input/lish-moa/test_features.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')

train_target_data = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_target_data_nonescored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
# Exploratory analysis: search for missing values

missing = False

for col in train_data:

    if 'NaN'in col:

        print(col)

        missing = True

if missing == False:

    print('No missing values in training data!')
# Perform t-SNE on g and c columns to try to find a pattern

from sklearn.manifold import TSNE

train_data_TSNE = train_data.select_dtypes(exclude=['object'])

train_data_TSNE.drop('cp_time', axis=1, inplace=True)

train_data_TSNE = TSNE().fit_transform(train_data_TSNE)
# transforming the result into a df

train_data_TSNE = pd.DataFrame(data=train_data_TSNE)

#train_data_TSNE.to_csv('train_TSNE.csv')
import matplotlib.pyplot as plt

plt.figure(figsize = [10.4, 8])

plt.scatter(train_data_TSNE[0], train_data_TSNE[1])
# One-hot encode the categorical columns

type_dummies = pd.get_dummies(train_data['cp_type'], drop_first = True, prefix='cp_type')

time_dummies = pd.get_dummies(train_data['cp_time'], drop_first = True, prefix='cp_time')

dose_dummies = pd.get_dummies(train_data['cp_dose'], drop_first = True, prefix='cp_dose')
# concatenate the columns generated from the t-SNE and the encoded columns



train_data_prep = pd.concat([train_data.drop(['cp_type','cp_time','cp_dose', 'sig_id'], axis=1) ,type_dummies, time_dummies, dose_dummies], axis = 1)
from sklearn.datasets import make_multilabel_classification

from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# perform a train/validation split



lines = len(train_target_data)

train_size = round(lines*0.8)





X_train = train_data_prep[0:train_size]

y_train = train_target_data.drop('sig_id', axis=1)[0:train_size]

X_val = train_data_prep[train_size:lines]

y_val = train_target_data.drop('sig_id', axis=1)[train_size:lines]
# Apply t-SNE on validation set



X_val_TSNE = TSNE().fit_transform(X_val)

X_val_TSNE = pd.DataFrame(data=X_val_TSNE, columns=['TSNE-0', 'TSNE-1'], index=y_val.index)
# fit the model and predict labels for validation data



regressor = MultiOutputRegressor(XGBRegressor())

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_val)
# create a dataset with the predictions

y_pred = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)



# replace negative values wtih zero

y_pred[y_pred < 0] = 0



y_pred.to_csv('validation_predictions.csv')
# calculate MSLE for each row 

error_matrix = []

for i in range (19051, 19051+len(y_val)):

    e = mean_squared_log_error(y_val.loc[i], y_pred.loc[i])

    error_matrix.append(e)





# find rows with 10%, 5% and 1% biggest errors

p10 = np.percentile(a=error_matrix, q=90)

p5 = np.percentile(a=error_matrix, q=95)

p1 = np.percentile(a=error_matrix, q=99)

error=[]

for obj in error_matrix:

    if obj > p1:

        error.append('1%')

    elif obj > p5:

        error.append('5%')

    elif obj > p10:

        error.append('10%')

    else:

        error.append('0')

    
# now let's check where the rows wih the biggest errors are located on the t-SNE

error = pd.DataFrame(error, index=y_val.index, columns=['error'])

X_val_TSNE = pd.concat([X_val_TSNE, error], axis=1)
plt.figure(figsize = [10.4, 8])

sns.lmplot(x='TSNE-0', y='TSNE-1', data=X_val_TSNE, fit_reg=False, hue='error')
global_error = mean_squared_log_error(y_pred, y_val)



print('mean_squared_log_error: ', global_error)
# Prepare the test data to be  predicted



# checking for missing values

missing = False

for col in test_data:

    if 'NaN'in col:

        print(col)

        missing = True

if missing == False:

    print('No missing values in test data!')
# get dummies for categorical columns

type_dummies = pd.get_dummies(test_data['cp_type'], drop_first = True, prefix='cp_type')

time_dummies = pd.get_dummies(test_data['cp_time'], drop_first = True, prefix='cp_time')

dose_dummies = pd.get_dummies(test_data['cp_dose'], drop_first = True, prefix='cp_dose')
X_test = pd.concat([test_data.drop(['cp_type','cp_time','cp_dose', 'sig_id'], axis=1) ,type_dummies, time_dummies, dose_dummies], axis = 1)
# predict the values

y_test = regressor.predict(X_test)

y_test = pd.DataFrame(y_test, columns=y_val.columns)



# replace negative values wtih zero

y_test[y_test < 0] = 0



# put the id column

submission = pd.concat([sample_sub['sig_id'], y_test], axis = 1)



# generate the csv file

submission.to_csv('submission.csv', index=False)