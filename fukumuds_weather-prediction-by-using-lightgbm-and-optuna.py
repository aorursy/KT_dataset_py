# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data.

df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')

df.head()
# get the size of the table.

df.shape
# drop RISK_MM and Date colums.

df = df.drop(columns=['Date', 'RISK_MM'])

# check the size of the table again.

df.shape
# count the NaN values.

df.isnull().sum()
# check categories of Location, WindDirs and RainToday(Tomorrow)

df.groupby('Location').size()
df.groupby('WindGustDir').size()
df.groupby('RainToday').size()
# Do label encode.

import category_encoders as ce

from sklearn_pandas import DataFrameMapper

mapper = DataFrameMapper([

    ('Location', ce.OrdinalEncoder(handle_missing='return_nan')),

    ('WindGustDir', ce.OrdinalEncoder(handle_missing='return_nan')),

    ('WindDir9am', ce.OrdinalEncoder(handle_missing='return_nan')),

    ('WindDir3pm', ce.OrdinalEncoder(handle_missing='return_nan')),

], default=None, df_out=True)

df = mapper.fit_transform(df)

# Check if the data was encoded correctly.

df.head()
# RainToday(Tomorrow) are converted to 0 and 1.

df = df.replace({'RainToday':{'No':0, 'Yes':1}})

df = df.replace({'RainTomorrow':{'No':0, 'Yes':1}})

df.head()
# Convert all data types to float.

df = df.astype(float)

# check data types.

df.dtypes
# Assign feature values to X.

X = df.drop(columns=['RainTomorrow'])

X.shape
# Assign criterion variable(Rain tomorrow) to y.

y = df['RainTomorrow']

y.head()
# Create training data, validation data and test data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=99)



# check each data size.

print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
import lightgbm as lgb



# Convert the data to datasets for LightGBM.

lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)



# set params for binary classification.

lgb_params = {

    'objective': 'binary',

}



# At first, we try a ordinary LightGBM.

model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_valid)
# Check parameters. 

# You can see a parameter only for binary classification.

print(model.params)
# Predict tomorrow's weather.

y_pred = np.round(model.predict(X_test, num_iteration=model.best_iteration))

print("Done!")
# Show confusion matrix.

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))
# Show accuracy, precision, recall and f-1 score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)))

print('precision:{:.2f}'.format(precision_score(y_test, y_pred)))

print('recall:{:.2f}'.format(recall_score(y_test, y_pred)))

print('f-1 score:{:.2f}'.format(f1_score(y_test, y_pred)))
# Use classification report alternatively.

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
# Next, we try optuna with lightgbm_tuner!

#### Before use it, we need latest version of optuna.

#### You need internet connection on you kernel to use pip.

!pip install optuna==0.19.0
# Optimize LightGBM!!

# (It takes about 10 minutes....)

from optuna.integration import lightgbm_tuner

tuned_model = lightgbm_tuner.train(lgb_params, lgb_train,

                                     valid_sets=lgb_valid,

                                     num_boost_round=300,

                                     early_stopping_rounds=30,

                                    )
# Check tuned parameters!

print(tuned_model.params)
# Predict tomorrow's weather by using the tuned model.

tuned_y_pred = np.round(tuned_model.predict(X_test))

print("Done!")
# Show classification report.

from sklearn.metrics import classification_report

print(classification_report(y_test,tuned_y_pred))
# Show importances of the features.

lgb.plot_importance(model, figsize=(20, 20))