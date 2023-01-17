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
test = pd.read_csv("../input/richters-predictor-modeling-earthquake-damage/test_values.csv")

train_label = pd.read_csv("../input/eqdataset/train_labels.csv")

train = pd.read_csv("../input/eqdataset/train_values.csv")

submission = pd.read_csv("../input/eqsub/submission_format.csv")
test
train
train_label
train_dataset = train.merge(train_label, on='building_id')

train_dataset
train_dataset['land_surface_condition']
train_dataset['land_surface_condition'].value_counts()
train_dataset['foundation_type'].value_counts()
train_dataset['roof_type'].value_counts()
train_dataset['ground_floor_type'].value_counts()
train_dataset['other_floor_type'].value_counts()
train_dataset['position'].value_counts()
train_dataset['plan_configuration'].value_counts()
train_dataset['legal_ownership_status'].value_counts()
train_dataset.dtypes
non_number_columns = train.dtypes[train.dtypes == object].index.values
from sklearn.preprocessing import LabelEncoder



for column in non_number_columns:

    le = LabelEncoder()

    train_dataset[column] = le.fit_transform(train_dataset[column]).astype(np.int64)

    test[column] = le.fit_transform(test[column]).astype(np.int64)
test.dtypes
x_train = train_dataset.iloc[:len(train_dataset)*9//10].drop(['damage_grade'], axis=1)

x_val = train_dataset.iloc[len(train_dataset)*9//10:].drop(['damage_grade'], axis=1)



y_train = train_dataset.iloc[:len(train_dataset)*9//10]['damage_grade']

y_val = train_dataset.iloc[len(train_dataset)*9//10:]['damage_grade']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
Y_pred = model.predict(x_val).clip(0, 20)

Y_test = model.predict(test).clip(0, 20)
Y_test
train_dataset.isnull().sum()
train_label
Y_test
Y_sub = Y_test
np.round(Y_sub).astype(int)
Y_test
submission['damage_grade'] = np.round(Y_sub).astype(int)

submission
submission.to_csv('submission_1.csv',index=False)
from IPython.display import display, Image

display(Image(filename='../input/results/Richter Predictor.PNG'))