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
## This notebook is created by MOHIT RATHOD (mohitrathod0910@gmail.com) in reference to 
## prelim evaluation process by TerraBlueXT.
## Dataset provided contains two csv files, trsin.csv and test.csv.
## train.csv had four columns named - id, time, feature_1, feature_2.
## We needed to use id, time & feature_1 to predict feature_2.
## I used sklearn Bayesian Ridge Regressor as a classifier.
## importing neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
from sklearn.linear_model import BayesianRidge
## While going through the dataframe, I found that column time is of object datatype
## to perform any action on it, we need to convert it to datetime
## I did this by adding a new column name date_conv with datatype datetime
## again i have extracted hours and minute from time and added it to dataframe in columns hr, min

df = pd.read_csv('../input/into-the-future/train.csv')
df['date_conv'] = pd.to_datetime(df['time'])
df['hr'] = df['date_conv'].dt.hour
df['min'] = df['date_conv'].dt.minute

df_2 = pd.read_csv('../input/into-the-future/test.csv')
df_2['date_conv'] = pd.to_datetime(df_2['time'])
df_2['hr'] = df_2['date_conv'].dt.hour
df_2['min'] = df_2['date_conv'].dt.minute
## plotting the trend

plt.figure()
plt.plot(df['time'], df['feature_2'], 'b')
## I have used hr, min & feature_1 as features for the classifier

X = df[['hr', 'min', 'feature_1']]
y = df['feature_2']
X2 = df_2[['hr', 'min', 'feature_1']]

## preparing the classifier

clf = BayesianRidge()
clf.fit(X, y)
pred = clf.predict(X)
pred2 = clf.predict(X2)
## if we plot predicted value on TRAIN (red) with original value (blue) we clearly see that the 
## classifier is properly genralizing. It does not overfit

plt.figure()
plt.plot(df['time'], df['feature_2'], 'b')
plt.plot(df['time'], pred, 'r')

## This plot shows the trend that will be followed by the predicted value (red) with 
## the original value (blue)

plt.figure()
plt.plot(df['time'], df['feature_2'], 'b')
plt.plot(df['time'], pred, 'r')
plt.plot(df_2['time'], pred2, 'r')

df_2['feature_2'] = pred2
df_2.index = df_2['id']
df_2
## Removing all the unneccesary columns and saving the file

del(df_2['id'])
del(df_2['time'])
del(df_2['hr'])
del(df_2['min'])
del(df_2['date_conv'])
del(df_2['feature_1'])
df_2.to_csv('submission_mohit_rathod.csv')
