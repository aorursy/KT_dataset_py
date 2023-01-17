# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_Train = pd.read_csv('../input/lish-moa/train_features.csv')
df_Test = pd.read_csv('../input/lish-moa/test_features.csv')
df_Train_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_Train_unscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
#To find the head of the Data
df_Train.head()
df_Train_scored.head()
#Columns List
df_Train.columns
#Columns List
df_Train_scored.columns
#Information of the Dataset Continuous Values
df_Train.describe()
#Information of the Dataset Values
df_Train.info()
#Information of the Dataset Values
df_Train_scored.info()
#Shape of the Train and Test Data
print('Shape of Train Data: ', df_Train.shape)
print('Shape of Train Scored Data: ', df_Train_scored.shape)
print('Shape of Train Unscored Data: ', df_Train_unscored.shape)
print('Shape of Test Data: ', df_Test.shape)
#Null values in the Train Dataset
print('Null values in Train Data: \n', df_Train.isnull().sum())
#Null Values in the Test Dataset
print('Null Values in Test Data: \n', df_Test.isnull().sum())
corrmat = df_Train.corr()
f, ax = plt.subplots(figsize=(14,14))
sns.heatmap(corrmat, square=True, vmax=.8)
corrmat = df_Train_scored.corr()
f, ax = plt.subplots(figsize=(14,14))
sns.heatmap(corrmat, square=True, vmax=.8)
corrmat = df_Train_unscored.corr()
f, ax = plt.subplots(figsize=(14,14))
sns.heatmap(corrmat, square=True, vmax=.8)
