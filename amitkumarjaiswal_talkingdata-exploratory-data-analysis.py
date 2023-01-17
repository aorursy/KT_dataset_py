import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
!pip install lightgbm
import lightgbm as lgb
DATA_PATH = r"../input"

def load_data(data_path=DATA_PATH):
    # PATHS TO FILE
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")
    ssize = 50000000
    return pd.read_csv(train_path,nrows=ssize), pd.read_csv(test_path)

train, test = load_data()
# Training sample
print(train.shape)
train.head()
# Plot the proportion of clicks that converted into a download or not
plt.figure(figsize=(6,6))
mean = (train.is_attributed.values == 1).mean()
ax = sns.barplot(['Converted (1)', 'Not Converted (0)'], [mean, 1-mean])
ax.set(ylabel='Proportion', title='Proportion of clicks converted into app downloads')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")
# Separate the 2 classes
train_0 = train[train['is_attributed'] == 0]
train_1 = train[train['is_attributed'] == 1]
print(len(train_1))
print(train_0.shape)
print(train.shape)
train['is_attributed'].value_counts()
# Undersample class 0 (without replacement)
train0_undersampled = resample(train_0, replace=False, n_samples=len(train_1), random_state=142)
# Combine minority class with downsampled majority class
train_us = pd.concat([train0_undersampled, train_1])
 
# Display new class counts
train_us.is_attributed.value_counts()
# Extract features from click_time
def ppClicktime(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['wday'] = df['click_time'].dt.dayofweek
    df['hour'] = df['click_time'].dt.hour
    return df
train_pp = ppClicktime(train)
test_pp = ppClicktime(test)
# Drop click_time
train_pp.drop('click_time', axis = 1, inplace = True)
test_pp.drop('click_time', axis = 1, inplace = True)
print(len(test_pp))
test_pp.head()
# Write to csv
train_pp.to_csv("train_pp.csv",index=None)