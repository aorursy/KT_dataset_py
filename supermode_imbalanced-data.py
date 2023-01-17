# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sklearn

import scipy.stats as stats

import keras

import imblearn 

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
plt.style.use('ggplot')

df = pd.read_csv('../input/creditcard.csv')

df.sample(5)
sns.countplot('Class', data = df)

plt.title("No Fraud (0) and Fraud (1)")
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 

      'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',

       'Amount']]

y = df['Class']
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state = 0)

X_resampled, y_resampled = ros.fit_resample(X, y)

# using Counter to display results of naive oversampling

from collections import Counter

print(sorted(Counter(y_resampled).items()))
from imblearn.over_sampling import SMOTE



# applying SMOTE to our data and checking the class counts

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

print(sorted(Counter(y_resampled).items()))
from imblearn.over_sampling import ADASYN



# applying ADASYN

X_resampled, y_resampled = ADASYN().fit_resample(X, y)

print(sorted(Counter(y_resampled).items()))