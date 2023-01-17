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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

heart_failure= pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart_failure
test=heart_failure['DEATH_EVENT']

heart_failure.drop(['DEATH_EVENT'], axis=1, inplace=True)

heart_failure
import seaborn as sns

ax = sns.barplot(x="sex", y="age", hue='sex' ,data=heart_failure)
ax = sns.barplot(x=test, y="age" ,data=heart_failure)
ax = sns.lineplot(x="age", y=test ,data=heart_failure)
#ax = sns.lineplot(x="anaemia", y=test ,data=heart_failure)

ax1=sns.barplot(x=test, y="anaemia" ,data=heart_failure)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(heart_failure, test, test_size=0.25, random_state=2)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.metrics import mean_squared_error

random= RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=None,

                              random_state=42, verbose=1)

model1= random.fit(x_train, y_train)

model1.score(x_test, y_test)