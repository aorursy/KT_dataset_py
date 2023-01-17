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
import pandas as pd

train = pd.read_csv("../input/train.csv",index_col="timestamp", parse_dates = True, dayfirst=True)



df = train[train.building_number==5]

df = df.drop(['building_number'],axis=1)

df.head()
df.info()
from sklearn.model_selection import train_test_split



Xfeatures = ['main_meter', 'sub_meter_1' ]

X = df[Xfeatures]

y = df['sub_meter_2']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import warnings

X_with_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_constant)



results = model.fit()

results.params
print(results.summary())
from statsmodels.stats.diagnostic import het_breuschpagan

from statsmodels.stats.diagnostic import het_white

import pandas as pd

import statsmodels.api as sm

from statsmodels.formula.api import ols
train = pd.read_csv("../input/train.csv",index_col="timestamp", parse_dates = True, dayfirst=True)

df = train[train.building_number==5]

df = df.drop(['building_number'],axis=1)

df.head()

Xfeatures = ['sub_meter_1', 'sub_meter_2' ]

X = df[Xfeatures]

y = df['main_meter']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_with_constant = sm.add_constant(X_train)

model1 = sm.OLS(y_train, X_with_constant)

results = model1.fit()

results.params
white_test = het_white(results.resid,  results.model.exog)
labels = ['LM Statistic','LM-Test p-value', 'F-Statistic', 'F-Test p-value']

#print(dict(zip(labels, bp_test)))

print(dict(zip(labels, white_test)))



#bp_test = het_breuschpagan(results.resid, [df.sub_meter_1, df.sub_meter_2])