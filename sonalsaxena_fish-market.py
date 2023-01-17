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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
fish = pd.read_csv('../input/fish-market/Fish.csv')

fish.head()
fish.isna().mean()
fish.describe()
sns.pairplot(fish)
sns.boxplot(x=fish['Length2']);
fishl2 = fish['Length2']

fishl2_Q1 = fishl2.quantile(0.25)

fishl2_Q3 = fishl2.quantile(0.75)

fishl2_IQR = fishl2_Q3 - fishl2_Q1

fishl2_lowerend = fishl2_Q1 - (1.5*fishl2_IQR)

fishl2_uperend = fishl2_Q3 + (1.5*fishl2_IQR)

fishl2_outlier = fishl2[(fishl2 < fishl2_lowerend) | (fishl2 > fishl2_uperend)]

fishl2_outlier
fish[142:145]
fish = fish.drop([142,143,144])

fish.head()
y = fish['Weight']

X = fish.iloc[:,2:7]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
Predictions = lm.predict(X_test)

from sklearn.metrics import mean_squared_error ,r2_score
r2_score(Predictions,y_test)
import statsmodels.formula.api as smf

lm1 = smf.ols(formula = 'Weight ~ Length1 + Length2 + Length3 + Height + Width',data=fish).fit()

lm1.summary()