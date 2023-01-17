# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(rc={'figure.figsize':(10,10)})



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw = pd.read_csv("/kaggle/input/chilled-eda/chilled_water_cleaned.csv", index_col = "timestamp", parse_dates = True)

raw.head(3)
raw = raw.resample("W").mean()
raw.head(3)
col = raw.columns

print(col[400:])
#creating dataframe for a single site



peacock = pd.DataFrame()

P = [col for col in raw.columns if 'Peacock' in col]

peacock[P] = raw[P]



moose = pd.DataFrame()

M = [col for col in raw.columns if 'Moose' in col]

moose[M] = raw[M]

 

bull = pd.DataFrame()

B = [col for col in raw.columns if 'Bull' in col]

bull[B] = raw[B]



hog = pd.DataFrame()

H = [col for col in raw.columns if 'Hog' in col]

hog[H] = raw[H]



eagle = pd.DataFrame()

E = [col for col in raw.columns if 'Eagle' in col]

eagle[E] = raw[E]



cockatoo = pd.DataFrame()

C = [col for col in raw.columns if 'Cockatoo' in col]

cockatoo[C] = raw[C]



panther = pd.DataFrame()

pan = [col for col in raw.columns if 'Panther' in col]

panther[pan] = raw[pan]



fox = pd.DataFrame()

f = [col for col in raw.columns if 'Fox' in col]

fox[f] = raw[f]



bobcat = pd.DataFrame()

bob = [col for col in raw.columns if 'Bobcat' in col]

bobcat[bob] = raw[bob]



crow = pd.DataFrame()

cr = [col for col in raw.columns if 'Crow' in col]

crow[cr] = raw[cr]



sites = [peacock, moose, bull, hog, eagle, cockatoo, panther, fox, bobcat, crow]
panther.head(3)
crow.head(2)
eagle.head(2)
bobcat.head(2)
panther.plot()
eagle.plot()
#summing the total chilled water consumption per week

for site in sites:

    site["Chilled_sum"] = site.sum(axis = 1)
#checking sum correct

eagle.head(3)
bull.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet



from sklearn import metrics



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import SCORERS



import datetime as dt



from sklearn.tree import DecisionTreeRegressor



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#splitting df for a site into train and test sets

model_panther = panther.copy()



train = model_panther.iloc[0:(len(model_panther)-30)]

test = model_panther.iloc[len(train):(len(model_panther)-1)]
test.plot()
#checking if stationary or not 

from statsmodels.tsa.stattools import adfuller



result = adfuller(panther["Chilled_sum"])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))
import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX

from matplotlib import pyplot as plt
endog = train["Chilled_sum"]



mod = sm.tsa.statespace.SARIMAX(endog=endog)

model_fit = mod.fit()

model_fit.summary()
train['Chilled_sum'].plot(figsize=(25,10))

model_fit.fittedvalues.plot()

plt.show()
predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1)

test['predicted'] = predict.values

test.tail(5)

test['predicted'].plot(color = 'red')

test["Chilled_sum"].plot()