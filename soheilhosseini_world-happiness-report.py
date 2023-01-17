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
# Importing the Libraries¶

import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy import stats
wh17 = pd.read_csv('../input/world-happiness/2017.csv')
wh17.head()

wh17.hist(figsize=(20,20))
wh17.info()
wh17.describe()
plt.xlim(0,12)

sns.distplot((wh17['Happiness.Score']))
print((wh17['Happiness.Score']).skew())
np.max(wh17['Happiness.Score'])
wh17.corr()['Happiness.Score']
#Correlation Matrix
fig = plt.figure(figsize=[10,10])
plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(wh17.corr().abs()>0.9, annot = True, square=True,linecolor='white',cmap='coolwarm', )
sns.boxplot(x = 'Whisker.high',data=wh17)
wh17.columns
sns.boxplot(x = 'Dystopia.Residual',data=wh17)
sns.boxplot(x = 'Health..Life.Expectancy.',data=wh17)
sns.boxplot(x = 'Family',data=wh17)
sns.boxplot(x = 'Economy..GDP.per.Capita.',data=wh17)
sns.boxplot(x = 'Whisker.low',data=wh17)
wh17.columns
wh17.corr()['Happiness.Score']
wh17.drop(columns = ['Country','Happiness.Rank','Whisker.low'], inplace = True)
wh17.columns
wh17.info()
y = wh17['Happiness.Score']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(wh17.drop(columns = 'Happiness.Score'))
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1, n_estimators=100)).fit(X_new, y)
X_new = selector.transform(X_new)
X_new = pd.DataFrame(data= X_new)
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=8)
neigh.fit(X_train, y_train)
neigh.score(X_test, y_test)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test)
reg.score(X_train, y_train)
