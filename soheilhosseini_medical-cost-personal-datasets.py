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

import pandas as pd

import numpy as np

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



from sklearn.model_selection import cross_val_score

insurance = pd.read_csv('/kaggle/input/insurance/insurance.csv')
insurance.head()
insurance.info()
insurance.describe()
insurance.describe(include='O')
sns.boxplot(data=insurance, x='children')

plt.show()
sns.boxplot(data=insurance, y='bmi')

plt.show()
sns.boxplot(data=insurance, x='age')
plt.figure(figsize = (30,30))

insurance.hist()
insurance.corr()
sns.pairplot(insurance, hue='smoker', palette="husl")

plt.show()
#Correlation Matrix

fig = plt.figure(figsize=[10,5])

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(insurance.corr(), cmap='plasma' )

plt.show()
insurance['children'] = np.log1p(insurance['children'])
(np.sqrt(insurance['children'])).hist()
print((insurance['age']).skew())
plt.figure(figsize = (10,10))

plt.xlim(0,100)

insurance.age.hist()

plt.show()
insurance.info()
ins = pd.get_dummies(data = insurance, drop_first = True )
ins.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_new = scaler.fit_transform(ins.drop(columns = ['charges']))
y=ins['charges']
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1, n_estimators=200)).fit(X_new, y)

X_new = selector.transform(X_new)
X_new = pd.DataFrame(data= X_new)
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(

    X_new, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(max_depth =3, random_state=0)

regressor.fit(X_train, y_train)

regressor.score(X_test, y_test)
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=10)

#neigh.fit(X_train, y_train)

#neigh.score(X_test, y_test)



scores = cross_val_score(neigh, X_new, y, cv=10)

scores
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

#reg.score(X_test, y_test)



scores = cross_val_score(reg, X_new, y, cv=10)

scores
from sklearn.linear_model import Ridge

clf = Ridge(alpha=.1)

#clf.fit(X_train, y_train)

#clf.score(X_test, y_test)



scores = cross_val_score(clf, X_new, y, cv=10)

scores

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor( n_estimators=200, max_depth=2, random_state=0)

#regr.fit(X_train, y_train)

#regr.score(X_test, y_test)



scores = cross_val_score(regr, X_new, y, cv=10)

scores
from sklearn.ensemble import GradientBoostingRegressor

clfg = GradientBoostingRegressor(n_estimators=200, max_depth=2,

                                learning_rate=.1, min_samples_leaf=9,

                                min_samples_split=9)

#clfg.fit(X_train, y_train)

#clfg.score(X_test, y_test)



scores = cross_val_score(clfg, X_new, y, cv=10)

scores
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(learning_rate=0.5, random_state=0, n_estimators=200)

scores = cross_val_score(ada, X_new, y, cv=5)

scores