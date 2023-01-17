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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as s

%matplotlib inline
import os

print((os.listdir('../input/')))




df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')







#dropping columns after visualization

#F8

df_train.drop(['F1', 'F2', 'F8','F6','F10','F14'], axis = 1, inplace = True)

df_test.drop(['F1', 'F2', 'F8','F6','F10','F14'], axis = 1, inplace = True)
df_train.info()

df_test.info()
df_train.describe()
df_train.head()
s.distplot(df_train['F13'])

s.distplot(df_train['F16'])
s.distplot(df_train['O/P'])
s.distplot(np.log(df_train['O/P']))
s.pairplot(df_train)
df_train.columns
#checking for columns to drop based on hist plots

for i in ['F3', 'F4', 'F5', 'F7', 'F11', 'F12', 'F13', 'F15', 'F16','F17']:

    plt.figure()

    s.catplot(x=i,y='O/P', data=df_train, kind='strip')

    plt.title(i)

    plt.show()
#checking for columns to drop based on hist plots

for i in ['F3', 'F4', 'F5', 'F7', 'F11', 'F12', 'F13', 'F15', 'F16','F17']:

    plt.figure()

    s.catplot(x=i,y='O/P', data=df_train, kind='box')

    plt.title(i)

    plt.show()
for i in ['F3', 'F4', 'F5', 'F7', 'F11', 'F12', 'F13', 'F15', 'F16','F17']:

    plt.figure()

    plt.title(i)

    df_train[i].plot.hist(bins=100)

    plt.show()

#correlation

#Found out that F13-F17 are normalized values, checking each of them for correlations

corr = df_train.corr()

corr.style.background_gradient(cmap='winter').set_precision(2)
#OUTLIER FOUND! 585, 8854, 9123

#df_train[df_train['F12']==4]

df_train.drop([585, 8854, 9123], axis=0, inplace =True)
df_train[df_train['F7']==0].describe()
from sklearn.model_selection import train_test_split
X = df_train.loc[:, 'F3':'F17']

y = df_train.loc[:, 'O/P']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
'''

#best parameters is n_estimators 300, 



#USING GridSearch for optimizing parameters

from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True, False],

    'n_estimators': [25, 50, 100, 200, 250, 300]

}# Create a based model



rf = RandomForestRegressor(random_state=43)# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring ='neg_root_mean_squared_error',

                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data

grid_search.fit(X_train, y_train)

best_grid = grid_search.best_estimator_

print(best_grid)

print(grid_search.best_score_)

'''
rf = RandomForestRegressor(n_estimators=50,random_state=43)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, predictions))

print('MSE: ', metrics.mean_squared_error(y_test, predictions))

print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

df_test = df_test.loc[:, 'F3':'F17']

pred = rf.predict(df_test)

print(pred)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output_only_rf_optimized.csv', index=False)