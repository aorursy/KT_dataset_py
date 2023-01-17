import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns',1000)

from scipy.stats import norm, skew

from scipy.special import boxcox1p

import warnings

warnings.filterwarnings('ignore')
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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.dtypes
df.isnull().sum()
df.drop(columns = ['Serial No.'], inplace = True)
for col in df.columns:

    if col != 'Chance of Admit':

        plt.figure(figsize = (10,5))

        sns.boxplot(x = df[col])

        plt.title(col)

        plt.show()
sns.distplot(df['LOR '])

plt.title('LOR')

plt.show()
q25, q75 = df['LOR '].quantile(0.25), df['LOR '].quantile(0.75)

IQR = q75-q25

cutoff = IQR*1.5

lower, upper = q25-cutoff, q75+cutoff

df = df[(df['LOR ']>lower) & (df['LOR ']<upper)]
plt.figure(figsize=(10,5))

sns.boxplot(df['LOR '])

plt.title('LOR after removing outliers')

plt.show()
plt.figure(figsize=(10,7))

sns.heatmap(df.corr(), 

            annot = True, 

            fmt = '.3g',

           vmax = 1, vmin = -1, center = 0,

           cmap = 'coolwarm',

           square = True)
rel = df.corr()

rel.reset_index(inplace = True)

rel.rename(columns={'index' : 'Factor'}, inplace = True)

rel = rel[['Factor', 'Chance of Admit ']]

rel.drop([7], axis = 0, inplace = True)

rel = rel.sort_values(by='Chance of Admit ', ascending = False)

rel
plt.figure(figsize = (7,5))

sns.barplot(x='Chance of Admit ',

                 y='Factor',

                 data=rel)
plt.scatter(x = df['University Rating'],

          y = df['Chance of Admit '])

plt.title('Rating Vs Chance of Admission')
df.head()
plt.scatter(x = df['GRE Score'],

           y = df['TOEFL Score'])

plt.title('GRE Vs TOEFL Score')
bins = [200, 300, 340]

name = ['<300', '>=300']

df['GRERange'] = pd.cut(df['GRE Score'], bins, labels=name)

plt.scatter(x = df['GRERange'],

           y = df['Chance of Admit '])
S_300 = df[df['GRE Score'] >= 300]

mean_300 = S_300['Chance of Admit '].mean()

S_sub300 = df[df['GRE Score'] < 300]

mean_sub300 = S_sub300['Chance of Admit '].mean()

likelihood_GREScore = mean_300/mean_sub300

likelihood_GREScore
df[['Research', 'Chance of Admit ']].groupby('Research').mean()
pie = df[['Research', 'Chance of Admit ']].groupby('Research').count().reset_index().rename(columns = {'index' : 'Research', 'Chance of Admit ' : 'Count'})

pie
labels = ['Yes', 'No']

plt.pie(pie['Count'], 

        labels=labels,

        autopct='%.2f%%')

plt.title('Candidates who have done research vs those who haven\'t')

plt.show()
df[['University Rating', 'Chance of Admit ']].groupby('University Rating').mean()
pie = df[['University Rating', 'Chance of Admit ']].groupby('University Rating').count().reset_index().rename(columns = {'index' : 'University Rating', 'Chance of Admit ' : 'Count'})

plt.pie(pie['Count'], 

       labels = pie['University Rating'],

       autopct = '%.2f%%')

plt.title('No. of Universities of different rating')

plt.show()
df[['LOR ', 'Chance of Admit ']].groupby('LOR ').mean()
pie = df[['LOR ', 'Chance of Admit ']].groupby('LOR ').count().reset_index().rename(columns = {'index' : 'LOR', 'Chance of Admit ' : 'Count'})

plt.pie(pie['Count'],

       labels=pie['LOR '],

       autopct='%.2f%%')

plt.title('%age of candidates getting various LOR scores')

plt.show()
df[['SOP', 'Chance of Admit ']].groupby('SOP').mean()
pie = df[['SOP', 'Chance of Admit ']].groupby('SOP').count().reset_index().rename(columns={'index' : 'SOP', 'Chance of Admit ' : 'Count'})

plt.pie(pie['Count'],

       labels=pie['SOP'],

       autopct='%.2f%%')

plt.title('%age of candidates getting various SOP scores')

plt.show()
bins = [0, 100, 120]

names = ['<100', '>=100']

df['TOEFLRange'] = pd.cut(df['TOEFL Score'], bins, labels = names)

df[['TOEFLRange', 'Chance of Admit ']].groupby('TOEFLRange').mean()
pie = df[['TOEFLRange', 'Chance of Admit ']].groupby('TOEFLRange').count().reset_index().rename(columns={'index' : 'TOEFLRange', 'Chance of Admit ' : 'Count'})

plt.pie(pie['Count'],

       labels=pie['TOEFLRange'],

       autopct='%.2f%%')

plt.title('>=300 GRE Score Vs <300 GRE Score')

plt.show()
bins = [0, 300, 320]

names = ['<300', '>=300']

df['GRERange'] = pd.cut(df['GRE Score'], bins, labels = names)

df[['GRERange', 'Chance of Admit ']].groupby('GRERange').mean()
pie = df[['GRERange', 'Chance of Admit ']].groupby('GRERange').count().reset_index().rename(columns={'index' : 'GRERange', 'Chance of Admit ' : 'Count'})

plt.pie(pie['Count'],

       labels=pie['GRERange'],

       autopct='%.2f%%')

plt.title('>=300 GRE Score Vs <300 GRE Score')

plt.show()
df.drop(columns = ['TOEFLRange', 'GRERange', 'Research'], inplace=True)
X = df.drop(columns = ['Chance of Admit '])

y = df['Chance of Admit ']
for col in X.columns:

    X[col] = (X[col]-X[col].min())/(X[col].max()-X[col].min())
from sklearn.metrics import SCORERS

SCORERS.keys()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

lin = LinearRegression()

r2score=cross_val_score(lin, X, y, cv=10, scoring='r2')

mse = cross_val_score(lin, X, y, cv=10, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Linear_rmse = np.sqrt(np.abs(mse.mean()))

Linear_r2score = r2score.mean()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_split=20, min_samples_leaf=9)

r2score = cross_val_score(rfr, X, y, cv=10, scoring = 'r2')

mse = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Rfr_rmse = np.sqrt(np.abs(mse.mean()))

Rfr_r2score = r2score.mean()
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(learning_rate=0.03, 

                                n_estimators=80, 

                                min_samples_split=2, 

                                min_samples_leaf=9, 

                                max_depth=3)

r2score = cross_val_score(gbr, X, y, cv=10, scoring = 'r2')

score = cross_val_score(gbr, X, y, cv=10, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Gbr_rmse = np.sqrt(np.abs(mse.mean()))

Gbr_r2score = r2score.mean()
from sklearn.model_selection import GridSearchCV

params = {

    'learning_rate' : [0.03],

    'n_estimators' : [80],

    'min_samples_split' : [2],

    'min_samples_leaf' : [9],

    'max_depth' : [3, 4, 5]

}

model = GradientBoostingRegressor()

grid = GridSearchCV(estimator=model, param_grid = params)

grid.fit(X,y)

grid.best_estimator_
from sklearn.ensemble import AdaBoostRegressor

reg = AdaBoostRegressor()

r2score = cross_val_score(reg, X, y, cv=10, scoring = 'r2')

score = cross_val_score(reg, X, y, cv = 10, scoring = 'neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Abr_rmse = np.sqrt(np.abs(mse.mean()))

Abr_r2score = r2score.mean()
from sklearn.linear_model import Ridge

rid=Ridge(alpha=0.01)

r2score = cross_val_score(rid, X, y, cv=10, scoring = 'r2')

score = cross_val_score(rid, X, y, cv = 10, scoring = 'neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Ridge_rmse = np.sqrt(np.abs(mse.mean()))

Ridge_r2score = r2score.mean()
from sklearn.model_selection import GridSearchCV

alphas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20]

reg=Ridge()

grid=GridSearchCV(estimator=reg,

                 param_grid=dict(alpha=alphas))

grid.fit(X,y)

grid.best_estimator_
params = {'n_estimators' : [70, 80, 90],

         'max_depth' : [5, 10, 15, 20],

         'min_samples_split' : [10,12,14,16, 20, 25],

         'min_samples_leaf' : [ 5, 7, 9, 11],

         }

model = RandomForestRegressor()

grid = GridSearchCV(estimator = model, param_grid = params)

grid.fit(X,y)

grid.best_estimator_
from sklearn.svm import SVR

svr = SVR(kernel='linear', C=13, gamma=1e-09, epsilon=0.1)

r2score = cross_val_score(svr, X, y, cv=10, scoring = 'r2')

score = cross_val_score(svr, X, y, cv = 10, scoring = 'neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

SVR_rmse = np.sqrt(np.abs(mse.mean()))

SVR_r2score = r2score.mean()
params = {

    'kernel' : ['linear'],

    'C' : [12, 13, 15, 17],

    'gamma' : [1e-9, 1e-8, 1e-7, 1e-6],

    'epsilon' : [0.1, 0.2, 0.3, 0.4]

}

model = SVR()

grid = GridSearchCV(estimator=model, param_grid=params)

grid.fit(X,y)

grid.best_estimator_
from mlxtend.regressor import StackingRegressor

streg = StackingRegressor(regressors=[lin, rfr, gbr, svr], 

                           meta_regressor=lin,

                         use_features_in_secondary=True)

r2score = cross_val_score(streg, X, y, cv=10, scoring = 'r2')

score = cross_val_score(streg, X, y, cv = 10, scoring = 'neg_mean_squared_error')

print(np.sqrt(np.abs(mse.mean())))

print(r2score.mean())

Stacking_rmse = np.sqrt(np.abs(mse.mean()))

Stacking_r2score = r2score.mean()
result = pd.DataFrame({'Algorithm' : ['Linear Regression', 'Random Forest Regression', 'Gradient Boosting', 'AdaBoost Regression', 'Ridge Regression', 'SVR', 'Stacking Regression'], 'r2 Score' : [Linear_r2score, Rfr_r2score, Gbr_r2score, Abr_r2score, Ridge_r2score, SVR_r2score, Stacking_r2score]})

result = result.sort_values(by='r2 Score', ascending = False)

result
result = pd.DataFrame({'Algorithm' : ['Linear Regression', 'Random Forest Regression', 'Gradient Boosting', 'AdaBoost Regression', 'Ridge Regression', 'SVR', 'Stacking Regression'], 'RMSE' : [Linear_rmse, Rfr_rmse, Gbr_rmse, Abr_rmse, Ridge_rmse, SVR_rmse, Stacking_rmse]})

result = result.sort_values(by='RMSE', ascending = True)

result