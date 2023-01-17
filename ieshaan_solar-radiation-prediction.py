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
df=pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')

df.head()
df.shape
df.describe()
df.isna().sum()
import matplotlib.pyplot as plt

import seaborn as sns

df['Radiation'].plot()
solar_data=df.drop(['UNIXTime','Data','Time','TimeSunRise','TimeSunSet'],axis=1)

print(solar_data)
import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X = solar_data.drop(columns=['Radiation'],axis=1)  #independent columns

y = solar_data['Radiation']    #target column i.e price range

y = y.astype('int')

#apply SelectKBest class to extract top 5 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
import xgboost as xgb

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.head()
y_train.head()
from sklearn.linear_model import LinearRegression

regresor= LinearRegression()

regresor.fit(X_train, y_train)

regresor_pred = regresor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)

randomforest_pred= rf_reg.predict(X_test)
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)

svm_pred= regressor.predict(X_test)
from sklearn.metrics import r2_score

print(r2_score(y_test, regresor_pred))

print(r2_score(y_test, randomforest_pred))

print(r2_score(y_test, svm_pred))