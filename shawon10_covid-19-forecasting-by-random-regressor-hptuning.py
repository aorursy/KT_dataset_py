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

from sklearn.svm import SVR

%matplotlib inline 

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import seaborn as sns
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv').fillna('-')

test= pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv').fillna('-')
train.head()


redate = pd.to_datetime(train['Date'], errors='coerce')

train['Date']= redate.dt.strftime("%Y%m%d").astype(int)

train.head()
targets = train['Target'].unique()

for index in range(0, len(targets)):

    train['Target'].replace(targets[index], index, inplace=True)

    

train.head()
# Get features

feature_cols = ['Population', 'Weight', 'Date', 'Target']

X = train[feature_cols] # Features

y = train['TargetValue'] # Target variable
#X_train= X[100:300]

#X_test= X[300001:]

#y_train=y[100:300000]

#y_test= y[300001:]
plt.plot(y , color = 'blue' , label = 'Covid 19 Prediction')

plt.title('Covid 19 Week 5')

plt.xlabel('Features')

plt.ylabel('Confirmed Cases')

plt.legend()

plt.show()
# Convert string to Date

redate = pd.to_datetime(test['Date'], errors='coerce')

test['Date']= redate.dt.strftime("%Y%m%d").astype(int)
test.head()
targets = test['Target'].unique()

for index in range(0, len(targets)):

    test['Target'].replace(targets[index], index, inplace=True)
#Get features

featureCols = ['Population', 'Weight', 'Date', 'Target']

test = test[featureCols]

test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
# Change n_estimators = 50 because give score > .9

model = RandomForestRegressor(n_jobs=-1, n_estimators = 50)

# Fit on training data

model.fit(X_train, y_train)
mse=mean_squared_error(y_test, model.predict(X_test))

print("MSE: %.4f" % mse)

rmse_rr=np.sqrt(mse)

print("Root Mean Square Error: %.4f" % rmse_rr) #root mean square error
# Score

score = model.score(X_test, y_test)

print("Score: "+ str(score))
predicted=model.predict(X)

real= y

plt.plot(real , color = 'red' , label = 'Real Confirmed Cases')

plt.plot(predicted , color = 'blue' , label = 'Predicted Confirmed Cases')

plt.title('Random Regressor Prediction')

plt.xlabel('Feature')

plt.ylabel('Confirmed Cases')

plt.legend()

plt.show()
from sklearn.model_selection import RandomizedSearchCV 

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
mse=mean_squared_error(y_test, rf_random.predict(X_test))

print("MSE: %.4f" % mse)

rmse_rr=np.sqrt(mse)

print("Root Mean Square Error: %.4f" % rmse_rr) #root mean square error
# Score

score = rf_random.score(X_test, y_test)

print("Score: "+ str(score))
predicted=rf_random.predict(X)

real= y

plt.plot(real , color = 'red' , label = 'Real Confirmed Cases')

plt.plot(predicted , color = 'blue' , label = 'Predicted Confirmed Cases')

plt.title('Random Regressor+ HP Tuning Prediction')

plt.xlabel('Feature')

plt.ylabel('Confirmed Cases')

plt.legend()

plt.show()
predicted=rf_random.predict(test)
# Set Format

listPrediction = [int(x) for x in predicted]

newDF = pd.DataFrame({'number': test.index, 'Population': test['Population'], 'val': listPrediction})
Q05 = newDF.groupby('number')['val'].quantile(q=0.05).reset_index()

Q50 = newDF.groupby('number')['val'].quantile(q=0.5).reset_index()

Q95 = newDF.groupby('number')['val'].quantile(q=0.95).reset_index()



Q05.columns=['number','0.05']

Q50.columns=['number','0.5']

Q95.columns=['number','0.95']
concatDF = pd.concat([Q05,Q50['0.5'],Q95['0.95']],1)

concatDF['number'] = concatDF['number'] + 1

concatDF.head(10)
sub = pd.melt(concatDF, id_vars=['number'], value_vars=['0.05','0.5','0.95'])

sub['ForecastId_Quantile']=sub['number'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head(10)