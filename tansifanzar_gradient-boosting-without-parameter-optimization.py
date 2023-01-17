# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.metrics import mean_squared_error



from sklearn import ensemble



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

data.head()
data.isnull().sum()
y_train = data.pop('revenue')
y_train.head()
data = data.drop(data.columns[[0, 1,2,3,4]], axis=1)

x_train=data[:]

data.head()
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(x_train, y_train)
from sklearn.ensemble import GradientBoostingRegressor

learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

#learing_rates=[]

train_results = []

test_results = []

#beta=range(.05,1.05,.05)

for eta in learning_rates:

   #learning_rates.append(eta)

   model = GradientBoostingRegressor(learning_rate=eta)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(learning_rates, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('Learning Rate')

plt.show()
from sklearn.ensemble import GradientBoostingRegressor

n_estimators = [ 1, 2, 4, 8, 16, 32, 64, 100, 200, 500, 1000, 2000]

train_results = []

test_results = []

for estimator in n_estimators:

   model = GradientBoostingRegressor(n_estimators=estimator)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('Number of Estimators')

plt.show()
from sklearn.ensemble import GradientBoostingRegressor

#min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

min_samples_splits=[2,3,4,5,6,7,8,9,10]

train_results = []

test_results = []

for min_samples_split in min_samples_splits:

   model = GradientBoostingRegressor(min_samples_split=min_samples_split)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('Min Samples Split')

plt.show()
from sklearn.ensemble import GradientBoostingRegressor

max_depths = [1,2,3,4,5,6,7,8,9,10]

train_results = []

test_results = []

for max_depth in max_depths:

   model = GradientBoostingRegressor(max_depth=max_depth)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('No. of Depth')

plt.show()
from sklearn.ensemble import GradientBoostingRegressor

#min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

min_samples_leafs = [1,2,3,4,5,6,7,8,9,10]

train_results = []

test_results = []

for min_samples_leaf in min_samples_leafs:

   model = GradientBoostingRegressor(min_samples_leaf=min_samples_leaf)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('No. of Samples Leaf')

plt.show()
from sklearn.ensemble import GradientBoostingRegressor

max_features = list(range(1,data.shape[1]))

train_results = []

test_results = []

for max_feature in max_features:

   model = GradientBoostingRegressor(max_features=max_feature)

   model.fit(x_train, y_train)

   from sklearn.metrics import mean_squared_error, r2_score

   model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

   y_predicted_train=model.predict(x_train)

   train_results.append(mean_squared_error(y_train, y_predicted_train))

   print('R2 sq: ',model_score)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_features, train_results, 'b', label="Training MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Mean Squared Error')

plt.xlabel('No. of Feature')

plt.show()
data1 = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

data1.head()
data1=data1.drop(data1.columns[[0, 1,2,3,4]], axis=1)
data1.head()
sample =pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')
params = {'n_estimators': 225, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.25, 'loss': 'ls','max_features':10,

             'min_samples_leaf':2}

model = ensemble.GradientBoostingRegressor(**params)

model.fit(x_train, y_train)

sample["Prediction"] = model.predict(data1)
sample.to_csv('submission.csv', index = False)
train.describe()