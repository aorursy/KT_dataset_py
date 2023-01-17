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
from tpot import TPOTRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
housing = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')  
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
housing.head()
housing.shape
housing['date'] = housing['date'].apply(lambda x: x.split('T')[0])
X = housing.drop(['id', 'price'], axis=1).values.astype(np.float)
y = housing.iloc[:, 2].values.astype(np.float)
X.shape
housing.isnull().any().any()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, test_size=0.33, random_state=42)
tpot = TPOTRegressor(generations=3, population_size=25, verbosity=3)
tpot.fit(X_train, y_train)
tpot.export('tpot_exported_pipeline.py')
print("Training neg_mean_squared_error: {:.4f}".format(tpot.score(X_train, y_train)))
print("Testing neg_mean_squared_error:  {:.4f}".format(tpot.score(X_test, y_test)))
mean_absolute_error(y_test, tpot.predict(X_test))
tpot.fitted_pipeline_
housing['price'].hist()
housing['price'].min()
housing['price'].max()
def number_of_houses_well_predicted(Y_pred, Y):
    nb_houses = 0
    for i in range(0, len(Y_pred)):
        if abs(Y_pred[i] - Y[i]) < Y_pred[i]*0.05:
            nb_houses += 1
    return nb_houses
number_of_houses_well_predicted(tpot.predict(X_test),y_test)
number_of_houses_well_predicted(tpot.predict(X_test),y_test)/X.shape[0]