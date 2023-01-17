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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
cars = pd.read_csv('/kaggle/input/belarus-used-cars-prices/cars.csv')
cars.head(5)
cars.info()
cars['model'] = pd.to_numeric(cars['model'], errors='coerce')
#Pre processing
cars['drive_unit'] = cars['drive_unit'].astype('str')
cars['segment'] = cars['segment'].astype('str')
cars.describe().T
cars.columns
cat_columns = ['make', 'model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']
dataset = pd.get_dummies(cars, columns=cat_columns, drop_first=True)
dataset.head(3)
Q1 = dataset['priceUSD'].quantile(0.25)
Q2 = dataset['priceUSD'].quantile(0.75)
IQR = Q2 - Q1
LL = Q1-IQR*1.5
UL = Q2+IQR*1.5
dataset = dataset.loc[(dataset['priceUSD']>LL) & (dataset['priceUSD']<UL)]
dataset.head(3)
sns.boxplot(dataset['priceUSD'])
sns.distplot(dataset['priceUSD'])
dataset.isnull().sum()
dataset['volume(cm3)'].fillna(dataset['volume(cm3)'].median(), inplace=True)
dataset.describe()
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
print("Root mean squared error : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R square : ", r2_score(y_test, y_pred))
residuals = y_test - y_pred
from scipy import stats
stats.probplot(residuals, plot=plt)
plt.show()
sns.distplot(residuals)
np.mean(residuals)
import statsmodels.api as sm
acf = sm.graphics.tsa.plot_acf(residuals)
acf.show()
