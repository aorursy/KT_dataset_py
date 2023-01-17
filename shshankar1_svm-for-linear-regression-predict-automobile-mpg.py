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
import pandas as pd



print(pd.__version__)
auto_data = pd.read_csv('/kaggle/input/auto-mpg.data', delim_whitespace = True, header = None,

                       names = [

                                'mpg',

                                'cylinders',

                                'displacement',

                                'horsepower',

                                'weight',

                                'aceeleration',

                                'model',

                                'origin',

                                'car_name'

    ])
auto_data.head()
auto_data.info()
auto_data.describe()
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')
auto_data.info()
auto_data['car_name'].nunique()
auto_data = auto_data.drop(['car_name'], axis=1)
auto_data.head()
auto_data_nan = auto_data[auto_data.isnull().any(axis=1)]

auto_data_nan.head(10)
auto_data_final = auto_data.dropna(axis=0)

auto_data_final[auto_data_final.isnull().any(axis=1)]
from sklearn.model_selection import train_test_split



X = auto_data_final.drop('mpg', axis=1)

y = auto_data_final['mpg']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state =0)
from sklearn.svm import SVR



model = SVR(kernel='linear', C=1.0)

model.fit(X_train, y_train)
model.coef_
y_predict = model.predict(X_test)
from sklearn.metrics import mean_squared_error



model_mse = mean_squared_error(y_predict, y_test)

print(model_mse)
# Check the correlation matrix to derive horsepower feature by help of other feature

corr = auto_data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(4)
auto_data_4_cylinders = auto_data[auto_data['cylinders'] ==4]

print(len(auto_data_4_cylinders))

auto_data_4_cylinders.head()
%matplotlib inline



auto_data_4_cylinders['horsepower'].plot.hist(bins=10, alpha=0.5)
import numpy as np

from sklearn.impute import SimpleImputer



imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
auto_data_4_cylinders['horsepower'] = imp_mean.fit_transform(auto_data_4_cylinders[['horsepower']])
auto_data_4_cylinders[auto_data_4_cylinders.isnull().any(axis=1)].head()
auto_data_6_cylinders = auto_data[auto_data['cylinders']==6]

auto_data_6_cylinders.head()
%matplotlib inline

auto_data_6_cylinders['horsepower'].plot.hist(bins=10, alpha=0.5)
auto_data_6_cylinders[auto_data_6_cylinders['horsepower']< 160]['horsepower'].plot.hist(bins=10, alpha=0.5)
auto_data_6_cylinders[auto_data_6_cylinders.isnull().any(axis=1)].head()
import numpy as np

from sklearn.impute import SimpleImputer



mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imp.fit(auto_data_6_cylinders[auto_data_6_cylinders['horsepower'] < 160][['horsepower']])



auto_data_6_cylinders['horsepower'] = mean_imp.transform(auto_data_6_cylinders[['horsepower']])
auto_data_6_cylinders[auto_data_6_cylinders.isnull().any(axis=1)]
auto_data_others = auto_data[~auto_data['cylinders'].isin((4,6))]

print(len(auto_data_others))
auto_data_final = pd.concat([auto_data_others, auto_data_4_cylinders, auto_data_6_cylinders], axis=0)

print(len(auto_data_final))
# Uncomment below if you want to drop the rows rather than data imputation

# auto_data_final = auto_data.dropna(axis=0)
auto_data_final[auto_data_final.isnull().any(axis=1)]
print(len(auto_data_final))

auto_data_final.head()
from sklearn.model_selection import train_test_split



X = auto_data_final.drop('mpg', axis=1)

y = auto_data_final['mpg']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state =0)
from sklearn.svm import SVR



model = SVR(kernel='linear', C=1.0)

model.fit(X_train, y_train)
model.coef_
model.score(X_train, y_train)
y_predict = model.predict(X_test)
%pylab inline

pylab.rcParams['figure.figsize'] = (15, 6)



plt.plot(y_predict, label='Predicted')

plt.plot(y_test.values, label='Actual')

plt.ylabel('MPG')



plt.legend()

plt.show()
model.score(X_test, y_test)
from sklearn.metrics import mean_squared_error



model_mse = mean_squared_error(y_predict, y_test)

print(model_mse)