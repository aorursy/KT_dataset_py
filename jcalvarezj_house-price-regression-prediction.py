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
house_df = pd.read_csv('/kaggle/input/housepricing/HousePrices_HalfMil.csv')

house_df.head()
house_df.dtypes
house_df.isna().any()
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



X = house_df.drop(['Prices'], axis = 1)

y = house_df[['Prices']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



linear_regression = sm.OLS(y_train, sm.add_constant(X_train)).fit()



y_predict = linear_regression.predict(sm.add_constant(X_test))



print('R2: ', r2_score(y_test, y_predict))
linear_regression.params
def calculate_prediction(area, garage, fireplace, baths, white_mar, black_mar, indian_mar, floors, city, solar, electric, fiber, glass_doors, pool, garden):

    X_test = [area, garage, fireplace, baths, white_mar, black_mar, indian_mar, floors, city, solar, electric, fiber, glass_doors, pool, garden]

    

    result = linear_regression.params[0]

    

    for i, x in enumerate(X_test):

        result += linear_regression.params[i+1] * x

    

    return result





prediction = calculate_prediction(150, 1, 0, 3, 1, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1)



print(f'The expected price for the above described house is of ${prediction:.2f}')