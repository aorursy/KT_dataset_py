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
df_csv = pd.read_csv('/kaggle/input/insurance/insurance.csv')

print(df_csv.info())
df_dummied = pd.get_dummies(df_csv, columns=['sex', 'smoker', 'region'])

df_dummied.drop(["sex_female", "smoker_yes", "region_southwest"],axis=1, inplace=True)

df_dummied.head()
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_dummied, test_size=0.2)
print(train.info())
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(train[['age','bmi','children','sex_male','smoker_no','region_northeast','region_northwest','region_southeast' ]], train['charges'])
print(linear_regression.intercept_)
print(linear_regression.coef_)
linear_regression.score(test[['age','bmi','children','sex_male','smoker_no','region_northeast','region_northwest','region_southeast' ]], test['charges'])