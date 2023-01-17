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
import matplotlib.pyplot as plt

import seaborn as sns
data= pd.read_csv('/kaggle/input/insurance/insurance.csv')

data.head()
data.describe(include='all')
from sklearn.preprocessing import OneHotEncoder
new_data = pd.get_dummies(data, prefix='sex', columns=['sex'],drop_first=True)

new_data.head()
new_data=pd.get_dummies(new_data, columns=['smoker','region'],drop_first=True)

new_data.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
new_data.columns
X = new_data[['age', 'bmi', 'children', 'sex_male', 'smoker_yes','region_northwest', 'region_southeast', 'region_southwest']]

y = new_data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)
pred = model.predict(X_test)
model.score(X_test, y_test)

print(r2_score(y_test, pred))
mean_squared_error(y_test, pred)