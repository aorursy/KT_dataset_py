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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression
insurance = pd.read_csv('../input/insurance/insurance.csv')

insurance
insurance['sex'] = insurance['sex'].map({'female':1, 'male':0})

insurance['smoker'] = insurance['smoker'].map({'yes':1, 'no':0})

insurance
insurance_new = insurance.copy()
x = insurance_new[['age', 'sex', 'bmi', 'children', 'smoker']]

y = insurance_new['charges']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x)
x_scaled = scaler.transform(x)

x_scaled
reg = LinearRegression()

reg.fit(x_scaled,y)
reg.coef_
reg.intercept_
reg.score(x_scaled,y)
r2 = reg.score(x_scaled,y)

n = x.shape[0]



p = x.shape[1]



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2

from sklearn.feature_selection import f_regression



f_regression(x_scaled,y)
p_values = f_regression(x_scaled,y)[1]

p_values
p_values.round(3)
reg_summary = pd.DataFrame([['Intercept'],['age'], ['sex'], ['bmi'], ['children'], ['smoker']], columns = ['Features'])

reg_summary['weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3], reg.coef_[4]
reg_summary
new_data = pd.DataFrame(data=[[19,1,27.900,0,1],[50,0,30.970,3,0]], columns =['age','sex','bmi','children','smoker'])

new_data
reg.predict(new_data)
new_data_scaled = scaler.transform(new_data)

new_data_scaled
reg.predict(new_data_scaled)