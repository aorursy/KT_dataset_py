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
data = pd.read_csv('/kaggle/input/insurance/insurance.csv')

data.head()
from sklearn.preprocessing import LabelEncoder



Lab_enc = LabelEncoder()



data['smoker'] = Lab_enc.fit_transform(data['smoker'])



data['sex'] = Lab_enc.fit_transform(data['sex'])



data['region'] = Lab_enc.fit_transform(data['region'])
data.head()
data.describe()
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(15,5))



sns.lmplot(data = data , x = 'age', y = 'charges', hue = 'smoker')
data.corr()['charges'].sort_values()
good_columns = ['smoker', 'age', 'bmi']



refined_data = pd.DataFrame()



for col in good_columns:

    refined_data[col] = data[col].copy()

    

refined_data.head()
X = refined_data.values

y = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.2, random_state = 42)
from sklearn.linear_model import LinearRegression



model = LinearRegression()



model.fit(X_train, y_train)



model.score(X_valid, y_valid)





from sklearn.metrics import mean_absolute_error



y_pred = model.predict(X_valid)

mean_absolute_error(y_pred, y_valid)



from xgboost import XGBRegressor

def my_model(X, y , Xt, Yt, n):

    model_ = XGBRegressor(n_estimators = n)

    

    model_.fit(X,y)

    

    Yp = model_.predict(Xt)

    

    return mean_absolute_error(Yt,Yp)
nestimators={}

for i in range (3,21):

    scr = my_model(X_train, y_train, X_valid, y_valid, i)

    

    nestimators[i] = scr

    

print(nestimators)

    
my_final_model = XGBRegressor(n_estimators = 9)



my_final_model.fit(X_train, y_train)



my_final_model.score(X_valid, y_valid)