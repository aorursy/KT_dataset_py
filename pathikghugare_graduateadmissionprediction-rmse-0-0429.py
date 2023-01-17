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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df.info()
df = df.drop(['Serial No.'], axis = 1 )

df.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
columns = ['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']

X = df[columns]

y = df['Chance of Admit ']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.20,shuffle = False)

len(train_X)
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)



model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]
def score_model(model, X_t = train_X, y_t = train_y, X_v = val_X, y_v = val_y):

    model.fit(X_t, y_t)

    return  np.sqrt(mean_squared_error(y_v, model.predict(X_v)))
for i in range(len(models)) :

    print("RMSE of model",i+1,":",score_model(models[i]))
from sklearn.linear_model import LinearRegression

model_Lreg = LinearRegression()

model_Lreg.fit(train_X, train_y)

Lreg_y = model_Lreg.predict(val_X)

print("RMSE of Linear regression model : ",np.sqrt(mean_squared_error(val_y, Lreg_y)))