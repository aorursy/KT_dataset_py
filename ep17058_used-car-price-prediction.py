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
df_train = pd.read_csv('/kaggle/input/1056lab-used-cars-price-prediction/train.csv').drop('Id', axis=1)

df_test = pd.read_csv('/kaggle/input/1056lab-used-cars-price-prediction/test.csv').drop('Id', axis=1)
df_train_dummies = pd.get_dummies(df_train, columns=['Transmission'], drop_first=True)

df_test_dummies = pd.get_dummies(df_test, columns=['Transmission'], drop_first=True)

df_train_dummies = df_train_dummies.drop(columns=['Name'])

df_train_dummies = df_train_dummies.drop(columns=['New_Price'])

df_train_dummies = df_train_dummies.drop(columns=['Location'])

df_test_dummies = df_test_dummies.drop(columns=['Name'])

df_test_dummies = df_test_dummies.drop(columns=['New_Price'])

df_test_dummies = df_test_dummies.drop(columns=['Location'])

df_train_dummies['Fuel_Type'] = df_train_dummies['Fuel_Type'].map({'Diesel':1,'Petrol':2,'CNG':3,'LPG':4})

df_train_dummies['Owner_Type'] = df_train_dummies['Owner_Type'].map({'First':1,'Second':2,'Third':3,'Fourth & Above':4})

df_test_dummies['Fuel_Type'] = df_test_dummies['Fuel_Type'].map({'Diesel':1,'Petrol':2,'CNG':3,'LPG':4,'Electric':5})

df_test_dummies['Owner_Type'] = df_test_dummies['Owner_Type'].map({'First':1,'Second':2,'Third':3,'Fourth & Above':4})
df_train_dummies['Mileage'] = df_train_dummies['Mileage'].str.strip(' kmpl')

df_train_dummies['Mileage'] = df_train_dummies['Mileage'].str.strip(' km/kg')

df_train_dummies['Engine'] = df_train_dummies['Engine'].str.strip(' CC')

df_train_dummies['Power'] = df_train_dummies['Power'].str.strip(' bhp')

df_test_dummies['Mileage'] = df_test_dummies['Mileage'].str.strip(' kmpl')

df_test_dummies['Mileage'] = df_test_dummies['Mileage'].str.strip(' km/kg')

df_test_dummies['Engine'] = df_test_dummies['Engine'].str.strip(' CC')

df_test_dummies['Power'] = df_test_dummies['Power'].str.strip(' bhp')

df_train_dummies['Mileage'] = pd.to_numeric(df_train_dummies['Mileage'], errors='coerce')

df_test_dummies['Mileage'] = pd.to_numeric(df_test_dummies['Mileage'], errors='coerce')

df_train_dummies['Engine'] = pd.to_numeric(df_train_dummies['Engine'], errors='coerce')

df_train_dummies['Power'] = pd.to_numeric(df_train_dummies['Power'], errors='coerce')

df_train_dummies['Seats'] = pd.to_numeric(df_train_dummies['Seats'], errors='coerce')

df_test_dummies['Engine'] = pd.to_numeric(df_test_dummies['Engine'], errors='coerce')

df_test_dummies['Power'] = pd.to_numeric(df_test_dummies['Power'], errors='coerce')

df_test_dummies['Seats'] = pd.to_numeric(df_test_dummies['Seats'], errors='coerce')
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
train_Engine = df_train_dummies['Engine']

'''pd.DataFrame(pd.Series(train_Engine.ravel()).describe()).transpose()'''

df_train_dummies['Engine'] = imputer.fit_transform(df_train_dummies['Engine'].values.reshape(-1, 1))
train_Power = df_train_dummies['Power']

'''pd.DataFrame(pd.Series(train_Power.ravel()).describe()).transpose()'''

df_train_dummies['Power'] = imputer.fit_transform(df_train_dummies['Power'].values.reshape(-1, 1))
test_Engine = df_test_dummies['Engine']

'''pd.DataFrame(pd.Series(test_Engine.ravel()).describe()).transpose()'''

df_test_dummies['Engine'] = imputer.fit_transform(df_test_dummies['Engine'].values.reshape(-1, 1))
test_Power = df_test_dummies['Power']

'''pd.DataFrame(pd.Series(test_Power.ravel()).describe()).transpose()'''

df_test_dummies['Power'] = imputer.fit_transform(df_test_dummies['Power'].values.reshape(-1, 1))
test_Mileage = df_test_dummies['Mileage']

'''pd.DataFrame(pd.Series(test_Mileage.ravel()).describe()).transpose()'''

df_test_dummies['Mileage'] = imputer.fit_transform(df_test_dummies['Mileage'].values.reshape(-1, 1))
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_Seats = df_train_dummies['Seats']

'''pd.DataFrame(pd.Series(train_Seats.ravel()).describe()).transpose()'''

df_train_dummies['Seats'] = imputer.fit_transform(df_train_dummies['Seats'].values.reshape(-1, 1))
test_Seats = df_test_dummies['Seats']

'''pd.DataFrame(pd.Series(test_Seats.ravel()).describe()).transpose()'''

df_test_dummies['Seats'] = imputer.fit_transform(df_test_dummies['Seats'].values.reshape(-1, 1))
X_train = df_train_dummies.drop(['Price'], axis=1).values

y_train = df_train_dummies['Price'].values

X_test = df_test_dummies.values
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

reg = RandomForestRegressor(criterion='mse')

'''params = {'max_depth':[11,12,13,14,15], 'n_estimators':[30,50]}

gscv = GridSearchCV(reg, params, cv=5, scoring='neg_mean_squared_log_error')

gscv.fit(X_train, y_train)'''
'''gscv.best_params_'''
reg =RandomForestRegressor(criterion='mse', max_depth=13, n_estimators=20)
reg.fit(X_train, y_train)
reg.score(X_train, y_train)
p = reg.predict(X_test)
p
df_submit = pd.read_csv('/kaggle/input/1056lab-used-cars-price-prediction/sampleSubmission.csv', index_col=0)

df_submit['Price'] = p

df_submit.to_csv('my_submission3.csv')