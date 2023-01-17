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
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
(all_df == '?').sum()
train_df=train_df.replace('?',np.nan)

test_df=test_df.replace('?',np.nan)
all_df = all_df.replace('?', np.nan)
all_df.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')



train_df['normalized-losses'] = imputer.fit_transform(train_df['normalized-losses'].values.reshape(-1, 1))

test_df['normalized-losses'] = imputer.transform(test_df['normalized-losses'].values.reshape(-1, 1))



train_df['bore'] = imputer.fit_transform(train_df['bore'].values.reshape(-1,1))

test_df['bore'] = imputer.transform(test_df['bore'].values.reshape(-1,1))

train_df['stroke'] = imputer.fit_transform(train_df['stroke'].values.reshape(-1,1))

test_df['stroke'] = imputer.transform(test_df['stroke'].values.reshape(-1,1))

train_df['horsepower'] = imputer.fit_transform(train_df['horsepower'].values.reshape(-1,1))

test_df['horsepower'] = imputer.transform(test_df['horsepower'].values.reshape(-1,1))

train_df['peak-rpm'] = imputer.fit_transform(train_df['peak-rpm'].values.reshape(-1,1))

test_df['peak-rpm'] = imputer.transform(test_df['peak-rpm'].values.reshape(-1,1))

train_df['price'] = imputer.fit_transform(train_df['price'].values.reshape(-1,1))

test_df['price'] = imputer.transform(test_df['price'].values.reshape(-1,1))



train_df=train_df.dropna(how='any')



#test_df=test_df.dropna(how='any')
train_df['make']=train_df['make'].map({'alfa-romero':0, 'audi':1, 'bmw':2, 'chevrolet':3, 'dodge':4, 'honda':5, 'isuzu':6, 'jaguar':7, 'mazda':8, 'mercedes-benz':9, 'mercury':10, 'mitsubishi':11, 'nissan':12, 'peugot':13, 'plymouth':14, 'porsche':15, 'renault':16, 'saab':17, 'subaru':18, 'toyota':19, 'volkswagen':20, 'volvo':21})

train_df['fuel-type']=train_df['fuel-type'].map({'diesel':0, 'gas':1})

train_df['aspiration']=train_df['aspiration'].map({'std':0, 'turbo':1})

train_df['num-of-doors']=train_df['num-of-doors'].map({'four':4, 'two':2})

train_df['body-style']=train_df['body-style'].map({'hardtop':0, 'wagon':1, 'sedan':2, 'hatchback':3, 'convertible':4})

train_df['drive-wheels']=train_df['drive-wheels'].map({'4wd':0, 'fwd':1, 'rwd':2})

train_df['engine-location']=train_df['engine-location'].map({'front':0, 'rear':1})

train_df['engine-type']=train_df['engine-type'].map({'dohc':0, 'dohcv':1, 'l':2, 'ohc':3, 'ohcf':4, 'ohcv':5, 'rotor':6})

train_df['num-of-cylinders']=train_df['num-of-cylinders'].map({'eight':8, 'five':5, 'four':4, 'six':6, 'three':3, 'twelve':12, 'two':2})

train_df['fuel-system']=train_df['fuel-system'].map({'1bbl':0, '2bbl':1, '4bbl':2, 'idi':3, 'mfi':4, 'mpfi':5, 'spdi':6, 'spfi':7})
test_df['make']=test_df['make'].map({'alfa-romero':0, 'audi':1, 'bmw':2, 'chevrolet':3, 'dodge':4, 'honda':5, 'isuzu':6, 'jaguar':7, 'mazda':8, 'mercedes-benz':9, 'mercury':10, 'mitsubishi':11, 'nissan':12, 'peugot':13, 'plymouth':14, 'porsche':15, 'renault':16, 'saab':17, 'subaru':18, 'toyota':19, 'volkswagen':20, 'volvo':21})

test_df['fuel-type']=test_df['fuel-type'].map({'diesel':0, 'gas':1})

test_df['aspiration']=test_df['aspiration'].map({'std':0, 'turbo':1})

test_df['num-of-doors']=test_df['num-of-doors'].map({'four':4, 'two':2})

test_df['body-style']=test_df['body-style'].map({'hardtop':0, 'wagon':1, 'sedan':2, 'hatchback':3, 'convertible':4})

test_df['drive-wheels']=test_df['drive-wheels'].map({'4wd':0, 'fwd':1, 'rwd':2})

test_df['engine-location']=test_df['engine-location'].map({'front':0, 'rear':1})

test_df['engine-type']=test_df['engine-type'].map({'dohc':0, 'dohcv':1, 'l':2, 'ohc':3, 'ohcf':4, 'ohcv':5, 'rotor':6})

test_df['num-of-cylinders']=test_df['num-of-cylinders'].map({'eight':8, 'five':5, 'four':4, 'six':6, 'three':3, 'twelve':12, 'two':2})

test_df['fuel-system']=test_df['fuel-system'].map({'1bbl':0, '2bbl':1, '4bbl':2, 'idi':3, 'mfi':4, 'mpfi':5, 'spdi':6, 'spfi':7})
train_df['bore']=train_df['bore'].astype(float)

train_df['stroke']=train_df['stroke'].astype(float)

train_df['horsepower']=train_df['horsepower'].astype(int)

train_df['peak-rpm']=train_df['peak-rpm'].astype(int)

train_df['price']=train_df['price'].astype(int)
test_df['bore']=test_df['bore'].astype(float)

test_df['stroke']=test_df['stroke'].astype(float)

test_df['horsepower']=test_df['horsepower'].astype(int)

test_df['peak-rpm']=test_df['peak-rpm'].astype(int)

test_df['price']=test_df['price'].astype(int)
#all_df['num-of-doors'].fillna(-9999,inplace=True)
#train_df['num-of-doors'] = imputer.fit_transform(train_df['num-of-doors'].values.reshape(-1,1))
numeric_columns = ['normalized-losses', 'num-of-doors', 'body-style', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'peak-rpm']

columns=train_df.columns

#numeric_columns=['normalized-losses','make','num-of-doors','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke','compression-ratio','horsepower','city-mpg','highway-mpg','price']

tcolumns=test_df.columns

#X = train_df[columns].drop('symboling',axis=1).to_numpy()

X=train_df[numeric_columns].to_numpy()

y = train_df['symboling'].to_numpy()

X_test = test_df[numeric_columns].to_numpy()

#X_test = test_df[tcolumns].to_numpy()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

len(X_train),len(X_valid),len(y_train),len(y_valid)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)

#model.fit(X_res,y_res)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

#model.fit(X_res,y_res)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from lightgbm import LGBMRegressor

model = LGBMRegressor(random_state=0)

model.fit(X_train, y_train)

#model.fit(X_res,y_res)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0,k_neighbors=2)

X_res, y_res = sm.fit_resample(X_train, y_train)
len(X_res)
from collections import Counter

Counter(y_train)
Counter(y_res)
import optuna

def objective(trial):

    max_depth = trial.suggest_int('max_depth', 1, 30)

    n_estimators = trial.suggest_int('n_estimators',10,300)

    model = RandomForestRegressor(criterion='mse', max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

    model.fit(X_res, y_res)

    y_pred = model.predict(X_valid)

    return np.sqrt(mean_squared_error(y_valid, y_pred))



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
max_depth = study.best_params['max_depth']

n_estimators = study.best_params['n_estimators']

model = RandomForestRegressor(criterion='mse', max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

model.fit(X_res, y_res)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
# feature_importanceを求める

feature_importances = model.feature_importances_

print(feature_importances)
import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(40, 5))

plt.ylim([0, 0.6])

y_ = feature_importances

x = np.arange(len(y_))

plt.bar(x, y_, align="center")

plt.xticks(x, train_df[columns])

plt.show()
model.fit(X,y)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv',index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')