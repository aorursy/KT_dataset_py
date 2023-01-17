# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH = "../input/bluebook-for-bulldozers/"
! ls {PATH}
df_raw = pd.read_csv(f'{PATH}TrainAndValid.csv', low_memory=False, parse_dates=["saledate"])

test = pd.read_csv(f'{PATH}Test.csv', low_memory=False)
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)




display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_raw.isna().sum()
features_to_consider = ['YearMade', 'datasource', 'state', 'fiBaseModel', 'fiProductClassDesc' , 'fiModelDesc']
model_performance = {

    "model_name":[],

    "performance":[],

    "model_score":[]

}
def model_score(model, X_trn, y_trn, X_val, y_val):

    model.fit(X_trn, y_trn)

    pred = model.predict(X_val)

    model_performance['model_name'].append(type(model).__name__)

    model_performance['performance'].append(np.sqrt(mse(pred,y_val)))

    print(model.score(X_val,y_val)*100)

    model_performance['model_score'].append(model.score(X_val,y_val))

    return np.sqrt(mse(pred, y_val))
X = df_raw[features_to_consider]

y = df_raw.SalePrice
LabelEnc = LabelEncoder()

X['state']=LabelEnc.fit_transform(X.state)

X['fiBaseModel']= LabelEnc.fit_transform(X.fiBaseModel)

X['fiProductClassDesc']= LabelEnc.fit_transform(X.fiProductClassDesc)

X['fiModelDesc']= LabelEnc.fit_transform(X.fiModelDesc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Random Forest Regressor

model = RandomForestRegressor(max_depth=30, min_samples_split=20, n_estimators=110, n_jobs= -1)

model_score(model, X_train, y_train, X_test, y_test)
# Linear Regression

model = LinearRegression()

model_score(model, X_train, y_train, X_test, y_test)
# MLP 

model = MLPRegressor(hidden_layer_sizes=(100), activation="relu", solver="adam", alpha=0.0001, verbose=True)

model_score(model, X_train, y_train, X_test, y_test)
# K Neighbours 

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(weights='distance', algorithm='auto')

model_score(model, X_train, y_train, X_test, y_test)
# SVM 

from sklearn.svm import SVR

model = SVR(max_iter=1000)

model_score(model, X_train, y_train, X_test, y_test)
# SGD Regressor 

from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter=1000, tol=1e-3)

model_score(model, X_train, y_train, X_test, y_test)
# Gradient Boosting Regressor 

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0, loss='ls')

model_score(model, X_train, y_train, X_test, y_test)
model_df = pd.DataFrame(model_performance)

model_df
model_df['model_score'] = model_df['model_score']*100

model_df
model_df = model_df.sort_values("model_score", ascending=False)

model_df
# Random Forest Regressor

model = RandomForestRegressor(max_depth=30, min_samples_split=20, n_estimators=110, n_jobs= -1)

model.fit(X_train, y_train)

model.score(X_test, y_test)
model.score(X_test, y_test)*100