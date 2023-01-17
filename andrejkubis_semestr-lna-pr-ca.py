import numpy as np 

import pandas as pd 



import os

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split  

from lightgbm import LGBMRegressor   

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error

import math





print(os.listdir("../input"))
train_dataset = pd.read_csv('../input/train.csv') #načitanie treningových dát





test_data = pd.read_csv('../input/test.csv') #testovacie dáta



##################################       report o dátach - zaujalo ma to     ########################################

#import pandas_profiling        

#profile_report = pandas_profiling.ProfileReport(train)

#profile_report
x = train_dataset.iloc[:,1:-1] #rozdelenie dát na X a Y

y = train_dataset.iloc[:,-1]



test_dataset = test_data

test_data.isnull().sum()

x.isnull().sum() # zistenie chýbajúcich hodnôt

col_miss_val = [col for col in train_dataset.columns if train_dataset[col].isnull().any()] # stplce s chýbajúcimi hodnotami

print(col_miss_val)



test_col_miss_val = [col for col in test_dataset.columns if test_dataset[col].isnull().any()] # stplce s chýbajúcimi hodnotami

print(test_col_miss_val)



import matplotlib.pyplot as plt

import seaborn as sns



#zobrazenie korelačnej matice

corr = train_dataset.corr()

top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(12, 8))

top_corr = train_dataset[top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()





from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline



categorical_inputs = ["OverallQual"] # -----

numeric_inputs = ["GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt"] # -----

output = ["SalePrice"]



input_preproc = make_column_transformer(

    (make_pipeline(

        SimpleImputer(strategy="most_frequent"),

        OrdinalEncoder()),

     categorical_inputs),

    

    (make_pipeline(

        SimpleImputer(strategy="most_frequent"),

        StandardScaler()),

     numeric_inputs)

)



x_train = input_preproc.fit_transform(x[categorical_inputs+numeric_inputs])

y_train = y



X_test = input_preproc.transform(test_dataset[categorical_inputs+numeric_inputs])

model = XGBRegressor()

model.fit(x_train,y_train)

#print(math.sqrt(mean_squared_log_error(y_train, model.predict(x_train))))



lightgbm = LGBMRegressor()

lightgbm.fit(x_train,y_train)

#print(math.sqrt(mean_squared_log_error(y_train, lightgbm.predict(x_train))))





from sklearn.ensemble import RandomForestRegressor

RFG = RandomForestRegressor(n_estimators=100)

RFG.fit(x_train, y_train)





from sklearn import linear_model

LM = linear_model.LinearRegression()

LM.fit(x_train, y_train)





from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

ridge = RidgeCV()

ridge.fit(x_train, y_train)





lasso = LassoCV()

lasso.fit(x_train, y_train)







from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(x_train, y_train)







def blend_models_predict(x_train):

    return ((0.2  * model.predict(x_train)) + \

            (0.4 * lightgbm.predict(x_train)) + \

            (0.1 * LR.predict(x_train)) + \

            (0.1 * RFG.predict(x_train)) + \

            (0.1 * LM.predict(x_train)) + \

            (0.05 * ridge.predict(x_train)) + \

            (0.05 * lasso.predict(x_train)))

print('RMSLE score on train data:')

print(math.sqrt(mean_squared_log_error(y_train, blend_models_predict(x_train))))
def blend_models_predict(X_test):

    return ((0.2  * model.predict(X_test)) + \

            (0.4 * lightgbm.predict(X_test)) + \

            (0.1 * LR.predict(X_test)) + \

            (0.1 * RFG.predict(X_test)) + \

            (0.1 * LM.predict(X_test)) + \

            (0.05 * ridge.predict(X_test)) + \

            (0.05 * lasso.predict(X_test)))

print('Predikcia:')

print(blend_models_predict(X_test))




