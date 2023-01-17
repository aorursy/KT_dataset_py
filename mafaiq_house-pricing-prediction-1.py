'''This is my first ever real submission after finishing the intro and intermediate machine learning courses on kaggle. 

The random forest regressor bit will be commented out, as with it, the code will take a lot longer to run, but I will not delete it, as it 

serves as one of two models that I was comparing one another against to see what worked better, and also as a learning step for me.



Hope you enjoy this snippet of code :) I still have a lot to learn, and will hopefully continue to do better. '''
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
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
#get dataset

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col = "Id") 

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col = "Id")



print(df_train.shape)

# df_train.columns
#separate X and y values, and drop missing values

'''this line gets column names to remvoe that have more than 60% missing values. 

Although i believe some of the features like "fence" might have a big effect on price.''' 

empty_cols = [col for col in df_train.columns if df_train[col].isnull().sum() > (0.6 * len(df_train.index))]



df_train.dropna(axis = 0, subset = ["SalePrice"], inplace = True)

y = df_train.SalePrice

df_train.drop(columns = empty_cols, axis = 1, inplace=True)

df_test.drop(columns = empty_cols, axis = 1, inplace=True)

df_train.drop(columns = ["SalePrice"], axis=1, inplace = True)



X_train_full, X_val_full, y_train, y_val = train_test_split(df_train, y, train_size = 0.8, test_size =  0.2, random_state = 0)



'''get the column names for categorical and numerical data for easy cleaning later'''

#get numeric data type columns

numeric_cols = [numcol for numcol in X_train_full.columns if X_train_full[numcol].dtype in ["int64", "float64"]]



#get categorical column names and also for the ones that have relatively low cardinality 

categorical_cols = [catcol for catcol in X_train_full.columns if X_train_full[catcol].dtype == "object"]



#setup for all the columns i am taking into the model

taken_cols = numeric_cols + categorical_cols



X_train = X_train_full[taken_cols].copy()

X_test = df_test[taken_cols].copy()
print(X_test.shape)

X_train.shape
'''This step is to pre_process the data'''



numerical_processor = SimpleImputer(strategy = "median")



categorical_processor = Pipeline(steps=[

    ("cat_imputer", SimpleImputer(strategy = "most_frequent")),

    ("onehot_encoder", OneHotEncoder(handle_unknown = "ignore", sparse = False))])



processor_bundler = ColumnTransformer(transformers = [

    ("for_numerics", numerical_processor, numeric_cols), 

    ("for_cats", categorical_processor, categorical_cols)])
'''this step is to find the best forest regressor model. You can uncomment the lines to let it run if you want'''



# #this function will help to find the best n_estimators for our forest regressor

# def find_forest_regressor(n_estimators):

#     model = RandomForestRegressor(n_estimators = n_estimators, random_state = 0)

#     model_pipeline = Pipeline(steps = [

#         ("pre_processor", processor_bundler),

#         ("model", model)])

#     score = -1 * cross_val_score(model_pipeline, X_train, y_train, cv = 5, scoring = "neg_mean_absolute_error")

#     return score.mean()



# best_n_estimators = 50 #arbitrary number as a base

# best_min_mae = find_forest_regressor(best_n_estimators) #based on above line



# #iterative process to find the best n_estimators

# for n in range (250, 750, 50): 

#     if find_forest_regressor(n) < best_min_mae:

#         best_min_mae = find_forest_regressor(n)

#         best_n_estimators = n



# print("best n = " + str(best_n_estimators) + ". MAE = " + str(best_min_mae))
'''function to find best xgb regressor parameters'''

def find_xgb_regressor(n_estimators, learning_rate):

    model = XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, random_state = 0)

    pipeline = Pipeline(steps = [

        ("pre_processor", processor_bundler),

        ("model", model)])

    score = -1 * cross_val_score(pipeline, X_train, y_train, cv = 5, scoring = "neg_mean_absolute_error")

    return score.mean()
#find best n_estimators with a learning_rate = 0.1

best_n_xgb = 100 #arbitrary n as a benchmark

best_mae_xgb = find_xgb_regressor(best_n_xgb, 0.1)



for n in range(100,200,10 ):

    if find_xgb_regressor(n, 0.1) < best_mae_xgb:

        best_mae_xgb = find_xgb_regressor(n, 0.1)

        best_n_xgb = n

        

#I have run the loop from n = 100 to n = 100. And the best MAE comes from n = 150. Then i did another loop to further get the best MAE

print("best n = " + str(best_n_xgb) + ". MAE = " + str(best_mae_xgb))
# find best learning rate

best_lrate_xgb = 0.01 #arbitrary n as a benchmark

best_mae_lrate = find_xgb_regressor(best_n_xgb, best_lrate_xgb)



lrates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

for n in range(0, len(lrates)):

    if find_xgb_regressor(best_n_xgb, lrates[n]) < best_mae_lrate:

        best_mae_lrate = find_xgb_regressor(best_n_xgb, lrates[n])

        best_lrate_xgb = lrates[n]

#         print(find_xgb_regressor(best_n_xgb, lrates[n]))



print("best learning_rate = " + str(best_lrate_xgb) + ". MAE = " + str(best_mae_lrate))
#create model based off findings above

model_type =  XGBRegressor(n_estimators = best_n_xgb, learning_rate = best_lrate_xgb, random_state = 0)

model = Pipeline(steps = [

    ("preprocessor", processor_bundler),

    ("model", model_type)])



model.fit(df_train, y)

preds = model.predict(X_test)
#save fil predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)