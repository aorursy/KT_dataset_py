# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import os 



import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train=pd.read_csv("../input/train.csv")

print('train data')

print(np.shape(train)) #data number=1460. Not enough for deep learning?

print(train.head())

print(train.columns)



test=pd.read_csv("../input/test.csv")

print('test data')

print(np.shape(test)) #data number=1460. Not enough for deep learning?

print(test.columns)



print(test.columns == train.drop("SalePrice", axis=1).columns)

# As evaluation is done based on log('SalePrice'), we add it as a new column.

#train_df['log_SalePrice']=math.log(train_df['SalePrice'])

# TO BE DONE
pd.set_option("display.max_rows", 10)

print('Number of NaN:',train.isnull().sum().sort_values(ascending=False))

print('Number of columns:',train.shape[1])



def drop_features(df):

    #PoolQC, Fence, MiscFeature, and Alley have too many NANs, so we drop it.

    list_drop_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']

    for feat in list_drop_features: 

        del df[feat]

    #del train['FireplaceQu']

    return df



train = drop_features(train)

# also drop Id

del train['Id']



#print(type(train))

print("train.shape (after droping some features)",train.shape)



# MSZoning, Utilities, Exterior1st, Exterior2nd, MasVnrType, BsmtFullBath, BsmtQual, contain only a few NANs, so we drop the index in which they contain NAN.

#train=train.dropna(subset=['MSZoning','Utilities','Exterior1st','Exterior2nd','Electrical',\

#                                 'MasVnrType','BsmtFullBath','BsmtQual','BsmtExposure','BsmtFinType2'])



#pd.set_option("display.max_rows", 81)

#print(train)

#print('Number of NaN:')

#print(train.isnull().sum())

#print(np.shape(train))



# Before plotting, NANs should be removed, so we make temporal dataframe just for plot

#train_tmp=train[['SalePrice','LotFrontage','GarageType','BsmtExposure','BsmtFinType2','Electrical']]

#train_tmp=train_tmp.dropna()

#print(train_tmp.isnull().sum())

#sns.set(style='whitegrid',context='notebook')

#sns.pairplot(train_tmp,size=4)

#plt.show()

#plt.bar(train_tmp['GarageType'],train_tmp['SalePrice']) #これでは各GarageTypeでの最大値をplotしたにすぎない。各typeでの平均値をplotしなければならない。

#plt.show()



# We saw that the both quantities above are important.

# We replace NANs in 'LotFrontage' with its mean value, and just drop data with NAN in 'GarageType'

# as I do not know what to do.

#train_df=train_df.dropna(subset=['GarageType'])

#train_df['LotFrontage']=train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())

print('Number of NANs:')

print(train.isnull().sum().sum())

print('Training data number decresead from 1460 to',len(train.index))

print('Number of columns:',train.shape[1])
# TO BE FIXED...

#import category_encoders as ce

import featuretools as ft



pd.set_option("display.max_columns", 170)

pd.set_option("display.max_rows", 170)



#print(train.dtypes)

#print(train.select_dtypes(include=object))

print(train.select_dtypes(include=object).columns)



X = train.drop("SalePrice", axis=1)

y = train[["SalePrice"]]



def preprocess_category_features(df):

    # Cathegory data: 

    list_category=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

               'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 

               'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 

               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',

               'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 

               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 

               'SaleType', 'SaleCondition',"FireplaceQu"]

    # Check what values are in each column.

    #for name in list_category:

        #print(df[[name]].apply(pd.value_counts))



    # Ordered cathegory data:

    list_ordered=['LotShape', 'LandContour', 'LandSlope', 'Condition1', 'Condition2', 'ExterQual',

              'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual',

              'GarageCond']

    ordered_map={'LotShape':{'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3},

             'LandContour':{'Lvl':0, 'Bnk':1, 'Low':2, 'HLS':3},

             'LandSlope':{'Gtl':0, 'Mod':1, 'Sev':2},

             'Condition1':{'Norm':0, 'Feedr':1, 'PosN':2, 'Artery':3, 'RRAe':4, 'RRNn':5,

                           'RRAn':6, 'PosA':7, 'RRNe':8},

             'Condition2':{'Norm':0, 'Artery':1, 'RRNn':2, 'Feedr':3, 'PosN':4, 'PosA':5,

                           'RRAn':6, 'RRAe':7},

             'ExterQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3},

             'ExterCond':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'BsmtQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'BsmtCond':{'Gd':0,'TA':1, 'Fa':2, 'Po':3}, 

             'HeatingQC':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4}, 

             'KitchenQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'GarageQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4},

             'GarageCond':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4},

            'FireplaceQu':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}}

    

    print("df.shape (original):",df.shape)

    for name in list_ordered:

        df[name]=df[name].map(ordered_map[name])

        list_category.remove(name)

    print("df.shape (after mapping ordered features):",df.shape)

    

    # Unordered cathegory data:

    print("length of list_category:",len(list_category))    

    print("list_category:",list_category)

    

    # OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

    ce_ohe = ce.OneHotEncoder(cols=list_category,handle_unknown='impute')

    # pd.DataFrameをそのまま突っ込む

    df_one_hot_encoded = ce_ohe.fit_transform(df)

    df_one_hot_encoded.head()

    

    #print("df[list_category].shape:",df[list_category].shape)

    #df_one_hot_encoded=pd.get_dummies(df[list_category])

    #print("df_one_hot_encoded.shape:",df_one_hot_encoded.shape)

    #df = pd.concat([df, df_one_hot_encoded], axis=1)

    #print("df.shape (after concat one-hot columns):",df.shape)

    #for col in list_category:

        #del df[col]

    print(df_one_hot_encoded.head())

    print(df["LotShape"].head(),df_one_hot_encoded["LotShape"].head())

    print("df_one_hot_encoded.columns:",df_one_hot_encoded.columns)

    print("df_one_hot_encoded.shape (final):",df_one_hot_encoded.shape)

    

    return df_one_hot_encoded, ce_ohe



X, ce_ohe_train = preprocess_category_features(X)

print("X.shape (after one-hot encoding)",X.shape)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)



'''

cols_number=['SalePrice','MSSubClass','LotFrontage','LotArea','OverallQual']

sns.pairplot(train_df[cols_number],size=4)

plt.show()

cols_number=['SalePrice','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea']

sns.pairplot(train_df[cols_number],size=4)

plt.show()

cols_number=['SalePrice','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']

sns.pairplot(train_df[cols_number],size=4)

plt.show()

plt.bar(train_df['MSZoning'],train_df['SalePrice'])

plt.show()

plt.bar(train_df['Street'],train_df['SalePrice'])

plt.show()

plt.bar(train_df['LotShape'],train_df['SalePrice'])

plt.show()

plt.bar(train_df['GarageType'],train_df['SalePrice'])

plt.show()

'''



corr_abs = corrmat.abs() 

print(corr_abs.columns)

print(corr_abs["SalePrice"].sort_values(ascending=False))

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_squared_log_error

import xgboost as xgb



print(xgb.__version__)



pd.set_option("display.max_columns", 230)

pd.set_option("display.max_rows", 230)



def XGBoost_train(X_train, X_test, y_train, y_test, xgb_params):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test, label=y_test)



    evals = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(xgb_params,

                    dtrain,

                    num_boost_round=1000,

                    early_stopping_rounds=10,

                    evals=evals,)



    y_pred = bst.predict(dtest)

    msle = mean_squared_log_error(y_test, y_pred) 

    print('RMSLE:', math.sqrt(msle)) 

    # " Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value 

    # and the logarithm of the observed sales price."



    return bst, msle



params = {'objective': 'reg:squarederror', # as we don't have SLE at xgboost==0.9 ...

        'eval_metric': 'rmse', # as we don't have RMSLE at xgboost==0.9 ...

}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,

                                                        shuffle=True,

                                                        random_state=0,)

print(X_train.shape)

bst, msle = XGBoost_train(X_train, X_test, y_train, y_test, params)
import optuna



def optimize_hyperparam(): 

    print("start optimization of hyperparameter for XGBoost")

    def objective(trial):

        #n_estimators = trial.suggest_int('n_estimators', 5, 150)

        max_depth = trial.suggest_int('max_depth', 2, 20)

        #learning_rate = trial.suggest_loguniform('learning_rate', 1e-2, 0.3)

        #booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])

        

        params = {'objective': 'reg:squarederror', # as we don't have SLE at xgboost==0.9 ...

        'eval_metric': 'rmse', # as we don't have RMSLE at xgboost==0.9 ...

        'verbosity' : 0, 

        #'n_estimators': n_estimators,

        'max_depth': max_depth          

        }



        reg, msle = XGBoost_train(X_train, X_test, y_train, y_test, params)

        return msle



    study = optuna.create_study()

    study.optimize(objective, n_trials=200)

    print("best RMSLE:",study.best_value)

    print("best parameters:",study.best_params)

    

    params_opt = {'objective': 'reg:squarederror', # as we don't have SLE at xgboost==0.9 ...

        'eval_metric': 'rmse', # as we don't have RMSLE at xgboost==0.9 ...

        'verbosity' : 0, 

        #'n_estimators': study.best_params["n_estimators"],

        'max_depth': study.best_params["max_depth"]          

        }

    reg_opt, msle_opt = XGBoost_train(X_train, X_test, y_train, y_test, params_opt)

    

    return reg_opt, study.best_params



#reg_opt_XGB, best_params = optimize_hyperparam()
dX_test = xgb.DMatrix(X_test)

y_pred = bst.predict(dX_test)

#y_pred = reg_opt_XGB.predict(dX_test)

print(y_test.shape, y_pred.shape)

y_pred = pd.DataFrame(y_pred)

y_pred.columns = ['SalePrice']

print(y_test.columns, y_pred.columns)



horizontal = np.arange(50000, 700000, 10000)

vertical = horizontal 

plt.plot(horizontal, vertical)



plt.plot(y_test['SalePrice'],y_pred['SalePrice'],marker='o', linestyle='None')

plt.title('')  

plt.xlabel('Actual SalePrice [$]')                            

plt.ylabel('Predicted SalePrice [$]')   

plt.xscale('log')

plt.yscale('log')

plt.xticks([100000, 1000000])

plt.yticks([100000, 1000000])

#plt.grid(which='major',color='black',linestyle='-')

#plt.grid(which='minor',color='black',linestyle='-')

plt.show()   

#list_imp = ["gain", "weight", "cover"] 

list_imp = ["gain"] 

# ”weight” is the number of times a feature appears in a tree

# ”gain” is the average gain of splits which use the feature

# ”cover” is the average coverage of splits which use the feature 

# where coverage is defined as the number of samples affected by the split

for imp_type in list_imp:

    print("importance_type:",imp_type)

    _, ax = plt.subplots(figsize=(12, 8))

    xgb.plot_importance(bst,

                        ax=ax,

                        importance_type=imp_type,max_num_features=10,

                        show_values=False)

plt.show()

# explanation by LIME

import lime

import lime.lime_tabular



#print(X_train.head)

print(type(X_train.values),X_train.values.shape)

print(X_train.values)



explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train.values,

                                                   mode="regression",

                                                   feature_names = X_train.columns,)

# train with all the training data

print("X.shape:",X.shape)

dtrain = xgb.DMatrix(X, label=y)

params_opt = {'objective': 'reg:squarederror', # as we don't have SLE at xgboost==0.9 ...

        'eval_metric': 'rmse', # as we don't have RMSLE at xgboost==0.9 ...

        'verbosity' : 0, 

        #'n_estimators': study.best_params["n_estimators"],

        'max_depth': 3 #best_params["max_depth"]          

        }

reg_opt_all = xgb.train(params_opt,

                    dtrain)



# predict with test data

test = drop_features(test)

print("test.shape (after dropping some features):",test.shape)

test_id = test['Id']

del test['Id']

# Ordered cathegory data:

list_ordered=['LotShape', 'LandContour', 'LandSlope', 'Condition1', 'Condition2', 'ExterQual',

              'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual',

              'GarageCond']

ordered_map={'LotShape':{'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3},

             'LandContour':{'Lvl':0, 'Bnk':1, 'Low':2, 'HLS':3},

             'LandSlope':{'Gtl':0, 'Mod':1, 'Sev':2},

             'Condition1':{'Norm':0, 'Feedr':1, 'PosN':2, 'Artery':3, 'RRAe':4, 'RRNn':5,

                           'RRAn':6, 'PosA':7, 'RRNe':8},

             'Condition2':{'Norm':0, 'Artery':1, 'RRNn':2, 'Feedr':3, 'PosN':4, 'PosA':5,

                           'RRAn':6, 'RRAe':7},

             'ExterQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3},

             'ExterCond':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'BsmtQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'BsmtCond':{'Gd':0,'TA':1, 'Fa':2, 'Po':3}, 

             'HeatingQC':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4}, 

             'KitchenQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3}, 

             'GarageQual':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4},

             'GarageCond':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4},

            'FireplaceQu':{'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}}

for name in list_ordered:

    test[name] = test[name].map(ordered_map[name])



#test, ce_ohe_test = preprocess_category_features(test)

test = ce_ohe_train.transform(test)

print("test.shape (after one-hot encoding):",test.shape)

print(type(test),test.columns)

print(test.head)

dX_test = xgb.DMatrix(test)

print("dX_test col,row:",dX_test.num_col(), dX_test.num_row())

y_pred = pd.DataFrame(reg_opt_all.predict(dX_test))

print(y_pred,type(y_pred))

y_pred = pd.concat([test_id, y_pred], axis=1)

y_pred.columns=["Id","SalePrice"]

print(y_pred.head)

if os.path.exists('output.csv'):

    os.remove('output.csv')

y_pred.to_csv('output.csv', index=False)

from sklearn.linear_model import LinearRegression



X_train = np.asarray(train["OverallQual"]).reshape(-1, 1)

y_train = np.asarray(train["SalePrice"]).reshape(-1, 1)

reg = LinearRegression().fit(X_train, y_train)



horizontal = np.arange(50000, 700000, 10000)

vertical = horizontal 

plt.plot(horizontal, vertical)

y_train_pred = reg.predict(X_train)

plt.plot(y_train, y_train_pred ,marker='o', linestyle='None')

plt.title('')  

plt.xlabel('Actual SalePrice [$]')                            

plt.ylabel('Predicted SalePrice [$]')   

plt.xscale('log')

plt.yscale('log')

plt.xticks([100000, 1000000])

plt.yticks([100000, 1000000])

#plt.grid(which='major',color='black',linestyle='-')

#plt.grid(which='minor',color='black',linestyle='-')

plt.show()   



X_test = np.asarray(test["OverallQual"]).reshape(-1, 1)

predict_test = reg.predict(X_test)

predict_test = pd.DataFrame(predict_test)

predict_test = pd.concat([test_id, predict_test], axis=1)

predict_test.columns=["Id","SalePrice"]

print(predict_test.head)

if os.path.exists('output_baseline.csv'):

    os.remove('output_baseline.csv')

predict_test.to_csv('output_baseline.csv', index=False)