import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np



from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax





import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import cross_val_score

from sklearn import linear_model

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge



from sklearn.svm import SVR



from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder



import xgboost as xgb

import lightgbm as lgb



from xgboost import plot_importance,plot_tree

from mlxtend.regressor import StackingCVRegressor

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print('Size of the training dataset',df_train.shape)

print('Size of the testing dataset',df_test.shape)
plt.style.use('default')



missing_train = df_train.isnull().sum()

missing_train = missing_train[missing_train > 0].sort_values(ascending = False)

plt.figure(figsize = (12, 3))

plt.subplot(1, 2, 1)

g = sns.barplot(missing_train.index, missing_train.values,color = 'seagreen', alpha = 0.7)

g.set_xticklabels(missing_train.index, rotation = 90, fontsize = 8)

plt.yticks(fontsize = 10)

plt.title('Missing values in the training set', fontsize = 10)

plt.ylabel('', fontsize = 10)



missing_test = df_test.isnull().sum()

missing_test = missing_test[missing_test > 0].sort_values(ascending = False)

plt.subplot(1, 2, 2)

g = sns.barplot(missing_test.index, missing_test.values, color='seagreen', alpha = 0.7)

g.set_xticklabels(missing_test.index, rotation = 90, fontsize = 8)

plt.yticks(fontsize = 10)

plt.title('Missing values in the test set', fontsize = 10)

plt.ylabel('', fontsize = 10)



plt.show()
def eda(df_train, df_test):

    df_train_explore = df_train.copy()#create copies so that the original sets remain unaltered after the EDA 

    df_test_explore = df_test.copy()

    col = df_test_explore.columns # I will look at the 'Sale Price' column later

    f = 16 #font size

    for i in range(1, len(col)): # start from 1 because I am not interested in the 'Id' column

        print(col[i])

        t = df_train_explore[col[i]].dtypes

        

        plt.figure(figsize = (28, 4))

        df_train_explore[col[i]].dropna(inplace = True) # for the first EDA I just remove the NaN values because 

                                                       #some Seaborn plots cannot handle them  

        df_test_explore[col[i]].dropna(inplace = True)

        

        if t != 'object': #numerical features

            plt.subplot(1, 4, 1)

            sns.distplot(df_train_explore[col[i]], color = 'seagreen', hist_kws = {"alpha": 0.5})

            plt.ylabel('Frequency', fontsize = f)

            plt.xlabel(col[i], fontsize = f)

            plt.title('Training set')

            plt.subplot(1, 4, 2)

            sns.distplot(df_test_explore[col[i]], color = 'seagreen', hist_kws = {"alpha": 0.5})

            plt.ylabel('Frequency', fontsize = f)

            plt.xlabel(col[i], fontsize = f)

            plt.title('Test set')

            plt.subplot(1, 4, 3)

            sns.scatterplot(x = col[i], y = 'SalePrice', data = df_train_explore, color = 'seagreen', alpha = 0.8)

            plt.ylabel('Sale Price',fontsize = f)

            plt.xlabel(col[i], fontsize = f)

            

            if len(df_train_explore[col[i]].unique()) > 20: #bin the continuous features

                df_train_explore[col[i]], bins=pd.cut(df_train_explore[col[i]], 10, retbins = True, duplicates = 'drop')

                

            grouped_train = df_train_explore.groupby(col[i])

            grouped_mean_train = grouped_train['SalePrice'].mean()   

            if grouped_mean_train.index.values.dtype == 'int64':

                x = grouped_mean_train.index.values

            else:

                x = (bins[0:len(bins)-1] + bins[1:len(bins)])/2

            plt.subplot(1, 4, 4)

            g = sns.lineplot(x, grouped_mean_train.values, color = 'seagreen',alpha = 0.8, marker = 'o')

            plt.ylabel('Mean Sale Price',fontsize = f)

            plt.xlabel(col[i], fontsize = f)

           

        else:#categorical features

            grouped_train = df_train_explore.groupby(col[i])

            grouped_test = df_test_explore.groupby(col[i])

            grouped_train_count = grouped_train.count()

            grouped_test_count = grouped_test.count()

            grouped_mean_train = grouped_train['SalePrice'].mean()   

            plt.subplot(1, 4, 1)

            g = sns.barplot(grouped_train_count.index.values, grouped_train_count['Id'].values, color = 'seagreen', alpha = 0.7)

            plt.ylabel('Frequency', fontsize = f)

            g.set_xticklabels(grouped_train_count.index.values, rotation = 90, fontsize = f)

            plt.title('Training set')

            plt.subplot(1, 4, 2)

            g = sns.barplot(grouped_test_count.index.values, grouped_test_count['Id'].values, color = 'seagreen', alpha = 0.7)

            g.set_xticklabels(grouped_test_count.index.values, rotation = 90, fontsize = f)

            plt.ylabel('Frequency',fontsize = f)

            g.set_xticklabels(grouped_test_count.index.values, rotation = 90, fontsize = f)

            plt.title('Test set')



            plt.subplot(1, 4, 3)

            g = sns.scatterplot(x = col[i], y = 'SalePrice', data = df_train_explore, color = 'seagreen', alpha = 0.8)

            plt.ylabel('Sale Price', fontsize = f)

            g.set_xticklabels(grouped_train_count.index.values, rotation = 90, fontsize = f)

            plt.xlabel('', fontsize = f)

            

            plt.subplot(1, 4, 4)

            g = sns.lineplot(grouped_mean_train.index.values, grouped_mean_train.values, color = 'seagreen', alpha = 0.8, marker = 'o')

            g.set_xticklabels(grouped_mean_train.index.values, rotation = 90, fontsize = f)

            plt.ylabel('Mean Sale Price', fontsize = f)

        plt.show()
eda(df_train, df_test)
plt.figure(figsize = (8, 3))

plt.subplot(1, 2, 1)

sns.distplot(df_train['SalePrice'], color = 'seagreen', hist_kws = {"alpha": 0.5})

plt.title('SalePrice')



plt.subplot(1, 2, 2)

sns.distplot(np.log1p(df_train['SalePrice']), color = 'seagreen', hist_kws = {"alpha": 0.5})

plt.title('Log(SalePrice)')

plt.show()
def clean_data(df):

    df = df.replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})

    df = df.replace({'Grvl': 1, 'Pave': 2})

    df['Alley'].fillna(0, inplace = True)

    df['GarageYrBlt'].fillna(0, inplace = True)

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    df = df.replace({'LotShape':{'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}})

    df['LotShape'].fillna(df['LotShape'].value_counts().idxmax(), inplace=True) #if missing fill with the most common value

    df['MSZoning'].fillna(df['MSZoning'].value_counts().idxmax(), inplace=True)

    df['Utilities'].fillna(df['Utilities'].value_counts().idxmax(), inplace=True)

    df = df.replace({'Utilities':{'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}})

    df['LandSlope'].fillna(df['LandSlope'].value_counts().idxmax(), inplace = True) 

    df = df.replace({'LandSlope':{'Gtl': 1, 'Mod': 2, 'Sev': 3}} )

    df['Exterior1st'].fillna(df['Exterior1st'].value_counts().idxmax(), inplace = True)

    df['Exterior2nd'].fillna(df['Exterior2nd'].value_counts().idxmax(), inplace = True)

    df['MasVnrType'].fillna(df['MasVnrType'].value_counts().idxmax(), inplace = True)

    df['MasVnrArea'].fillna(0, inplace = True)

    df['BsmtFinType1'].fillna('NoBsmt', inplace = True)

    df['BsmtFinType2'].fillna('NoBsmt', inplace = True)

    df['BsmtCond'].fillna(0, inplace = True)

    df['BsmtQual'].fillna(0, inplace = True)

    df['BsmtExposure'].fillna(0, inplace = True)

    df = df.replace({'BsmtExposure':{'Av': 3, 'Mn': 2, 'No': 1}})

    df['BsmtFinSF1'].fillna(0, inplace = True)

    df['BsmtFinSF2'].fillna(0, inplace = True)

    df['BsmtUnfSF'].fillna(0, inplace = True)

    df['TotalBsmtSF'].fillna(0, inplace = True)

    df = df.replace({'CentralAir':{'Y': 1, 'N': 0}})

    df['1stFlrSF'].fillna(0, inplace = True)

    df['2ndFlrSF'].fillna(0, inplace = True)

    df['LowQualFinSF'].fillna(0, inplace = True)

    df['BsmtFullBath'].fillna(0, inplace = True)

    df['BsmtHalfBath'].fillna(0, inplace = True)

    df['FullBath'].fillna(0, inplace = True)

    df['HalfBath'].fillna(0, inplace = True)

    df['KitchenQual'].fillna(df['KitchenQual'].value_counts().idxmax(),inplace = True)

    df['Functional'].fillna('Typ',inplace = True)

    df = df.replace({'Functional':{'Typ': 7, 'Min1': 6, 'Maj1': 5, 'Min2': 4, 'Mod': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}})

    df['FireplaceQu'].fillna(0, inplace = True)

    df['GarageType'].fillna('NoGarage', inplace = True)

    df['GarageFinish'].fillna(0, inplace = True)

    df = df.replace({'GarageFinish':{'Fin': 3, 'RFn': 2, 'Unf': 1}})

    df['GarageArea'].fillna(0, inplace = True)

    df['GarageCars'].fillna(0, inplace = True)

    df['GarageQual'].fillna(0, inplace = True)

    df['GarageCond'].fillna(0, inplace = True)

    df = df.replace({'PavedDrive':{'Y': 2, 'P': 1, 'N': 0}})

    df['Electrical'].fillna(df['Electrical'].value_counts().idxmax(), inplace = True)

    df['PoolQC'].fillna(0, inplace = True)

    df['Fence'].fillna('NoFence', inplace = True)

    df['MiscFeature'].fillna('NoMiF', inplace = True)

    df['SaleType'].fillna(df['SaleType'].value_counts().idxmax(), inplace = True)

    return df
def num_to_cat(df):

    #df['MSSubClass'] = df['MSSubClass'].astype(str)

    df['YrSold'] = df['YrSold'].astype(str)

    #df['MoSold'] = df['MoSold'].astype(str)

    return df
def remove_columns(df):

    col = ['Id', 'Utilities', 'Street', 'PoolArea' ,'LowQualFinSF', 'Alley', 'EnclosedPorch', '3SsnPorch', 'PoolQC',

           'KitchenAbvGr']

    df.drop(columns = col, inplace = True)

    return df
def add_features(df):

    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

    df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    

    df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    return df
def find_outliers(df):

    outliers = df[df['GrLivArea'] > 4000].index

    return outliers
def add_polynomial(df):

    col = ['OverallQual', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'GarageQual', 'KitchenQual',

           'TotalSF']

    for c in col:

        c_2 = c + '-2'

        c_3 = c + '-3'

        c_sqrt = c + '-sqrt'

        df[c_2] = df[c]**2

        df[c_3] = df[c]**3

        df[c_sqrt] = np.sqrt(df[c])

    return df   
def transf_skewed(df):

    numerical = [var for var in df.columns if df[var].dtype != 'object' and var not in ['Id', 'SalePrice']]

    skew_features = df[numerical].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]

    skew_index = high_skew.index

    for i in skew_index:

        df[i] = boxcox1p(df[i],  boxcox_normmax(df[i] + 1))       

    return df
def one_hot_encoding(df):

    df = pd.get_dummies(df)

    return df
def label_encoding(train):

    obj = [var for var in train.columns if train[var].dtype == 'object']

    for c in obj:

        le = LabelEncoder()

        train[c] = le.fit_transform(train[c])

    return train
s = df_train.shape[0]

y = df_train['SalePrice']

y_log = np.log1p(y)

df_train = df_train.drop(columns = 'SalePrice')



outliers=find_outliers(df_train)

y_log.drop(outliers, inplace = True)



X = pd.concat([df_train, df_test])
score = pd.DataFrame(columns = ['model', 'score'])
cleaned_X = clean_data(X)

remove_columns(cleaned_X)

label_encoding(cleaned_X)



X_train_TM = cleaned_X.iloc[:s, :]

X_test_TM = cleaned_X.iloc[s:, :]



X_train_TM.drop(outliers, inplace = True)



print('Missing values in the cleaned training set:', X_train_TM.isnull().sum().sum())

print('Missing values in the cleaned testing set:', X_test_TM.isnull().sum().sum())
xgb_model = xgb.XGBRegressor(learning_rate = 0.01, n_estimators = 5800,

                                     max_depth = 2, min_child_weight = 1,

                                     gamma = 0, subsample = 0.5,

                                     colsample_bytree = 0.5,

                                     objective = 'reg:linear', nthread = -1,

                                     scale_pos_weight = 1, seed = 27,

                                     reg_alpha = 0.2, random_state = 42)

sc = cross_val_score(xgb_model, X_train_TM, y_log, scoring = "neg_mean_squared_error", cv = 10)

np.sqrt(-sc.mean())

score = score.append({'model': 'XGBoost', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the XGBoost model:', np.sqrt(-sc.mean()))
xgb_model.fit(X_train_TM, y_log)



plt.style.use('seaborn') 

fig, ax1 = plt.subplots(figsize = (6, 5))

plot_importance(xgb_model, importance_type = 'weight', grid = 'on', height = 0.8, max_num_features = 25, color = 'seagreen',

                ax = ax1, alpha = 0.6)

plt.xticks(fontsize = 8)

plt.yticks(fontsize = 8)

plt.xlabel('F score', fontsize = 8)

plt.ylabel('Features', fontsize = 8)

plt.title('Feature importance', fontsize = 12)

plt.show()
rforest_model = RandomForestRegressor(criterion = 'mse', max_depth = 18,

           max_features = 0.4, min_samples_leaf = 1, min_samples_split = 2,

           min_weight_fraction_leaf = 0, n_estimators = 5500, random_state = 0)

sc = cross_val_score(rforest_model, X_train_TM, y_log, scoring = "neg_mean_squared_error", cv = 10)

np.sqrt(-sc.mean())

score = score.append({'model': 'RandomForest', 'score': np.sqrt(-sc.mean())}, ignore_index = True)



print('The score of the RandomForest model:', np.sqrt(-sc.mean()))
gbr_model = GradientBoostingRegressor(loss = 'lad', learning_rate = 0.025, n_estimators = 5500,

                                    subsample = 0.5, criterion = 'friedman_mse',

                                    min_samples_split = 7, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0,

                                    max_depth = 2, min_impurity_decrease = 0.0, min_impurity_split = None, init = None, 

                                    random_state = 42, max_features = 'sqrt', alpha = 0.9, verbose = 0, 

                                    max_leaf_nodes = None, warm_start = False, presort = 'auto', validation_fraction = 0.3, 

                                    n_iter_no_change = None, tol = 0.0001)

sc = cross_val_score(gbr_model, X_train_TM, y_log, scoring = "neg_mean_squared_error", cv = 10)

np.sqrt(-sc.mean())

score = score.append({'model': 'GradientBoosting', 'score': np.sqrt(-sc.mean())}, ignore_index = True)



print('The score of the GradientBoosting model:', np.sqrt(-sc.mean()))
lgb_model = lgb.LGBMRegressor(objective = 'regression', num_leaves = 5,

                              learning_rate = 0.009, n_estimators = 5500,

                              max_bin = 200, bagging_fraction = 0.705,

                              bagging_freq = 5, feature_fraction = 0.2,

                              feature_fraction_seed = 9, bagging_seed = 9,

                              min_data_in_leaf = 2)

sc = cross_val_score(lgb_model, X_train_TM, y_log, scoring = "neg_mean_squared_error", cv = 10)

np.sqrt(-sc.mean())

score = score.append({'model': 'LightGBM', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the LightGBM model:', np.sqrt(-sc.mean()))
cleaned_X = clean_data(X)

cleaned_X = add_features(cleaned_X)

cleaned_X = add_polynomial(cleaned_X)

cleaned_X = num_to_cat(cleaned_X)

cleaned_X = transf_skewed(cleaned_X)



numerical = [var for var in cleaned_X if cleaned_X[var].dtype != 'object']

columns_scaling = (cleaned_X[numerical].max() > 3).index.values

robust_scaler = RobustScaler()

cleaned_X[columns_scaling] = robust_scaler.fit_transform(cleaned_X[columns_scaling])



remove_columns(cleaned_X)

cleaned_X = one_hot_encoding(cleaned_X)



X_train_LM = cleaned_X.iloc[:s, :]

X_test_LM = cleaned_X.iloc[s:, :]



X_train_LM.drop(outliers, inplace = True)



print('Missing values in the cleaned training set:', X_train_LM.isnull().sum().sum())

print('Missing values in the cleaned testing set:', X_test_LM.isnull().sum().sum())
lasso_model = linear_model.Lasso(alpha = 0.00046, max_iter = 500, fit_intercept = True)

sc = cross_val_score(lasso_model, X_train_LM, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'Lasso', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the Lasso model:', np.sqrt(-sc.mean()))
elastic_model = linear_model.ElasticNet(alpha = 0.00046, l1_ratio = 1, fit_intercept = True, normalize = False, 

                                        precompute = False, max_iter = 500, copy_X = True, tol = 0.001, warm_start = False,

                                        positive = False, random_state = 42, selection = 'cyclic')

sc = cross_val_score(elastic_model, X_train_LM, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'ElasticNet', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the ElasticNet model:', np.sqrt(-sc.mean()))
lassolars_model = LassoLarsIC(criterion = 'aic', eps = 0.3)

sc = cross_val_score(lassolars_model, X_train_LM, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'LassoLars', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the LassoLars model:', np.sqrt(-sc.mean()))
svr_model = SVR(C= 20, epsilon = 0.0117, gamma = 0.00041)

sc = cross_val_score(svr_model, X_train_LM, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'SVR', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the SVR model:', np.sqrt(-sc.mean()))
bridge_model = BayesianRidge(alpha_1 = 45, alpha_2 = 0, compute_score = False,

                            copy_X = True, fit_intercept = True, lambda_1 = 0, lambda_2 = 0,

                            n_iter = 500, normalize = False, tol = 0.001, verbose = False)

sc = cross_val_score(bridge_model, X_train_LM, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'BayesianRidge', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the BayesianRidge model:', np.sqrt(-sc.mean()))
cleaned_X = clean_data(X)

cleaned_X = add_features(cleaned_X)

cleaned_X = add_polynomial(cleaned_X)

cleaned_X = num_to_cat(cleaned_X)

cleaned_X = transf_skewed(cleaned_X)



remove_columns(cleaned_X)

cleaned_X = one_hot_encoding(cleaned_X)



X_train_LM2 = cleaned_X.iloc[:s, :]

X_test_LM2 = cleaned_X.iloc[s:, :]



X_train_LM2.drop(outliers, inplace = True)



print('Missing values in the cleaned training set:', X_train_LM2.isnull().sum().sum())

print('Missing values in the cleaned testing set:', X_test_LM2.isnull().sum().sum())
kridge_model = KernelRidge(alpha = 7.5)

sc = cross_val_score(kridge_model,  X_train_LM2, y_log, scoring = "neg_mean_squared_error", cv = 10)

score = score.append({'model': 'KernelRidge', 'score': np.sqrt(-sc.mean())}, ignore_index = True)

print('The score of the KernelRidge model:', np.sqrt(-sc.mean()))                                             
score
#Tree based models

X_train, X_val, y_train, y_val = train_test_split(X_train_TM, y_log, test_size = 0.3,random_state = 0)



xgb_model.fit(X_train, y_train)

y_val_pred_xgb = np.expm1(xgb_model.predict(X_val))

print('XGB:', np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_xgb)))



gbr_model.fit(X_train, y_train)

y_val_pred_gbr = np.expm1(gbr_model.predict(X_val))

gbr_score = np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_gbr))

print('GradientBoosting:', gbr_score)



lgb_model.fit(X_train, y_train)

y_val_pred_lgb = np.expm1(lgb_model.predict(X_val))

lgb_score = np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_lgb))

print('LightGBM:', lgb_score)



#Linear models

X_train, X_val, y_train, y_val=train_test_split(X_train_LM,  y_log, test_size = 0.3, random_state = 0)



lasso_model.fit(X_train, y_train)

y_val_pred_lasso = np.expm1(lasso_model.predict(X_val))

print('Lasso:', np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_lasso)))



elastic_model.fit(X_train, y_train)

y_val_pred_elastic = np.expm1(elastic_model.predict(X_val))

print("ElasticNet:",np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_elastic)))



bridge_model.fit(X_train, y_train)

y_val_pred_bridge = np.expm1(bridge_model.predict(X_val))

print('BRidge:',np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_bridge)))



svr_model.fit(X_train, y_train)

y_val_pred_svr = np.expm1(svr_model.predict(X_val))

print('SVR:',np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_svr)))



#KernelRidge

X_train, X_val, y_train, y_val = train_test_split(X_train_LM2,  y_log, test_size=0.3,random_state = 0)



kridge_model.fit(X_train, y_train)

y_val_pred_kridge = np.expm1(kridge_model.predict(X_val))

print('Kridge:', np.sqrt(mean_squared_log_error(np.expm1(y_val), y_val_pred_kridge)))
d = {'y': np.expm1(y_val), 'XGB': y_val_pred_xgb, 'GBR': y_val_pred_gbr, 'LGB':  y_val_pred_lgb,

     'Lasso': y_val_pred_lasso, 'ElasticNet': y_val_pred_elastic, 'BRidge': y_val_pred_bridge,

     'SVR':  y_val_pred_svr, 'KRidge': y_val_pred_kridge}



prediction_val = pd.DataFrame(data = d)



prediction_val_x = prediction_val[['ElasticNet','BRidge','GBR','SVR']]

prediction_val_y = prediction_val['y']

prediction_val.head(10)
reg = LinearRegression()

reg.fit(prediction_val_x, prediction_val_y)

stacked_prediction = reg.predict(prediction_val_x)

c = prediction_val_x.columns.values

for i in range(0, len(reg.coef_)):

    print(np.round(reg.coef_[i], decimals = 3).astype(str) + ' x ' + c[i])

print('Score:' + np.sqrt(mean_squared_log_error(np.expm1(y_val), stacked_prediction)).astype(str))
plt.style.use('default') 

l = range(min(np.expm1(y_val).astype(int)), max(np.expm1(y_val).astype(int)))

plt.figure(figsize = (10, 6))

plt.plot(np.expm1(y_val), y_val_pred_elastic, color = 'aqua', marker = 'o', linestyle = 'None', alpha = 0.5, label = 'ElasticNet')

plt.plot(np.expm1(y_val), y_val_pred_bridge, color = 'mediumpurple', marker = 'o', linestyle = 'None', alpha = 0.5, label = 'BRidge')

plt.plot(np.expm1(y_val), y_val_pred_gbr, color = 'pink', marker = 'o', linestyle = 'None', alpha = 0.5, label = 'GBR')

plt.plot(np.expm1(y_val), y_val_pred_svr, color = 'seagreen', marker = 'o', linestyle = 'None', alpha = 0.5, label = 'SVR')

plt.plot(np.expm1(y_val), pd.Series(stacked_prediction), color = 'gray', marker = 'o', linestyle = 'None', markerfacecolor = 'white', label = 'Stack')

plt.plot(l, l, color = 'gray', linestyle = 'dashed')

plt.xlabel('SalePrice', fontsize = 16)

plt.ylabel('Prediction', fontsize = 16)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.legend(fontsize = 12)

plt.show()
xgb_model.fit(X_train_TM, y_log)

y_test_pred_xgb = np.expm1(xgb_model.predict(X_test_TM))
gbr_model.fit(X_train_TM, y_log)

y_test_pred_gbr = np.expm1(gbr_model.predict(X_test_TM))
lgb_model.fit(X_train_TM, y_log)

y_test_pred_lgb = np.expm1(lgb_model.predict(X_test_TM))
lasso_model.fit(X_train_LM, y_log)

y_test_pred_lasso = np.expm1(lasso_model.predict(X_test_LM))
elastic_model.fit(X_train_LM, y_log)

y_test_pred_elastic = np.expm1(elastic_model.predict(X_test_LM))

bridge_model.fit(X_train_LM, y_log)

y_test_pred_bridge = np.expm1(bridge_model.predict(X_test_LM))
svr_model.fit(X_train_LM, y_log)

y_test_pred_svr = np.expm1(svr_model.predict(X_test_LM))
kridge_model.fit(X_train_LM2, y_log)

y_test_pred_kridge = np.expm1(kridge_model.predict(X_test_LM2))
av =(y_test_pred_xgb + y_test_pred_elastic + y_test_pred_kridge + y_test_pred_bridge + y_test_pred_gbr + y_test_pred_lgb +\

    y_test_pred_svr)/7

pred_av = {'Id': df_test['Id'], 'SalePrice': av}

prediction_av = pd.DataFrame(data = pred_av)

prediction_av.head(5)
d = {'XGB': y_test_pred_xgb, 'GBR': y_test_pred_gbr, 'LGB':  y_test_pred_lgb, 'Lasso': y_test_pred_lasso,

     'ElasticNet': y_test_pred_elastic, 'BRidge': y_test_pred_bridge, 'SVR':  y_test_pred_svr, 'KRidge': y_test_pred_kridge}

prediction_test = pd.DataFrame(data = d)

prediction_test.head()
prediction_test_x = prediction_test[['ElasticNet','BRidge','GBR','SVR']]

prediction_test_x.head()
stacked_prediction_test = reg.predict(prediction_test_x)

df_stacked_prediction_test = pd.DataFrame(data = {'Id': df_test['Id'], 'SalePrice': stacked_prediction_test})

df_stacked_prediction_test.head()
df_stacked_prediction_test.to_csv('mysubmission.csv', index = False)