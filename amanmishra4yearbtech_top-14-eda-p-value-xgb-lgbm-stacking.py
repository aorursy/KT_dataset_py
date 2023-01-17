# importing libraries



# preprocessing and validation libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn 

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler,LabelEncoder as le

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, KFold, learning_curve



# model related libraries

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.decomposition import PCA

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor

from xgboost import XGBRegressor 

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor



from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error

import statsmodels.api as sm  # for p values analysis

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

train_data.head(5) # printing some of them
# test data info

test_data.head()
# data exploration

print('no.of examples in train data : ', len(train_data))

print('no.of examples in test data : ', len(test_data))

print('no.of features in data : ', train_data.shape[1]-1)



# Firstly remove the SalePrice Column in train data in order to merge train and test



# take it as y_train

y_train = train_data['SalePrice']



# X_train

X_train = train_data.drop('SalePrice',axis=1)



# concatenate the test data an train data to apply same operations

dataset = pd.concat(objs = [X_train,test_data],axis=0,sort=False).reset_index(drop=True)

print(len(dataset))

dataset.head(5)
# finding the no. of null values in both train and test

dataset_null_val = dataset.isnull().sum()



# finding those columns which contains null values

print('For Dataset \n')

print('{} no of features in  dataset contain missing values \n'.format(len(dataset_null_val.values[dataset_null_val.values !=0])))

print('column names with null values {}\n'.format(dataset.columns[dataset_null_val.values !=0]))      

#Percentage of NAN Values 

NAN = [(c, dataset[c].isna().mean()*100) for c in dataset]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN
NAN = NAN[NAN.percentage > 50]

NAN.sort_values("percentage", ascending=False)
#Drop PoolQC, MiscFeature, Alley and Fence features

dataset = dataset.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

dataset.shape[1]
object_columns_df = dataset.select_dtypes(include=['object'])

numerical_columns_df =dataset.select_dtypes(exclude=['object'])

object_columns_df.dtypes
numerical_columns_df.dtypes
#Number of null values in each feature

null_counts = object_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']

object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')
columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']

#fill missing values for each column (using its own most frequent value)

object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])
#Number of null values in each feature

null_counts = numerical_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
print((numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']).median())  # median age of a house

print(numerical_columns_df["LotFrontage"].median())
numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)

numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)

numerical_columns_df= numerical_columns_df.fillna(0)
object_columns_df['Utilities'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Utilities'].value_counts() 
object_columns_df['Street'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Street'].value_counts() 
object_columns_df['Condition2'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Condition2'].value_counts() 
object_columns_df['RoofMatl'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['RoofMatl'].value_counts() 
object_columns_df['Heating'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Heating'].value_counts() #======> Drop feature one Type
object_columns_df = object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)
numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

numerical_columns_df['Age_House'].describe()
Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]

Negatif
numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'],'YrSold' ] = 2009

numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

numerical_columns_df['Age_House'].describe()
numerical_columns_df.head()
bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,

            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6

            }

object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)

object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)

object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)

object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)

object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)

object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)

object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)

object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)

object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)

object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)

object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)

object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)

object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)

object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)



PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}

object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)
#Select categorical features

rest_object_columns = object_columns_df.select_dtypes(include=['object'])

#Using One hot encoder

object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns) 
object_columns_df.head()
numerical_columns_df.describe()
# finding min values

numerical_columns_df.describe().iloc[3,:]
# finding max values

numerical_columns_df.describe().iloc[7,:]
# we will use robust scaler of sklearn as it is less prone to outliers in data



rs = RobustScaler()

scaled_features = rs.fit_transform(numerical_columns_df.iloc[:,1:])



numerical_columns_df.iloc[:,1:] = scaled_features
numerical_columns_df.describe()
# final conactenate of both features

df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)

df_final.head()
# seprating train and test data features

# firstly drop id

df_final = df_final.drop('Id',axis=1)



# train features

x_train = df_final.iloc[0:len(train_data),:]

# test feature

x_test = df_final.iloc[len(train_data):,:]

len(x_train)
## firstly let's plot the correlation of each individual on SalePrice 



plt.figure(figsize = (16,16))



# making heatmap of correlation

sns.heatmap(pd.concat([x_train.iloc[:,:30],y_train]).corr(),annot = True, fmt = '.2f')
# checking importance using f-static

import statsmodels.api as sm

X2 = sm.add_constant(x_train)

regressor_OLS = sm.OLS(endog = y_train, exog = X2).fit()

regressor_OLS.summary()
#  applying PCA for all features

# identifying which features adds how much variance to data



covar_matrix = PCA(n_components = df_final.shape[1]) #we have 8 numerical features

covar_matrix.fit(StandardScaler().fit_transform(df_final))

variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios



var=np.cumsum(np.round(variance, decimals=3)*100)

#components = covar_matrix.fit_transform(dataset[['Answers','Reputation','Tag','Username','Views']])

plt.plot(var)#cumulative sum of variance explained with [n] features

plt.title('Variance percenatage vs no. of features')

var
#plotting the distribution of sale price

g = sns.boxplot(train_data['SalePrice'])

# looks like the distribution of price is skewed 

f = sns.violinplot(train_data['SalePrice'], color = 'r')
# using k folds cross validation with 5 splits

kfold = KFold( n_splits = 5)

# import from sklearn

linear_reg = LinearRegression()

MSLE = cross_val_score(linear_reg,x_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error), cv = kfold , n_jobs =-1)

mean_squared_log_err = np.mean(np.sqrt(MSLE))

print('mean squared log error is : ',mean_squared_log_err)
import math

# now fitting the model on entire data for final predictions

linear_reg.fit(x_train,y_train)



# let's find out the error on whole train data to see is there any overfitting of model or not

math.sqrt(mean_squared_log_error(y_train,linear_reg.predict(x_train)))
print('R-squared value for simple regression on train data is ', r2_score(y_train,linear_reg.predict(x_train)))
plt.figure(figsize = (15,50))

ft_importances_lr  = pd.Series(linear_reg.coef_ , index = x_train.columns)

ft_importances_lr.plot(kind = 'barh')

linear_reg_pred = linear_reg.predict(x_test)
ridge = Ridge()



# ridge has a hyperparameter lambda 

parameters = {'alpha' : [1,2]}  # possible values to search



# applying GridSearch cv



rid = GridSearchCV(ridge,param_grid = parameters, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



rid.fit(x_train,y_train) # fitting model





print(math.sqrt(rid.best_score_))

print(rid.best_params_)
rid_best = rid.best_estimator_  # Getting best estimator based on cv
print('R-squared value for ridge regression on train data is ', r2_score(y_train,rid_best.predict(x_train)))
plt.figure(figsize = (15,50))

ft_importances_rid  = pd.Series(rid_best.coef_ , index = x_train.columns)

ft_importances_rid.plot(kind = 'barh')

ridge_pred = rid_best.predict(x_test)
lasso = Lasso()



# lasso also has a hyperparameter lambda 

parameters = {'alpha' : [0.1,1,2]}  # possible values to search



# applying GridSearch cv



las = GridSearchCV(lasso,param_grid = parameters, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



las.fit(x_train,y_train) # fitting model





print(math.sqrt(las.best_score_))

print(las.best_params_)
las_best = las.best_estimator_  # Getting best estimator based on cv
print('R-squared value for lasso regression on train data is ', r2_score(y_train,las_best.predict(x_train)))
plt.figure(figsize = (15,50))

ft_importances_las  = pd.Series(las_best.coef_ , index = x_train.columns)

ft_importances_las.plot(kind = 'barh')

# predicting on test data

lasso_pred = las_best.predict(x_test)
elastic = ElasticNet()



# lasso also has a hyperparameter lambda 

parameters = {'alpha' : [ .5],

               'l1_ratio' : [1e-5,0.1,1]}  # possible values to search



# applying GridSearch cv



elas = GridSearchCV(elastic,param_grid = parameters, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_error), n_jobs= -1, verbose = 1)



elas.fit(x_train,y_train) # fitting model





print(math.sqrt(elas.best_score_))

print(elas.best_params_)
elas_best = elas.best_estimator_  # Getting best estimator based on cv
print('R-squared value for elasticnet regression on train data is ', r2_score(y_train,elas_best.predict(x_train)))
# plotting feature representation for elasticnet

plt.figure(figsize = (15,50))

ft_importances_elas  = pd.Series(elas_best.coef_ , index = x_train.columns)

ft_importances_elas.plot(kind = 'barh')

# predicting on test data

elastic_pred = elas_best.predict(x_test)
### SVR classifier

SVMC = SVR(kernel = 'rbf')

svc_param_grid = { 

                  'gamma': [500],

                  'C': [500,1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



gsSVMC.fit(x_train,y_train)







# Best score

print(math.sqrt(gsSVMC.best_score_))

print(gsSVMC.best_params_)
svm_best = gsSVMC.best_estimator_  # Getting best estimator based on cv
print('R-squared value for Supprt vector regression on train data is ', r2_score(y_train,svm_best.predict(x_train)))
# predicting on test data

svm_pred = svm_best.predict(x_test)
### SVR classifier

SVML = SVR(kernel = 'linear')

svc_param_grid = { 

                  'gamma': [0.001],

                  'C': [10]}



gsSVML = GridSearchCV(SVML,param_grid = svc_param_grid, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



gsSVML.fit(x_train,y_train)







# Best score

print(math.sqrt(gsSVML.best_score_))

print(gsSVML.best_params_)
svml_best = gsSVML.best_estimator_  # Getting best estimator based on cv
print('R-squared value for Supprt vector regression on train data is ', r2_score(y_train,svml_best.predict(x_train)))
# predicting on test data

svml_pred = svml_best.predict(x_test)
# first drfining random state it is needed for all tree based models if you want to produce reproducible results

random_state = 2



# now call model from sklearn library

DT = DecisionTreeRegressor(random_state=random_state)



# Applying cross validation to see the validation score

MSLE = cross_val_score(DT,x_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error), cv = kfold , n_jobs =-1)

mean_squared_log_err = np.mean(np.sqrt(MSLE))

print('mean squared log error is : ',mean_squared_log_err)





# now fitting the model on entire data for final predictions

DT.fit(x_train,y_train)



# let's find out the error on whole train data to see is there any overfitting of model or not

math.sqrt(mean_squared_log_error(y_train,DT.predict(x_train)))
# predicting on test data

DT_pred = DT.predict(x_test)


# RFC

RFC = RandomForestRegressor(random_state = random_state)





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [2],

              "min_samples_split": [5],

              "min_samples_leaf": [2],

              "bootstrap": [True],

              "n_estimators" :[200],

              }



# Applying grid search 

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



gsRFC.fit(x_train,y_train)





# Best score



print(math.sqrt(gsRFC.best_score_))

print(gsRFC.best_params_)
RFC_best = gsRFC.best_estimator_  # Getting best estimator based on cv
# predicting on test data

rfc_pred = RFC_best.predict(x_test)
# Gradient boosting



GBC = GradientBoostingRegressor(random_state = random_state)

gb_param_grid = {}



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error), n_jobs= -1, verbose = 1)



gsGBC.fit(x_train,y_train)





# Best score



print(math.sqrt(gsGBC.best_score_))
GBC_best = gsGBC.best_estimator_
# predicting on test data

gbc_pred = GBC_best.predict(x_test)


# using desicion tree as base regressor

DT1 = DecisionTreeRegressor(random_state=random_state)



adaDTC = AdaBoostRegressor(DT1, random_state=random_state)



ada_param_grid = {"n_estimators":[100,500],

              "learning_rate":  [ 0.01]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring= sklearn.metrics.make_scorer(mean_squared_log_error) , n_jobs= -1, verbose = 1)



gsadaDTC.fit(x_train,y_train)



Ada_best = gsadaDTC.best_estimator_
## checking the cross val score of best estimator



# finding best params

print(Ada_best.get_params())



# finding ccv score

math.sqrt(cross_val_score(Ada_best, x_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error) , cv = kfold , n_jobs =-1).mean())
# predicting on test data

ada_pred = Ada_best.predict(x_test)
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=1.5, n_estimators=2500,

             n_jobs=1, nthread=None, objective='reg:linear',

             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

             silent=None, subsample=0.8, verbosity=1)





MSLE = cross_val_score(xgb,x_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error), cv = kfold , n_jobs =-1)

mean_squared_log_err = np.mean(np.sqrt(MSLE))

print('mean squared log error is : ',mean_squared_log_err)
# now fitting the model on entire data for final predictions

xgb.fit(x_train,y_train)



# predicting on test data

xgb_pred = xgb.predict(x_test)
lgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=11000, 

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.4, 

                                       )

MSLE = cross_val_score(lgbm,x_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error), cv = kfold , n_jobs =-1)

mean_squared_log_err = np.mean(np.sqrt(MSLE))

print('mean squared log error is : ',mean_squared_log_err)
# now fitting the model on entire data for final predictions

lgbm.fit(x_train,y_train,eval_metric= sklearn.metrics.make_scorer(mean_squared_log_error))



# predicting on test data

lgbm_pred = lgbm.predict(x_test)
# plotting learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(lgbm,"LGBM learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(xgb,"xgboost learning curves",x_train,y_train,cv=kfold)
nrows = ncols = 2

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,50))



names_classifiers = [("AdaBoosting", Ada_best),("XGB",xgb),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]



nclassifier = 0

for row in range(nrows):

    for col in range(ncols):

        name = names_classifiers[nclassifier][0]

        classifier = names_classifiers[nclassifier][1]

        indices = np.argsort(classifier.feature_importances_)[::-1][:40]

        g = sns.barplot(y=x_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])

        g.set_xlabel("Relative importance",fontsize=12)

        g.set_ylabel("Features",fontsize=12)

        g.tick_params(labelsize=9)

        g.set_title(name + " feature importance")

        nclassifier += 1
# making panda series of predictions on test data

test_LReg = pd.Series(linear_reg_pred, name="linear_reg")

test_ridge = pd.Series(ridge_pred, name="ridge")

test_lasso = pd.Series(lasso_pred, name="lasso")

test_elastic = pd.Series(elastic_pred, name="elasticnet")

test_SVMR = pd.Series(svm_pred, name="SVC-R")

test_SVML = pd.Series(svml_pred, name="SVC-L")

test_DT = pd.Series(DT_pred, name="DT")

test_RFC = pd.Series(rfc_pred, name="RFC")

test_AdaC = pd.Series(ada_pred, name="Ada")

test_GBC = pd.Series(gbc_pred, name="GBC")

test_XGB = pd.Series(xgb_pred, name="XGB")

test_LGBM = pd.Series(lgbm_pred, name="LGBM")







# Concatenate all regressors results

ensemble_results = pd.concat([test_LReg, test_ridge, test_lasso, test_elastic, 

                              test_SVMR, test_SVML, test_DT, test_RFC,test_GBC,

                              test_AdaC, test_XGB, test_LGBM ],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True, fmt = '.2f')
## stacking will be automated by using vecstack library



from vecstack import stacking



# firstly defining base models with optimal parameters find using grid search cv



models = [ linear_reg,

          

           Ridge(alpha = 1),

          

           Lasso(alpha = 0.1),

          

           ElasticNet(alpha = 0.5, l1_ratio = 1e-5),

          

           SVR(kernel = 'rbf', C= 500, gamma = 500 ),

          

           SVR(kernel = 'linear',C = 10, gamma = 0.001),

           

           DT,  # Decision tree



           

           RandomForestRegressor(random_state=2, n_jobs=-1,

                                 min_samples_split= 5,

                                 min_samples_leaf = 2,

                                  bootstrap = True,

                                n_estimators=120,

                                 max_features = 200,

                               max_depth= 2),

         

         

          GradientBoostingRegressor(random_state = random_state),

          

          AdaBoostRegressor(DT1, random_state=random_state, learning_rate = 0.01), 

          

          xgb,

          

          lgbm

        

         ]


# Here Mean squared error will be used as the metric



# S_train, S_test will be the out of fold predictions which will be used for meta model





S_train, S_test = stacking(models,                   

                           x_train, y_train, x_test,   

                           regression= True, 

     

                           mode='oof_pred_bag', 

       

                           needs_proba=False,

         

                           save_dir=None, 

            

                           metric= mean_squared_error, 

    

                           n_folds=5, 

                 

                           stratified=False,

            

                           shuffle=True,  

            

                           random_state=0,    

         

                           verbose=2)
# defining metal model as linear reg, it is sometimes called as blending



meta_model1 = LinearRegression()

    

meta_model1.fit(S_train, y_train)

y_pred1 = meta_model1.predict(S_train)

stack_pred1 = meta_model1.predict(S_test)



print('Final train prediction score: [%.8f]' % math.sqrt(mean_squared_log_error(y_train, y_pred1)))
plt.figure(figsize = (10,10))

model_importances  = pd.Series(meta_model1.coef_ )

model_importances.plot(kind = 'barh')
# making weighted predictions by combine both

weighted_pred = (0.45*xgb_pred + 0.55*lgbm_pred)
# taking best 4 models



models1 = [ 

         

         GradientBoostingRegressor(random_state = random_state),

     

          xgb,

          

          lgbm

        

         ]
# doing same tasks for these models also



S1_train, S1_test = stacking(models1,                   

                           x_train, y_train, x_test,   

                           regression= True, 

     

                           mode='oof_pred_bag', 

       

                           needs_proba=False,

         

                           save_dir=None, 

            

                           metric= mean_squared_error, 

    

                           n_folds=5, 

                 

                           stratified=False,

            

                           shuffle=True,  

            

                           random_state=0,    

         

                           verbose=2)
# Again using metal model as linear reg



meta_model2 = LinearRegression()

    

meta_model2.fit(S1_train, y_train)

y_pred2 = meta_model2.predict(S1_train)

stack_pred2 = meta_model2.predict(S1_test)



print('Final train prediction score: [%.8f]' % math.sqrt(mean_squared_log_error(y_train, y_pred2)))
# checking weights assigned to each system

plt.figure(figsize = (10,10))

model_importances  = pd.Series(meta_model2.coef_ )

model_importances.plot(kind = 'barh')
# checking cross validation score for this model

MSLE = cross_val_score(meta_model2,S1_train, y = y_train, scoring = sklearn.metrics.make_scorer(mean_squared_log_error), cv = kfold , n_jobs =-1)

mean_squared_log_err = np.mean(np.sqrt(MSLE))

print('mean squared log error is : ',mean_squared_log_err)
# submitting predictions finally using weighted

sample_submission['SalePrice'] = (0.55*stack_pred2 + 0.45*weighted_pred)

sample_submission.to_csv('stacking.csv',index = False)