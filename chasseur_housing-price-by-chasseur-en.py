import pandas as pd

import numpy as np

import scipy.stats as ss

from decimal import Decimal as Dec





import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns



from IPython.display import FileLinks

from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

import lightgbm as lgb



from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNetCV, Ridge, Lasso

from sklearn.kernel_ridge import KernelRidge



from sklearn.metrics import make_scorer



%matplotlib inline
## Load of datas

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



## Shape of the datas

print("The training set shape is : {} and the test set shape is : {}\n".format(df_train.shape, df_test.shape))
# Saving the index of element from the training set

train_index = df_train.index



# Merge of training and test set

df = df_train.append(df_test).reset_index(drop = True)



# Delection of the variable ID from "df"

df.drop("Id", axis = 1, inplace = True)
def missing(df):

    df_null = pd.DataFrame([df.isnull().sum(),round(100*df.isnull().sum()/ len(df), 2), df.dtypes]).transpose().reset_index()

    df_null.columns = ["variable", "valeur_NA", "Pourcentage_NA", "type"]

    df_null = df_null[df_null.valeur_NA != 0].sort_values("valeur_NA",ascending = False).reset_index(drop = True)

    return df_null
missing(df)
# Mechanic missings generate by the merge by - 1000

df["SalePrice"]=df["SalePrice"].fillna(-1000)
# Variables with more that 30% of missings

list(df.columns[df.isnull().sum() / len(df) > 0.30])
# Imputation of quantitative variables

varQuant = ["GarageYrBlt","GarageArea","GarageCars","BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF","TotalBsmtSF", "BsmtFullBath", 

            "BsmtHalfBath","MasVnrArea"]

df[varQuant] = df[varQuant].apply(lambda x: x.fillna(0),axis = 1)

del varQuant

# Imputation of qualitatives variables

varQual = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",

          "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2","MasVnrType","MSSubClass"]

df[varQual] = df[varQual].apply(lambda x: x.fillna('Absent'),axis = 1)

del varQual
# Variables with missing values ​​unexplained by metadata

missing(df)
# Delete the variable "Utilities"

print(pd.crosstab(df.Utilities,columns = "Utilities"))

df.drop("Utilities", axis = 1, inplace = True)



# Neighbor median imputation for LotFrontage for the quantitative variable

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# Mode imputation for qualitative variables

#VarQual = ["Functional","Electrical","KitchenQual","Exterior1st",,"Exterior2nd","SaleType"]

#df[VarQual].apply(lambda x: x.fillna(x.mode()[0], inplace = True), axis = 1)



# Alternatively

df["Functional"] = df["Functional"].fillna(df["Functional"].mode()[0])

df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])

df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])

df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])

df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])

df["MSZoning"] = df["MSZoning"].fillna(df["MSZoning"].mode()[0])

print("There still {} missing values".format(df.isnull().sum().sum()))
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.scatter(x = df['GrLivArea'][train_index] + df["TotalBsmtSF"][train_index],

           y = df['SalePrice'][train_index])

plt.ylabel('Sale Price', fontsize=12)

plt.xlabel('Total Area', fontsize=12)

plt.gca().set_title('Prix Vs surface totale')



plt.subplot(1,2,2)

plt.scatter(x = df['GrLivArea'][train_index] + df["TotalBsmtSF"][train_index],

           y = df['SalePrice'][train_index])

# Outliers

df1 = df.loc[train_index,['GrLivArea',"TotalBsmtSF",'SalePrice']]

t = list(df1.loc[(df1['GrLivArea'] + df1["TotalBsmtSF"] > 6000) & (df1['SalePrice'] < 300000)].index)



plt.scatter(df1['GrLivArea'][t[0]] + df1["TotalBsmtSF"][t[0]],df1['SalePrice'][t[0]],

            color = 'r', marker = 'D' , s = 150, alpha= 0.5)

plt.scatter(df1['GrLivArea'][t[1]] + df1["TotalBsmtSF"][t[1]],df1['SalePrice'][t[1]],

            color = 'r', marker = 'D' , s = 150 , alpha=0.5)



plt.ylabel('Sale Price', fontsize=12)

plt.xlabel('Total Area', fontsize=12)

plt.gca().set_title('Identification des outliers')



plt.subplots_adjust( wspace = 0.4)

plt.show()
# deleting outliers

df.drop(t, axis = 0, inplace = True)

train_index = [i for i in train_index if i not in t]
# Continues' variables

VarCont = ['1stFlrSF','2ndFlrSF', 'BsmtUnfSF', 'EnclosedPorch', 'BsmtFinSF1', 'GarageArea', 

           'GrLivArea','BsmtFinSF2','LotArea', 'LotFrontage', 'LowQualFinSF','MasVnrArea',

           'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF',

           'WoodDeckSF']

### Continuous variable plots

for  col in VarCont:  



    f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {"height_ratios": (.15, .85)})

    sns.distplot(df[col], color = 'Sienna', ax = ax_hist)

    sns.boxplot(df[col], color = 'Sienna', ax = ax_box)

    ax_box.set(xlabel='')

    plt.show()
def whisker_based_outliers(data, whisker = 1.5):

    import numpy as np

    import pandas as pd

    import scipy.stats as ss

    from statsmodels.stats import stattools as st

    """

    This function only handle Pandas DataFame. It helps to identify the numbers of outliers within numeric

    variables of the data set.

    The function is base on a whisker outliers approach. That is, all observations out of whisker are strange

    and must be tag.

    Originaly, I wrote this function to help me quickly filter get an idea about how severe are outliers in data.

    One can use this information as explanatory variable to tag how bizare is an observation.

    """

    # turning data into numpy columns array or DataFrame

    if len(data.shape) == 1:

        data = data[:,None]

    

    # restrict data to numerics variable type

    df = data[data.columns[data.dtypes.values !="object"]]

    

    # Lower and Upper wkisker

    ## computation of a MedCouple (needed to ajusted the length of the whisker in case of non symetric distribution)

    mc = st.medcouple(df, axis = 0)

    

    ## Ajusted coefficient fo whisker

    lo_whisker = np.vectorize(lambda x: whisker*np.exp(-4*x) if x >= 0 else whisker*np.exp(-3*x))

    lo_whisker = lo_whisker(mc)

    up_whisker = np.vectorize(lambda x: whisker*np.exp(3*x) if x >= 0 else whisker*np.exp(4*x))

    up_whisker = up_whisker(mc)

            

    lo_whisk = np.maximum(np.percentile(df, 25, axis = 0) -  ss.iqr(df, axis = 0)* lo_whisker, np.min(df, axis = 0))

    up_whisk = np.minimum(np.percentile(df, 75, axis = 0) +  ss.iqr(df, axis = 0)* up_whisker, np.max(df, axis = 0))

    

    # Identification of columns with outliers

    df = df.apply(lambda x: (x < lo_whisk) | (x > up_whisk), axis = 1)

    return df.sum(axis = 1)
df['whisker'] = whisker_based_outliers(df[VarCont])

df['whisker'].value_counts()
# Discrete and categorical variables 

VarDist = ['3SsnPorch', 'BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','Fireplaces', 'FullBath',

           'GarageCars','HalfBath',  'KitchenAbvGr', 'TotRmsAbvGrd','FireplaceQu','BsmtQual',

           'BsmtCond', 'GarageQual', 'GarageCond','ExterQual', 'ExterCond','HeatingQC','PoolQC',

           'KitchenQual','BsmtFinType1','BsmtFinType2','GarageFinish','OverallCond', 'MSSubClass',

           'OverallQual','BsmtExposure','Fence','LandSlope']



# Box plots of SalePrice by explanatories

for var in VarDist:

    ax = sns.boxplot(x = var, y = "SalePrice" , data = df.loc[train_index,])

    plt.show()
plt.figure(figsize=(10,5))

# Building date

df.YearBuilt.value_counts().sort_index().plot();plt.legend(loc='best')

# Renovation date

df.YearRemodAdd.value_counts().sort_index().plot(); plt.legend(loc='best') 

plt.xlabel('Years', fontsize = 12)

plt.ylabel('Number of Houses', fontsize = 12)
# Variation in selling prices of houses by year of sale

ax = sns.boxplot(x = "YrSold", y = "SalePrice", data = df.loc[train_index,])

plt.show()
plt.figure(figsize=(14,5))



plt.subplot(1,2,1)

sns.distplot(df.loc[train_index,"SalePrice"], color = 'Sienna')

plt.xlabel('SalePrice', fontsize=12)



plt.subplot(1,2,2)

sns.distplot(np.log(1 + df.loc[train_index,"SalePrice"]), color = 'Sienna')

plt.xlabel('Log( SalePrice )', fontsize=12)



plt.show()
# Creation of new variables

## Age of housing (Year of sale - Year of construction)

df['AgeHouse'] = df['YrSold'] -  df['YearBuilt']



## Indicator if the dwelling has been renovated or not: 1 if YearBuild <> YearRemodAdd and O otherwise

df['RemodHouse'] = 1 * (df.YearBuilt != df.YearRemodAdd)



## Total living area of ​​housing (GrLivArea + TotalBsmtSF)

df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]



## Ratio of living space to property area

df["TotalArea"] = df["TotalArea"] / df["LotArea"]



## Ratio Between Number Bathrooms / Number of Bedrooms

df["BathRoomBedRoom"] = (df["FullBath"] + df["HalfBath"]) / (1 + df["BedroomAbvGr"])



df.drop("YearBuilt",axis = 1,inplace = True)

df.drop("YearRemodAdd",axis = 1,inplace = True)

df.drop("GarageYrBlt",axis = 1,inplace = True)
# Mean Absolute Persentage Error

def mape(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# Root Mean Squared Error

def rmse(y_pred, y_true):

    return np.sqrt(((y_pred - y_true) ** 2).mean())
## Target variable

y_train = df.SalePrice[train_index]

df.drop("SalePrice", axis = 1, inplace = True)



## Labelisation of categorical variables

## variables with scale of appreciation Ex, Gd, Ta, Fa, Po, Absent

var = ["FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond","ExterQual", "ExterCond",

       "HeatingQC", "PoolQC", "KitchenQual"]

df[var] = df[var].apply(lambda x: x.astype('category', ordered = True,

                                           categories= ["Absent", "Po", "Fa" , "Ta", "Gd","Ex"]).cat.codes, axis =1)



## Others variables

var = ["BsmtFinType1","BsmtFinType2"]

df[var] = df[var].apply(lambda x: x.astype('category', ordered = True,

                                           categories = ["Absent","Unf","LwQ","Rec","BLQ","ALQ","GLQ"]).cat.codes,axis =1)



df['GarageFinish'] = df['GarageFinish'].astype('category', ordered = True,

                                               categories = ["Absent", "Unf", "RFn" , "Fin"]).cat.codes



df['BsmtExposure'] = df['BsmtExposure'].astype('category', ordered = True,

                                               categories = ["Absent", "No", "Mn" , "Av","Gd"]).cat.codes



df['Fence'] = df['Fence'].astype('category', ordered = True,

                                 categories = ["Absent", "MnWw", "GdWo" , "MnPrv","GdPrv"]).cat.codes



df['LandSlope'] = df['LandSlope'].astype('category', ordered = True, categories= ["Gtl", "Mod", "Sev"]).cat.codes



df = df.replace({'C (all)':'Absent'})



df.drop('Condition2', axis = 1, inplace = True)



## Variables de type numérique et categoriel ordonnés recodés

var = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFullBath',

       'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'BsmtFinSF1',

       'GarageArea', 'GarageCars', 'GrLivArea','HalfBath', 'KitchenAbvGr', 'BsmtFinSF2',

       'LotArea', 'LotFrontage', 'LowQualFinSF','MasVnrArea', 'MiscVal',

       'OpenPorchSF', 'PoolArea', 'SalePrice', 'ScreenPorch', 'TotRmsAbvGrd',

       'TotalBsmtSF', 'WoodDeckSF','AgeHouse','RemodHouse','FireplaceQu','BsmtQual',

       'BsmtCond', 'GarageQual', 'GarageCond','ExterQual', 'ExterCond','HeatingQC',

       'PoolQC', 'KitchenQual','BsmtFinType1','BsmtFinType2','GarageFinish','OverallCond',

       'MSSubClass','OverallQual','TotalArea','BathRoomBedRoom','whisker']



## Transformation of non-numeric variables into an object

nonVar = list(set(df.columns).difference(set(var)))

df[nonVar] = df[nonVar].apply(lambda x: x.astype(str))



## We will transform all other variables into dummies

df = pd.get_dummies(df, columns = nonVar, sparse = True)

del var, nonVar
## Sample of learning, validatin sample (train, valid sets)

df_train = df.loc[train_index,]



df_train, df_valid, y_train, y_valid = train_test_split(df_train, y_train, test_size = 0.15)



## test Sample (test set)

df_test = df.loc[[i for i in df.index.tolist() if i not in train_index],]
# fixe random seed for reproducibility

seed = 2018

np.random.seed(seed)

n_folds = 5



## Root Mean Squared Error ross validation score

def rmse_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = seed).get_n_splits(df_train.values)

    rmse = np.sqrt(- cross_val_score(model, df_train.values, y_train, scoring = "neg_mean_squared_error", cv = kf))

    return(rmse)



## Mean Absolute Percentage Error cross validation score

def mape_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = seed).get_n_splits(df_train.values)

    MAPE = np.abs(cross_val_score(model, df_train.values, y_train,

                                    scoring = make_scorer(mape, greater_is_better = False), cv = kf))

                                    ## mape does not implement in for scoring, we construe it

    return(MAPE)
# Model 1: LinearRegression

regr = LinearRegression()

rmse_lm = rmse_cv(regr)

mape_lm = mape_cv(regr)



print('Linear Regression RMSE: {:.3f} ({:.3f})'.format(rmse_lm.mean(), rmse_lm.std()))

print('Linear Regression MAPE: {:.2f} ({:.2f})'.format(mape_lm.mean(), mape_lm.std()))
# Model 2: Ridge

params = {'alpha': [25,20, 15, 12, 10, 8, 6 ,5,4,3,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]}

 

rdge = GridSearchCV(Ridge(), params)



rmse_rdge = rmse_cv(rdge)

mape_rdge = mape_cv(rdge)



print('Ridge Regression RMSE: {:.3f} ({:.3f})'.format(rmse_rdge.mean(), rmse_rdge.std()))

print('Ridge Regression MAPE: {:.2f} ({:.2f})'.format(mape_rdge.mean(), mape_rdge.std()))
# Model 3: LassoCV



ls = LassoCV(eps = 0.001, n_alphas = 1000, alphas = None, fit_intercept=True, normalize=True,

      precompute= 'auto', max_iter = 1000, tol=0.0001, copy_X = True, cv = None,

      verbose = False, n_jobs=1, positive = False, random_state = None, selection = 'cyclic')



rmse_ls = rmse_cv(ls)

mape_ls = mape_cv(ls)



print('Lasso Regression RMSE: {:.3f} ({:.3f})'.format(rmse_ls.mean(), rmse_ls.std()))

print('Lasso Regression MAPE: {:.2f} ({:.2f})'.format(mape_ls.mean(), mape_ls.std()))
# Model 4: RandomForestRegressor

#params = {'max_depth':[None, 2 ,5, 10, 15, 20],

#          'min_samples_split' : [2, 5 ,10,20]

#         }

#rf = GridSearchCV(RandomForestRegressor(n_estimators = 500, criterion = 'mse', n_jobs = -1), params)



rf = RandomForestRegressor(n_estimators = 500, criterion = 'mse', max_depth = 10, min_samples_split = 10 ,

                              min_weight_fraction_leaf = 0.0, max_features='auto', max_leaf_nodes=None, bootstrap = True,

                              oob_score=False, n_jobs=-1,random_state= None, verbose = 0, warm_start = False)

# 27090.604 (2822.427)

# opt: Random Forest Regression score: 26557.765 (2547.867)

rmse_rf = rmse_cv(rf)

mape_rf = mape_cv(rf)



print('Random Forest Regression RMSE: {:.3f} ({:.3f})'.format(rmse_rf.mean(), rmse_rf.std()))

print('Random Forest Regression MAPE: {:.2f} ({:.2f})'.format(mape_rf.mean(), mape_rf.std()))
# Modèl 5 : GradientBoosting

#params = {'max_depth':[None, 2 ,5, 10],

#          'min_samples_split' : [2, 5 ,10,20]

#         }

#gb = GridSearchCV(GradientBoostingRegressor(n_estimators = 500, criterion = 'mse', 

#                                            max_features = 'sqrt' , loss='huber'), params)

gb = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.05, max_depth = 5, max_features='sqrt',

                               min_samples_split= 20,loss='huber', random_state = 5)

# score: 25009.752 (3103.486)

# Score: Optimized : 21857.153 (1404.104)

rmse_gb = rmse_cv(gb)

mape_gb = mape_cv(gb)

print('Gradient Boosting Regression RMSE: {:.3f} ({:.3f})'.format(rmse_gb.mean(), rmse_gb.std()))

print('Gradient Boosting Regression MAPE: {:.2f} ({:.2f})'.format(mape_gb.mean(), mape_gb.std()))
# Model 6 : Xgboosting

one_to_left = ss.beta(10, 1)

from_zero_positive = ss.expon(0, 50)



params = {"n_estimators": ss.randint(3, 40), # range(3, 40) proposition from a reader

          "max_depth": ss.randint(3, 40),

          "learning_rate": ss.uniform(0.05, 0.4),

          "colsample_bytree": one_to_left,

          "subsample": one_to_left,

          "gamma": ss.uniform(0, 10),

          'reg_alpha': from_zero_positive,

          "min_child_weight": from_zero_positive,

         }

xgbreg = XGBRegressor()

xgb = RandomizedSearchCV(xgbreg, params, n_jobs = 1)



rmse_xgb = rmse_cv(xgb)

mape_xgb = mape_cv(xgb)



print('XgBoosting Regression RMSE: {:.3f} ({:.3f})'.format(rmse_xgb.mean(), rmse_xgb.std()))

print('XgBoosting Regression MAPE: {:.2f} ({:.2f})'.format(mape_xgb.mean(), mape_xgb.std()))
# Modèl 7: ElasticNetCV with GridSearch

elastic = ElasticNetCV(alphas = None, copy_X = True, eps = 0.001, fit_intercept = True,

       l1_ratio = [0.01, 0.1, 0.3, 0.5, 0.7, 0.99], max_iter = 1000, n_alphas = 1000, n_jobs = 1,

       normalize = False, positive = False, precompute = 'auto', random_state = 0,

       selection = 'random', tol = 0.01, verbose = 0)



rmse_elastic = rmse_cv(elastic)

mape_elastic = mape_cv(elastic)



print('Elastic Regression RMSE: {:.3f} ({:.3f})'.format(rmse_elastic.mean(), rmse_elastic.std()))

print('Elastic Regression MAPE: {:.2f} ({:.2f})'.format(mape_elastic.mean(), mape_elastic.std()))
# Modèl 8 : Xgboost  with GridSearch

params = {'max_depth': [3]}

xgbs = GridSearchCV(XGBRegressor(n_estimators = 100), params, n_jobs = -1)



#Grid Xgboosting Regression RMSE: 24706.905 (2769.723)

#Grid Xgboosting Regression MAPE: 9.65 (0.65)



rmse_xgbs = rmse_cv(xgbs)

mape_xgbs = mape_cv(xgbs)



print('Grid Xgboosting Regression RMSE: {:.3f} ({:.3f})'.format(rmse_xgbs.mean(), rmse_xgbs.std()))

print('Grid Xgboosting Regression MAPE: {:.2f} ({:.2f})'.format(mape_xgbs.mean(), mape_xgbs.std()))
# RMSE et MAPE sur 5-folds cross validation

Estimateurs = ['RandomForest','XgBoostingRandomize', 'LinearModel','GradientBoosting','Lasso',

               'Ridge','ElasticNet', 'XgboostGrid']

RMSE = [rmse_rf,rmse_xgb, rmse_lm,rmse_gb,rmse_ls,rmse_rdge,rmse_elastic,rmse_xgbs]



RMSE, Var_RMSE = [round(Dec(x)) for x in list(map(np.mean, RMSE))], [round(Dec(x)) for x in list(map(np.std, RMSE))]



MAPE = [mape_rf,mape_xgb, mape_lm,mape_gb,mape_ls,mape_rdge,mape_elastic,mape_xgbs]

MAPE, Var_MAPE = [round(Dec(x),2) for x in list(map(np.mean, MAPE))], [round(Dec(x),2) for x in list(map(np.std, MAPE))]



t = pd.DataFrame({'Estimateurs':Estimateurs, 'RMSE':RMSE, 'RMSE_Var':Var_RMSE,

              'MAPE':MAPE, 'MAPE_Var':Var_MAPE}).sort_values('MAPE').reset_index()

t.drop("index",axis = 1, inplace = True)

t
## Result of predictions in the validation sample

results = pd.DataFrame({'y_valid':y_valid})



## Estimation and prediction of models

models = {'lm':regr, 'Ridge' : rdge,'Lasso' :ls, 'RF':rf, 'GB': gb,

          'XgbRand':xgb,'XgbGrid':xgbs, 'ElasticNet':elastic}



## Modeling and prediction

for key in models.keys():

    models[key].fit(df_train, y_train)

    results[key] = list(models[key].predict(df_valid))
results.iloc[:,2:8].corr().sort_values("GB", ascending = False).round(3)
corr = results.iloc[:,2:8].corr().sort_values("GB", ascending = True)

plt.figure(figsize = (7,5))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

     ax = sns.heatmap(corr, mask=mask,square=True)

plt.show()
results['Ensembling'] = 0.70*results['GB'] + 0.2*results['XgbRand'] + 0.1*results['Lasso']
# Plot feature importance

top = 15

feature_importance = models['GB'].feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())



sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize = (10,6))

plt.barh(pos[-top:], feature_importance[sorted_idx][-top:], align='center')

plt.yticks(pos[-top:], df.columns[sorted_idx][-top:])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
performance = [[rmse(results['y_valid'], results[x]), mape(results['y_valid'], results[x])] for x in results.columns[1:]]

performance = pd.DataFrame(performance, columns = ['RMSE','MAPE']).round(2)

performance['Estimateurs'] = list(results.columns[1:])

performance.sort_values('MAPE', ascending = True, inplace = True)

performance = performance[['Estimateurs','MAPE','RMSE']]

performance['RMSE'] = list(performance.RMSE.astype('int'))

performance