import numpy as np

import pandas as pd 

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ShuffleSplit

from scipy.stats import kde, boxcox, pearsonr

from scipy.special import inv_boxcox

import seaborn as sns

from sklearn.metrics import make_scorer, confusion_matrix, roc_curve, auc, r2_score, classification_report, mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn import linear_model, svm, tree, neighbors

from collections import defaultdict

from itertools import chain, combinations

from scipy.stats.stats import pearsonr

from sklearn.decomposition import PCA

import math 

from yellowbrick.features import RFECV

from scipy.stats import normaltest

from scipy.special import boxcox1p

from scipy.stats import boxcox, boxcox_normmax

pd.options.mode.chained_assignment = None

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import sys

from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint as sp_randint

import xgboost as xgb

import lightgbm as lgb

from itertools import chain, combinations

import itertools



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams.update({'figure.max_open_warning': 0})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os 

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





# import all our functions

os.chdir("../input/myscripts2/")

from scripts import *

os.chdir("../../working/")



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

class Model:

    def __init__(self, model, name, cutoff = 0.001):

        self.name = name

        self.model = model

        self.cutoff = cutoff

        self.scores_summary = []

    #

    # Set the model parameters based on passed in dict.

    #

    def set_parameters(self, params):

        self.model.set_params(**params)

    #

    # This is used during inital feature selection. For all features the

    # error is calculated and then each feature is iteratively randomized

    # and the increase in error is calculated. Features with high scores

    # indicate features that are important for the model.

    #

    def feature_select_mda(self, error_test, X_train, Y_train):

        scores = defaultdict(list)

        for j in range(100):

            X_train_cv,X_test_cv,Y_train_cv,Y_test_cv = train_test_split(X_train,Y_train)

            self.model.fit(X_train_cv, np.ravel(Y_train_cv) )

            Y_pred = self.model.predict(X_test_cv)

            #conf=confusion_matrix(Y_test_cv,model.predict(X_test_cv))

            error_total = np.sqrt(error_test(np.ravel(Y_test_cv), Y_pred))

            for i in range(X_train.shape[1]):

                X_t = X_test_cv.copy().reset_index(drop=True)

                X_t.iloc[:,i] = X_t.iloc[:,i].sample(frac=1).reset_index(drop=True)

                #suff_matrix=confusion_matrix(Y_test_cv,model.predict(X_t))

                Y_pred_i = self.model.predict(X_t)

                error_i = np.sqrt(error_test(np.ravel(Y_test_cv), Y_pred_i))

                #shuff_accuracy=(suff_matrix[0,0]+suff_matrix[1,1])/(suff_matrix[0,0]+suff_matrix[1,1]+suff_matrix[0,1]+suff_matrix[1,0])

                scores[X_t.columns[i]].append((error_i-error_total)/error_total)

        #print("Feature importance:")

        #for mscore in scores:

        #    print( "{}: {:0.4f}".format(mscore,np.mean(scores[mscore]) ) )

        self.scores_summary = []

        for mscore in scores:

            self.scores_summary.append(np.mean(scores[mscore]))

    #

    # 

    #

    def model_performance(self, X_train, Y_train, model_results):

        model_performance = np.sqrt(-cross_val_score(self.model, X_train, Y_train, scoring = "neg_mean_squared_error", cv = 10))

        if model[0] not in model_results:

            model_results[model[0]] = [model, model_performance]

        else:

            if np.mean(model_results[model[0]][1]) < np.mean(model_performance):

                model_results[model[0]][1] = model_performance

        return model_results

    #

    # Return a dataframe containing features that are above a certain cutoff

    # based on the feature_select_mda function. If no cutoff is given then

    # the cutoff saved for the Model object will be used. 

    #

    def get_important_features(self, df, cutoff = None):

        if cutoff is None:

            cutoff = self.cutoff

        filter = pd.DataFrame(self.scores_summary) > cutoff

        return df[df.columns[np.ravel(filter)]]



# Try various transformations of the data and keep the one most correlated to Y

# If it is normal then standardize or else scale to [0,1]

def autoTransformRegression(df, Y):

    best_transforms = []

    for column in df:

        #print(column)

        df[column] = df[column].astype(float)

        try:

            best_R = pearsonr(df[column],np.ravel(Y) )[0]

        except:

            best_R = 0

        best_transform = lambda x: x

        best_type = "not"

        

        #with np.errstate(all='raise'):

        #    try: 

        #        best_lambda = boxcox_normmax( df[column] + 1, method = 'pearsonr')

        #    except:

        #        best_lambda = 0.15

        #if best_lambda < 1e-5:

        #    best_lambda = 0.15

        transforms = {

            0 : (lambda x: boxcox1p(x, 0.15),'boxcox'), 

             1 : (np.log1p,'log'), 

             2 : (np.sqrt,'sqrt'), 

             3 : (lambda x: 1/(x+1),'inv'), 

             4 : (lambda x: np.sqrt(np.sqrt(x)),'qtrt'), 

             5 : (np.cbrt,'cbrt'),

             6 : (lambda x: np.power(x,2),'sq'),

             7 : (lambda x: np.power(x,3),'cb'),

             8 : (lambda x: np.power(x,4),'fth')}

        

        for i in range(6):

            try:

                test_R = pearsonr(transforms[i][0](df[column]),np.ravel(Y) )[0]

            except:

                test_R = 0

            if test_R > best_R:

                best_type = transforms[i][1]

                best_transform = transforms[i][0]

                best_R = test_R

        if normaltest(best_transform(df[column]))[1] > 0.01/df.shape[1]:

            scaler = StandardScaler().fit(best_transform(df[[column]]))

            #rescaledX = scaler.transform(best_transform)

            method = 'standardized'

        else:

            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(best_transform(df[[column]]))

            #rescaledX = scaler.fit_transform(best_transform)

            method = 'scaled'

        #df.loc[:,column] = rescaledX

        best_transforms.append([best_transform, scaler, best_type, method])

        

        #print('{} was {} transformed and {}.'.format(column,  best_type, method))

    return best_transforms



train_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# Let's have a look at the data

train_set.dropna(subset = ['Id'], inplace = True, axis = 0)

train_set.drop(columns = ['Id'], inplace = True)

train_set.head()



get_NAs(train_set)
cols=['PoolQC','Fence','Alley','MiscFeature','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageFinish','GarageType','GarageQual','GarageCond']



for col in cols:

    train_set[col].fillna("None", inplace=True)





train_set.loc[:,'LotFrontageSq'] = train_set.LotFrontage**2

fig, ax = plt.subplots(figsize=(8, 4))

ax.set_yscale('log')

ax.set_xscale('log')

plt.title('Before imputation')

sns.scatterplot(y='LotFrontageSq',x='LotArea',hue='LotConfig',data=train_set)



imputed = train_set['LotFrontageSq'].isnull()



X_model = train_set.loc[train_set['LotFrontageSq'].notnull(),['LotArea','LotConfig']]#.apply(np.log)

X2 = pd.get_dummies(X_model[['LotConfig']])

X_model = pd.concat([X_model,X2],axis=1).drop(columns=['LotConfig'])

X_model2 = [ [np.log(y[0]),y[1],y[2],y[3],y[4]] for x,y in X_model.iterrows() ]

Y_model = train_set.loc[train_set['LotFrontageSq'].notnull(),'LotFrontageSq'].apply(np.log)

X_impute = train_set.loc[train_set['LotFrontageSq'].isnull(),['LotArea','LotConfig']]

X2 = pd.get_dummies(X_impute[['LotConfig']])

X_impute = pd.concat([X_impute,X2],axis=1).drop(columns=['LotConfig'])

X_impute2 = [ [np.log(y[0]),y[1],y[2],y[3],y[4]] for x,y in X_impute.iterrows() ]

Y_impute = np.exp(linearInterpolation(X_model2, Y_model, X_impute2))



fig, ax = plt.subplots(figsize=(8, 4))

sns.scatterplot(y=Y_impute,x=train_set[imputed].LotArea,hue=train_set[imputed].LotConfig)

ax.set_yscale('log')

ax.set_xscale('log')

plt.ylabel('PltFrontage^2')

plt.title('After imputation')

plt.show()



train_set.loc[train_set['LotFrontageSq'].isnull(),'LotFrontageSq'] = Y_impute

train_set.loc[train_set['LotFrontage'].isnull(),'LotFrontage'] = train_set.loc[train_set['LotFrontage'].isnull(),'LotFrontageSq'].apply(math.sqrt)



print(train_set.MasVnrArea.describe())



train_set.loc[train_set.MasVnrArea.isnull(),'MasVnrArea'] = 0

train_set.loc[train_set.MasVnrType.isnull(),'MasVnrType'] = 'None'



fig, ax = plt.subplots(figsize=(8, 4))

sns.countplot(train_set.Electrical)

fig, ax = plt.subplots(figsize=(8, 4))

sns.boxplot(x=train_set.Electrical,y=train_set.YearBuilt)



train_set.loc[train_set.Electrical.isnull() & (train_set.YearBuilt>=1960),'Electrical'] = 'SBrkr'

train_set.loc[train_set.Electrical.isnull() & (train_set.YearBuilt<1960),'Electrical'] = 'FuseA'





fig, ax = plt.subplots(figsize=(8, 4))

sns.distplot(train_set.GarageYrBlt.dropna())

fig, ax = plt.subplots(figsize=(8, 4))

ax.set_yscale('log')

sns.scatterplot(x=train_set.GarageYrBlt,y=train_set.GarageArea)



#pearsonr(data_train[data_train.GarageYrBlt.notnull()].GarageYrBlt,data_train[data_train.GarageYrBlt.notnull()].GarageArea.apply(np.log))



train_set.drop(columns=['GarageYrBlt'],inplace=True)
train_set.describe()
fig, ax= plt.subplots(figsize=(30, 30))

cmap = sns.diverging_palette( 10, 220, sep=20, n=6 )

train_set.loc[:,'NegPrice'] = -train_set.SalePrice

sns.heatmap(train_set.corr(), annot=True)

train_set.drop(columns=['NegPrice'],inplace = True)
def handle_NAs(df):

    cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','PoolQC','Fence','Alley','MiscFeature','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageFinish','GarageType','GarageQual','GarageCond']

    

    for col in cols:

        df[col].fillna("None", inplace=True)

    

    cols = ['BsmtHalfBath','BsmtFullBath','BsmtFinSF2','BsmtFinSF1','TotalBsmtSF','BsmtUnfSF','MasVnrArea','GarageArea','GarageCars']

    for col in cols:

        df.loc[df[col].isnull(),col] = 0

    df.loc[:,'MSSubClass'] = df[['MSSubClass']].astype('str')

    

    df.loc[df.KitchenQual.isnull(),'KitchenQual'] = 'TA'

    df.loc[df.SaleType.isnull(),'SaleType'] = 'WD'

    df.loc[df.MasVnrType.isnull(),'MasVnrType'] = 'None'

    df.loc[df.MSZoning.isnull(),'MSZoning'] = 'RL'

    df.loc[df.Utilities.isnull(),'Utilities'] = 'AllPub'

    df.loc[df.Exterior1st.isnull(),'Exterior1st'] = 'VinylSd'

    #df.loc[df.Exterior2nd.isnull(),'Exterior2nd'] = 'VinylSD'

    

    df.loc[df.Electrical.isnull() & (df.YearBuilt>=1960),'Electrical'] = 'SBrkr'

    df.loc[df.Electrical.isnull() & (df.YearBuilt<1960),'Electrical'] = 'FuseA'

    df.loc[df.Electrical.isnull(),'Electrical'] = df.loc[df.Electrical.notnull(),'Electrical'].mode()

    

    df.drop(columns=['GarageYrBlt'],inplace=True)

    

    #df.loc[:,'LotFrontageSq'] = df.LotFrontage**2

    #imputed = df['LotFrontageSq'].isnull()

    return df
def my_features(df):

    df.loc[:,'LotAreaSq'] = (df.LotArea**2 )

    

    df.loc[:,'OverallQualCond'] = (df.OverallQual + df.OverallCond)

    df.loc[:,'OveralQuallCondMult'] = (df.OverallQual * df.OverallCond)

    

    df.loc[:,'Remodeled'] = (df.YearBuilt == df.YearRemodAdd).astype(int)

    df.loc[:,'RemodeledTimeGap'] = (df.YearRemodAdd - df.YearBuilt)

    df.loc[:,'Age'] = (df.YrSold - df.YearBuilt)

    df.loc[:,'AgeRemodeled'] = (df.YrSold - df.YearRemodAdd)

    df.loc[df.Age < 0,'Age'] = 0

    df.loc[df.AgeRemodeled < 0,'AgeRemodeled'] = 0

    df.loc[df.RemodeledTimeGap < 0,'RemodeledTimeGap'] = 0

    

    for column in ['PoolQC','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual','GarageCond','GarageFinish','GarageQual','FireplaceQu']:

        df.replace({column: {"None": 0, "No": 0, "Unf": 1,"Po": 1, "RFn": 3, "Fa": 3, "Mn": 3, "Fin": 5, "Av": 5, "TA": 5, "Gd": 7,"Ex": 9}}, inplace=True)

    for column in ['BsmtFinType1','BsmtFinType2','Fence']:

        df.replace({column: {"None": 0, "Unf": 1,  "MnPrv": 3, "MnWw": 3,"LwQ": 3, "BLQ": 3,"ALQ": 5, "Rec": 5, "GdWo": 7,"GdPrv": 7, 'GLQ': 7}}, inplace=True)

    

    df.loc[:,'ExterQualCond'] = (df.ExterQual + df.ExterCond)

    df.loc[:,'ExterQualCondMult'] = (df.ExterQual * df.ExterCond)

    df.loc[:,'BsmtQualCond'] = (df.BsmtQual + df.BsmtCond + df.BsmtFinType1 + df.BsmtFinType2)

    df.loc[:,'BsmtQualCondMult'] = df.BsmtQual * df.BsmtCond  * ( df.BsmtFinType1 + df.BsmtFinType2)

    df.loc[:,'BsmtQualCondSize'] = (df.BsmtFinType1 * df.BsmtFinSF1) + (df.BsmtFinType2 * df.BsmtFinSF2)

    df.loc[:,'KitchenQuCount'] = (df.KitchenQual * df.KitchenAbvGr)

    df.loc[:,'BsmtBathTotal'] = (df.BsmtFullBath + df.BsmtHalfBath)

    df.loc[:,'BathTotal'] = (df.FullBath + df.HalfBath)

    df.drop(columns=["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],inplace=True) # "BedroomAbvGr","KitchenAbvGr","Fireplaces"

    df.loc[:,'FireplaceQuCount'] = (df.FireplaceQu * df.Fireplaces)

    df.loc[:,'GarageQualCond'] = (df.GarageQual + df.GarageCond + df.GarageFinish)

    df.loc[:,'GarageQualCondMult'] = (df.GarageQual * df.GarageCond * df.GarageFinish)

    df.loc[:,'GarageQualCondArea'] = (df.GarageQualCondMult * df.GarageArea)

    df.loc[:,'GarageQualCondCars'] = (df.GarageQualCondMult * df.GarageCars)

    df.loc[:,'PoolQualSize'] = (df.PoolArea * df.PoolQC)

    

    df.drop(columns=['Condition2','RoofMatl','Exterior2nd','Functional','MiscFeature'],inplace=True)

    

    return df

    
def interpolate_NA(X_train,X_test):

    X_train.loc[:,'LotFrontageSq'] = X_train.LotFrontage**2

    X_model = X_train.loc[X_train['LotFrontageSq'].notnull(),['LotArea']].apply(np.log)

    X_model2 = X_train.loc[X_train['LotFrontageSq'].notnull(),['LotConfig']]

    X_model = pd.concat([X_model,pd.get_dummies(X_model2[['LotConfig']])],axis=1)

    Y_model = X_train.loc[X_train['LotFrontageSq'].notnull(),'LotFrontageSq'].apply(np.log)

    X_impute = X_train.loc[X_train['LotFrontageSq'].isnull(),['LotArea']].apply(np.log)

    X_impute2 = X_train.loc[X_train['LotFrontageSq'].isnull(),['LotConfig']]

    X_impute = pd.concat([X_impute,pd.get_dummies(X_impute2[['LotConfig']])],axis=1)

    X_model,X_impute = fill_dummy_vars(X_model,X_impute)

    imputation = linear_model.LinearRegression()

    imputation.fit(X_model, Y_model)

    Y_impute =imputation.predict(X_impute)

    X_train.loc[X_train['LotFrontageSq'].isnull(),['LotFrontageSq']] = Y_impute

    X_train.loc[X_train['LotFrontage'].isnull(),'LotFrontage'] = X_train.loc[X_train['LotFrontage'].isnull(),'LotFrontageSq'].apply(math.sqrt)

    

    X_test.loc[:,'LotFrontageSq'] = X_test.LotFrontage**2

    X_impute = X_test.loc[X_test['LotFrontageSq'].isnull(),['LotArea']].apply(np.log)

    X_impute2 = X_test.loc[X_test['LotFrontageSq'].isnull(),['LotConfig']]

    X_impute = pd.concat([X_impute,pd.get_dummies(X_impute2[['LotConfig']])],axis=1)

    X_model,X_impute = fill_dummy_vars(X_model,X_impute)

    Y_impute = imputation.predict(X_impute)

    X_test.loc[X_test['LotFrontageSq'].isnull(),['LotFrontageSq']] = Y_impute

    X_test.loc[X_test['LotFrontage'].isnull(),'LotFrontage'] = X_test.loc[X_test['LotFrontage'].isnull(),'LotFrontageSq'].apply(math.sqrt)

    

    return X_train,X_test

train_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



seed = 7



# Split data for training and testing

data_train, data_test = train_test_split(train_set, random_state = seed)

data_train.dropna(subset = ['Id'], inplace = True, axis = 0)



X_train = data_train.copy()

X_test = data_test.copy()

X_train,X_test = interpolate_NA(X_train,X_test)



X_train = handle_NAs(X_train)

X_train = my_features(X_train)

categoricalDict = categorical_to_numericDict(X_train, 'SalePrice')

Y_train = X_train[['SalePrice']]

X_train.drop(columns=['SalePrice','Id'],inplace=True)

X_train.replace(categoricalDict, inplace=True)



Y_test = X_test[['SalePrice']]

X_test.drop(columns=['SalePrice','Id'],inplace=True)

X_test = handle_NAs(X_test)

X_test = my_features(X_test)

X_test.replace(categoricalDict, inplace=True)



##X_train,X_test = fill_dummy_vars(X_train,X_test)

temp = Y_train.copy()

print(autoTransform(temp))



Y_train['SalePrice'] = Y_train['SalePrice'].apply(np.log)



print("Normality test p-value  = ", normaltest(np.log(Y_train))[1], " with a cutoff of ", 0.01/X_train.shape[1])

#sns.distplot(Y_train)



scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Y_train)

Y_train = pd.DataFrame(scaler.transform(Y_train))



##X_train = categorical_to_dummy(X_train)

X_train = X_train.astype(float)

transformer = autoTransformRegression(X_train,Y_train)

X_train = make_transformations(X_train, transformer)



##X_test = categorical_to_dummy(X_test)

X_test = X_test.astype(float)

X_test = make_transformations(X_test, transformer)

pca_transform = PCA().fit(X_train)

pca_train = pd.DataFrame(pca_transform.transform(X_train))
scorer = make_scorer(mean_squared_error, greater_is_better=False)



models = [

    linear_model.LinearRegression(),

    linear_model.Ridge( random_state = seed),

    linear_model.Lasso( random_state  = seed),

    linear_model.BayesianRidge(),

    linear_model.ElasticNet( selection = 'random', random_state  = seed),

    GradientBoostingRegressor(loss = 'huber', random_state  = seed),

    xgb.XGBRegressor(),

    lgb.LGBMRegressor(random_state = seed),

    svm.SVR(gamma = 'scale'),

    tree.DecisionTreeRegressor(),

    MLPRegressor(max_iter=1000, random_state  = seed),

    neighbors.KNeighborsRegressor()

]

params = [

    dict(),

    dict(alpha = [1e-6,1e-4,1e-2,1e-1,1,10]),

    dict(alpha = [1e-6,1e-4,1e-2,1e-1,1,10] ),

    dict(),

    dict(alpha = [1e-6,1e-4,1e-2,1e-1,1,10]),

    dict( learning_rate = [1e-6,1e-4,1e-2,1e-1,1,10]),

    dict(),

    dict(),

    dict(),

    dict( min_samples_split = [2,5,10,20]),

    dict(),

    dict( n_neighbors = [3,5,10,20]),

]

names = ['OLS','Ridge','Lasso','BR','Net','GBR','XGB','LGB','SVM','Tree','MLPR','KNN']



my_models = []



warnings.filterwarnings("error")

for model,param,name in zip(models,params,names):

    my_model = Model(model, name)

    my_model.feature_select_mda(mean_squared_error, pca_train, Y_train)

    best_score = float("inf")

    best_cutoff = 0

    best_params = {}

    for cutoff in -10, 0, 1e-15,1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1:

        pca_subset = my_model.get_important_features(pca_train, cutoff)

        try: 

            model_score = -cross_val_score(my_model.model, pca_subset, np.ravel(Y_train), scoring=scorer, cv = 10)

        except:

            model_score = pd.Series(float("inf"))

        if model_score.mean() < best_score:

            best_score = model_score.mean()

            best_cutoff = cutoff

        for key, param_list in param.items():

            for param_value in param_list: 

                my_model.set_parameters({key: param_value})

                try: 

                    model_score = -cross_val_score(my_model.model, pca_subset, np.ravel(Y_train), scoring=scorer, cv = 10)

                except:

                    model_score = pd.Series(float("inf"))

                if model_score.mean() < best_score:

                    best_score = model_score.mean()

                    best_cutoff = cutoff

                    best_params = {key: param_value}

    my_model.cutoff = best_cutoff

    if best_params:

        my_model.set_parameters(best_params)

    my_models.append(my_model)

    print("Model {},    parameter {},    PCA cutoff {}:   CV MSE {:.6f}".format(name, best_params.items(), my_model.cutoff, best_score))

models_to_keep = [1,1,0,1,0,1,1,0,1,0,0,0]

my_models = list(itertools.compress(my_models, models_to_keep))
params = [

    dict(),

    dict(solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], alpha = [1e-10,1e-7,1e-5,1e-3,1e-1,1,5,10]),

    dict(),

    dict( learning_rate = [1e-7,1e-5,1e-3,1e-1,1], n_estimators = [500], max_depth = [2,3]),

    ##dict(n_iter = [100], alpha_1 = [1e-50,1e-10,1e-1,1,100,10000], alpha_2 = [1e-50,1e-10,1e-1,1,100,10000], lambda_1 = [1e-50,1e-10,1e-1,1,100,10000], lambda_2 = [1e-50,1e-10,1e-1,1,100,10000]),

    ##dict(alpha = [1e-6,1e-5,1e-4,1e-3,1e-2, 0.1], l1_ratio = [0.2,0.4,0.6, 0.8, 1], selection = ['random']),

    dict( eta = [1e-2, 1e-3, 1e-5, 1e-4, 0.1,1], gamma = [0, 1e-5,1e-4,0.001, 0.01, 0.1], alpha = [0, 1e-10, 1e-9, 1e-8, 1e-7]),

    #dict( learning_rate = [ 1e-2, 1e-3,1e-1], n_estimators = [500], reg_alpha = [1e-2, 1e-3,1e-4], reg_lambda = [1e-5, 1e-7, 1e-9,1e-15,0]),

    dict( gamma = [1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1], C = [0.01, 0.1, 1, 10, 50, 100, 250, 500, 750, 1000, 2000, 5000, 10000] )

    ##dict( alpha = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1], learning_rate = ['constant', 'invscaling', 'adaptive'], learning_rate_init = [1e-5,1e-4,1e-3,1e-2,1e-1] )

]



warnings.simplefilter(action='ignore', category=FutureWarning)

scorer = make_scorer(mean_squared_error, greater_is_better=False)

for i,my_model in enumerate(my_models):

    if params[i]:

        pca_subset = my_model.get_important_features(pca_train)

        grid = GridSearchCV( estimator = my_model.model, 

                param_grid = params[i], cv = 10, iid = False, scoring = scorer)

        grid.fit(pca_subset, np.ravel(Y_train) )

        print(my_model.name)

        print(grid.best_params_)

        my_model.set_parameters(grid.best_params_)
for my_model in my_models:

    pca_train_subset = my_model.get_important_features(pca_train)

    my_model.model.fit(pca_train_subset,np.ravel(Y_train))

    score = -cross_val_score(my_model.model, pca_subset, np.ravel(Y_train), scoring=scorer, cv = 10)

    print("Model {}:   Mean score {:.6f},   Score SD {:.4f}".format(my_model.name, score.mean(), score.std()))

# Handle cases where dummy variables are missing in one of the data sets

X_train,X_test = fill_dummy_vars(X_train,X_test)
pca_test = pd.DataFrame(pca_transform.transform(X_test))
Y_test['SalePrice'] = Y_test['SalePrice'].apply(np.log)

Y_test = pd.DataFrame(scaler.transform(Y_test))
Y_preds = pd.DataFrame() 

Y_preds = Y_preds.append(Y_test.copy(), ignore_index = True)

Y_preds.columns = ["SalePrice"]



for my_model in my_models:

    pca_test_subset = my_model.get_important_features(pca_test)

    Y_preds.loc[:,my_model.name] = pd.Series(my_model.model.predict(pca_test_subset))

    score = mean_squared_error( np.ravel(Y_test), np.ravel(Y_preds.loc[:,my_model.name]) )

    print("Model {}:  MSE {:.6f}".format(my_model.name, score))

train_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_set = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



seed = 7



X_train = train_set.copy()

X_test = test_set.copy()
X_train,X_test = interpolate_NA(X_train,X_test)



X_train = handle_NAs(X_train)

X_train = my_features(X_train)

categoricalDict = categorical_to_numericDict(X_train, 'SalePrice')

Y_train = X_train[['SalePrice']]

X_train.drop(columns=['SalePrice','Id'],inplace=True)

X_train.replace(categoricalDict, inplace=True)



X_test = handle_NAs(X_test)

X_test = my_features(X_test)

X_test.replace(categoricalDict, inplace=True)



X_train,X_test = fill_dummy_vars(X_train,X_test)



Y_train['SalePrice'] = Y_train['SalePrice'].apply(np.log)

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Y_train)

Y_train = pd.DataFrame(scaler.transform(Y_train))



##X_train = categorical_to_dummy(X_train)

X_train = X_train.astype(float)

transformer = autoTransformRegression(X_train, Y_train)

X_train = make_transformations(X_train, transformer)



##X_test = categorical_to_dummy(X_test)

X_test = X_test.astype(float)

X_test = make_transformations(X_test, transformer)

pca_transform = PCA().fit(X_train)

pca_train = pd.DataFrame(pca_transform.transform(X_train))

pca_test = pd.DataFrame(pca_transform.transform(X_test))
Y_preds = pd.DataFrame() 



for my_model in my_models:

    my_model.feature_select_mda(mean_squared_error, pca_train, Y_train)

    pca_train_subset = my_model.get_important_features(pca_train)

    pca_test_subset = my_model.get_important_features(pca_test)

    my_model.model.fit(pca_train_subset, np.ravel(Y_train))

    Y_preds.loc[:,my_model.name] = pd.Series(my_model.model.predict(pca_test_subset))
Y_preds2 = pd.DataFrame(np.exp(scaler.inverse_transform(Y_preds)))

Y_preds2.columns = Y_preds.columns.values