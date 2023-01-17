# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



data = pd.concat([train, test], ignore_index=True)
percent_null = data.isnull().sum()/len(data)*100

percent_null = percent_null[percent_null>0]

print(percent_null.sort_values())
num_data = data.select_dtypes(include = [np.number])

cat_data = data.select_dtypes(exclude = [np.number])
# replacing nulls

def null_cols(dataframe):

    for col in dataframe.columns:

        null_count = dataframe[col].isnull().sum()

        if null_count > 0:

            percent_null = null_count/len(dataframe[col])*100

            print(f"{col} percent null: {round(percent_null,3)}")
num_data.columns
null_cols(num_data) # note: sale price nulls come from the test set
data.LotFrontage.fillna(np.mean(data.LotFrontage), inplace=True)

data.MasVnrArea.fillna(np.mean(data.MasVnrArea), inplace=True)

data.BsmtFinSF1.fillna(np.mean(data.BsmtFinSF1), inplace=True)

data.BsmtFinSF2.fillna(np.mean(data.BsmtFinSF2), inplace=True)

data.BsmtUnfSF.fillna(np.mean(data.BsmtUnfSF), inplace=True)

data.TotalBsmtSF.fillna(np.mean(data.TotalBsmtSF), inplace=True)

data.BsmtFullBath.fillna(np.mean(data.BsmtFullBath), inplace=True)

data.BsmtHalfBath.fillna(np.mean(data.BsmtHalfBath), inplace=True)

data.GarageYrBlt.fillna(np.mean(data.GarageYrBlt), inplace=True)

data.GarageCars.fillna(np.mean(data.GarageCars), inplace=True)

data.GarageArea.fillna(np.mean(data.GarageArea), inplace=True)
# numerical columns that are categorical in nature: MSSubClass, MoSold, YrSold

# convert these to categorical columns with one hot encoding

encoder = OneHotEncoder()

#np.sort(data['MSSubClass'].unique())

temp = pd.DataFrame(encoder.fit_transform(data[['MSSubClass']]).toarray(), columns=['MS20','MS30','MS40','MS45','MS50','MS60','MS70','MS75','MS80','MS85','MS90','MS120','MS150','MS160','MS180','MS190'])

data = data.join(temp) # https://stackoverflow.com/questions/38256104/differences-between-merge-and-concat-in-pandas

data.drop('MSSubClass', 1, inplace=True)



temp = pd.DataFrame(encoder.fit_transform(data[['MoSold']]).toarray(), columns=['SoldJan','SoldFeb','SoldMar','SoldApr','SoldMay','SoldJun','SoldJul','SoldAug','SoldSep','SoldOct','SoldNov','SoldDec'])

data = data.join(temp)

data.drop('MoSold', 1, inplace=True)



#np.sort(data['YrSold'].unique())

temp = pd.DataFrame(encoder.fit_transform(data[['YrSold']]).toarray(), columns=['Sold2006', 'Sold2007', 'Sold2008', 'Sold2009', 'Sold2010'])

data = data.join(temp)

data.drop('YrSold', 1, inplace=True)
null_cols(cat_data)
cat_data.columns
# categorical to numerical (some are categorical variables that have to be encoded, others are categorical variables that are actually numerical in nature)

# either way, we should change all cat variables to numerical to observe their usefulness with a correlation matrix



#1. cat variables that are numerical in nature

data['LotShape'].replace({"IR3": 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, inplace=True)

data['Utilities'] = data['Utilities'].fillna('AllPub')

data['Utilities'].replace({'NoSeWa': 1, 'AllPub': 2}, inplace=True)

data['BldgType'].replace({"Twnhs": 1, 'TwnhsE': 2, 'Duplex': 3, '2fmCon': 4, '1Fam': 5}, inplace=True)

data['ExterQual'].replace({"Fa": 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

data['ExterCond'].replace({"Po": 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)

data['BsmtQual'] = data['BsmtQual'].fillna(0)

data['BsmtQual'].replace({'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['BsmtCond'] = data['BsmtCond'].fillna(0)

data['BsmtCond'].replace({'Po': 1,'Fa': 2, 'TA': 3, 'Gd': 4}, inplace=True)



data['BsmtExposure'] = data['BsmtExposure'].fillna(0)

data['BsmtExposure'].replace({"No": 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, inplace=True)



data['BsmtFinType1'] = data['BsmtFinType1'].fillna(0)

data['BsmtFinType1'].replace({"Unf": 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=True)



data['BsmtFinType2'] = data['BsmtFinType2'].fillna(0)

data['BsmtFinType2'].replace({"Unf": 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=True)

data['HeatingQC'].replace({"Po": 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['CentralAir'].replace({"Y": 1, 'N': 0}, inplace=True)

data['Electrical'] = data['Electrical'].fillna(0)

data['Electrical'].replace({"Mix": 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, inplace=True)



data['KitchenQual'] = data['KitchenQual'].fillna(0)

data['KitchenQual'].replace({'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['Functional'] = data['Functional'].fillna(7)

data['Functional'].replace({"Sev": 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}, inplace=True)



data['FireplaceQu'] = data['FireplaceQu'].fillna(0)

data['FireplaceQu'].replace({"Po": 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['GarageType'] = data['GarageType'].fillna(0)

data['GarageType'].replace({"Detchd": 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6}, inplace=True)



data['GarageFinish'] = data['GarageFinish'].fillna(0)

data['GarageFinish'].replace({"Unf": 1, 'RFn': 2, 'Fin': 3}, inplace=True)



data['GarageQual'] = data['GarageQual'].fillna('TA')

data['GarageQual'].replace({"Po": 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['GarageCond'] = data['GarageCond'].fillna('TA')

data['GarageCond'].replace({"Po": 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)



data['PavedDrive'].replace({"N": 1, 'P': 2, 'Y': 3}, inplace=True)



data['PoolQC'] = data['PoolQC'].fillna(0)

data['PoolQC'].replace({"Fa": 1, 'Gd': 2, 'Ex': 3}, inplace=True)



data['Fence'] = data['Fence'].fillna(0)

data['Fence'].replace({"MnWw": 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}, inplace=True)
#2. encoding cat variables

data['MSZoning'] = data['MSZoning'].fillna('RL')

temp = pd.DataFrame(encoder.fit_transform(data[['MSZoning']]).toarray(), columns=['RLZone', 'RMZone', 'CZone', 'FVZone', 'RHZone'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Street']]).toarray(), columns=['Pave','Grvl'])

data = data.join(temp)



data['Alley'] = data['Alley'].fillna('No alley access')

temp = pd.DataFrame(encoder.fit_transform(data[['Alley']]).toarray(), columns=['NoAlley','GrvlAlley','PvdAlley'])

data = data.join(temp) # https://stackoverflow.com/questions/38256104/differences-between-merge-and-concat-in-pandas



temp = pd.DataFrame(encoder.fit_transform(data[['LandContour']]).toarray(), columns=['Lvl','Bnk','Low','HLS'])

data = data.join(temp)

data['NotLvl'] = data.Bnk + data.Low + data.HLS

data.drop(['Bnk', 'Low', 'HLS'], 1, inplace=True)



temp = pd.DataFrame(encoder.fit_transform(data[['LotConfig']]).toarray(), columns=['Inside','FR2','Corner','CulDSac','FR3'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['LandSlope']]).toarray(), columns=['Gtl','Mod','Sev'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Neighborhood']]).toarray(), columns=data.Neighborhood.unique())

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Condition1']]).toarray(), columns=data.Condition1.unique())

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Condition2']]).toarray(), columns=['Norm2', 'Artery2', 'RRNn2', 'Feedr2', 'PosN2', 'PosA2', 'RRAn2', 'RRAe2'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['HouseStyle']]).toarray(), columns=['2Story','1Story','1.5Fin','1.5Unf','SFoyer','SLvl','2.5Unf','2.5Fin'])

data = data.join(temp)

data['Split'] = data.SLvl + data.SFoyer

data.drop(['SLvl', 'SFoyer'], 1, inplace=True)



temp = pd.DataFrame(encoder.fit_transform(data[['RoofStyle']]).toarray(), columns=['Gable','Hip','Gambrel','Mansard','Flat','Shed'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['RoofMatl']]).toarray(), columns=['CompShg','WdShngl','Metal','WdShake','Membran','Tar&Grv','Roll','ClyTile'])

data = data.join(temp)



data['Exterior1st'] = data['Exterior1st'].fillna('Other')

temp = pd.DataFrame(encoder.fit_transform(data[['Exterior1st']]).toarray(), columns=['VinylSd1','MetalSd1','Wd Sdng1','HdBoard1','BrkFace1','WdShing1','CemntBd1','Plywood1','AsbShng1','Stucco1','BrkComm1','AsphShn1','Stone1','ImStucc1','CBlock1','Other1'])

data = data.join(temp)



data['Exterior2nd'] = data['Exterior2nd'].fillna('Other')

temp = pd.DataFrame(encoder.fit_transform(data[['Exterior2nd']]).toarray(), columns=['VinylSd2','MetalSd2','Wd Shng2','HdBoard2','Plywood2','Wd Sdng2','CmentBd2','BrkFace2','Stucco2','AsbShng2', 'Brk Cmn2','ImStucc2','AsphShn2','Stone2','Other2','CBlock2'])



data['MasVnrType'] = data['MasVnrType'].fillna('None')

temp = pd.DataFrame(encoder.fit_transform(data[['MasVnrType']]).toarray(), columns=['BrkFace','NoMasVnr','StoneMasVnr','BrkCmn'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Foundation']]).toarray(), columns=['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'StoneFound'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['Heating']]).toarray(), columns=['GasA', 'GasW', 'GravF', 'WallF', 'OthW', 'FloorF'])

data = data.join(temp)



data['SaleType'] = data['SaleType'].fillna('WD')

temp = pd.DataFrame(encoder.fit_transform(data[['SaleType']]).toarray(), columns=['WDSale', 'NewSale', 'CODSale', 'ConLDSale', 'ConLISale', 'CWDSale', 'ConLwSale', 'ConSale', 'OthSale'])

data = data.join(temp)



temp = pd.DataFrame(encoder.fit_transform(data[['SaleCondition']]).toarray(), columns=['NormalCond', 'AbnormlCond', 'PartialCond', 'AdjLandCond', 'AllocaCond', 'FamilyCond'])

data = data.join(temp)



data.drop(['MiscFeature','MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',

       'Heating', 'SaleType', 'SaleCondition'], 1, inplace=True) # dropping categorical columns now that we've encoded them



# I was dumb and did not think of a faster way to do encoding. This is a better way to do it from https://www.kaggle.com/d0nghe/house-price-6

'''

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'HouseStyle',

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','Neighborhood', 'SaleCondition',

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','Condition1','Condition2',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','GarageType','SaleType','BldgType')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lb = LabelEncoder() 

    lb.fit(list(data[c].values)) 

    data[c] = lb.transform(list(data[c].values))

'''
data['TotalBaths'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']

data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

data['BsmtFinType'] = data['BsmtFinType1'] + data['BsmtFinType2']

data['BsmtFinSF'] = data['BsmtFinSF1'] + data['BsmtFinSF2']

data['GarageRating'] = data['GarageQual'] + data['GarageCond']

data['ExterRating'] = data['ExterQual'] + data['ExterCond']

data['OvrRating'] = data['OverallQual'] + data['OverallCond']



data.drop(['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','1stFlrSF','2ndFlrSF','TotalBsmtSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','GarageQual','GarageCond','ExterQual','ExterCond','OverallQual','OverallCond'], 1, inplace=True)
cat_data = data.select_dtypes(exclude = [np.number])

cat_data.columns
null_cols(data)
num_data = data.select_dtypes(include = [np.number])

num_data.columns
#data.drop(cat_data.columns, 1, inplace=True)
feature_cols = []

for col in data.columns:

    feature_cols.append(col)

    

feature_cols.remove('Id')
'''# Outlier detection 

from collections import Counter



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than n outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

outliers_to_drop = detect_outliers(data[:1460], 15, data.columns) #we can use data because all of it is numeric at this point



print(len(outliers_to_drop))

data = data.drop(outliers_to_drop, axis = 0).reset_index(drop=True)'''
test_index = data[data['SalePrice'].isnull()].index.tolist()

test_index
# sns.distplot(train['SalePrice'])

# SalePrice is not normally distributed; taking the log makes ML algos predict better

sns.distplot(np.log(train['SalePrice']))
train_data = data[:1460]

test_data = data[1460:].drop(['SalePrice'], 1)

x = train_data.drop(['SalePrice'], 1)

y = np.log1p(train_data['SalePrice'])

#y = train_data['SalePrice']
len(test_data)
'''from sklearn.preprocessing import  MinMaxScaler

sc = MinMaxScaler()

x = scaler.fit_transform(x)

test_data = scaler.fit_transform(test_data)'''
'''import keras



model = keras.models.Sequential()



model.add(keras.layers.Dense(16, activation='relu', input_dim = 227))

model.add(keras.layers.Dense(16, activation='relu'))

model.add(keras.layers.Dense(1))



model.compile(optimizer='adam', loss='mean_squared_error')



model.fit(x, y, batch_size=30, epochs=400, callbacks=[keras.callbacks.EarlyStopping(patience=2)])'''
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor # did not use the last two, might want to in future version

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV

from sklearn.kernel_ridge import KernelRidge # did not use, might want to in future version

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor # did not use, might want to in future version



from sklearn import model_selection

from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV # did not use gridsearchcv, might want to in future version



from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



from scipy.stats import uniform as sp_randFloat

from scipy.stats import randint as sp_randInt
# https://www.dezyre.com/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
'''kfolds = KFold(n_splits=5, shuffle=True, random_state=1)'''
'''GBR = GradientBoostingRegressor()

XGB = XGBRegressor(objective='reg:squarederror')

LGBM = LGBMRegressor()

RF = RandomForestRegressor()

alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

ridge = RidgeCV(alphas=alphas, cv=kfolds)

elas = ElasticNetCV(cv=kfolds, n_jobs=-1)'''
'''def cv_rmse(model, x=x):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kfolds))

    return rmse

    

models = [GBR, XGB, LGBM, RF, ridge, elas]

for model in models:

    print(f"{model} cv_rmse: {np.mean(cv_rmse(model))}")'''
'''ridge.fit(x,y)

ridge.score(x,y)'''
'''elas.fit(x,y)

elas.score(x,y)'''
'''parameters = {'learning_rate': sp_randFloat(0.01, 1.0),

'n_estimators' : sp_randInt(100, 5000),

}

GBR_RSCV = RandomizedSearchCV(estimator=GBR, param_distributions = parameters, n_iter = 10, n_jobs=-1, verbose=2)

GBR_RSCV.fit(x, y)



print(" Results from Random Search" )

print("\n The best estimator across ALL searched params:\n", GBR_RSCV.best_estimator_)

print("\n The best score across ALL searched params:\n", GBR_RSCV.best_score_)'''
'''params = {'nthread': [-1],

             'objective':['reg:squarederror'],

             'learning_rate': sp_randFloat(0.005, 1.0),

             'min_child_weight': sp_randInt(2, 5),

             'subsample': sp_randFloat(0.5, 1.5),

             'colsample_bytree': sp_randFloat(0.5, 1.5),

             'n_estimators': sp_randInt(100,5000)}

XGB_RSCV = RandomizedSearchCV(estimator=XGB, param_distributions = params, n_iter = 10, n_jobs=-1, verbose=2)

XGB_RSCV.fit(x, y)



print(" Results from Random Search" )

print("\n The best estimator across ALL searched params:\n", XGB_RSCV.best_estimator_)

print("\n The best score across ALL searched params:\n", XGB_RSCV.best_score_)'''
'''params = {'num_leaves': sp_randInt(1, 40),

          'learning_rate': sp_randFloat(0.005, 1.0),

          'n_estimators': sp_randInt(100, 8000),

         "min_child_weight": sp_randFloat(),

         'min_child_samples': sp_randInt(5, 35),

         'subsample_for_bin': sp_randInt(50000, 350000)}

LGBM_RSCV = RandomizedSearchCV(estimator=LGBM, param_distributions = params, n_iter = 10, n_jobs=-1, verbose=2)

LGBM_RSCV.fit(x, y)



print(" Results from Random Search" )

print("\n The best estimator across ALL searched params:\n", LGBM_RSCV.best_estimator_)

print("\n The best score across ALL searched params:\n", LGBM_RSCV.best_score_)'''
'''params = {'n_estimators': sp_randInt(100, 1000), 

            'max_features': ('auto', 'sqrt'), 

            'min_samples_split': sp_randInt(2,10), 

            'min_samples_leaf': sp_randInt(1, 5)}

RF_RSCV = RandomizedSearchCV(estimator=RF, param_distributions = params, n_iter = 10, n_jobs=-1, verbose=2)

RF_RSCV.fit(x, y)



print(" Results from Random Search" )

print("\n The best estimator across ALL searched params:\n", RF_RSCV.best_estimator_)

print("\n The best score across ALL searched params:\n", RF_RSCV.best_score_)'''
clf =  LGBMRegressor().fit(x,y)

                          

pred = np.expm1(clf.predict(test_data))

#pred = clf.predict(test_data)



pred = pd.DataFrame({"id": test.Id, "SalePrice": pred})

pred.to_csv('sample_submission.csv',index=False)