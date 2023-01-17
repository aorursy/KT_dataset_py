import numpy as np

import pandas as pd

df= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# Missing Data Pattern in Training Data

import seaborn as sns

sns.heatmap(df.isnull(), cbar=False, cmap='PuBu')
# Detailed Information on Missing Data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing.head(20)
# Drop the fields having > 40% missing values to avoid bias in the model

df.drop(['Alley'],axis=1,inplace=True)

df.drop(['PoolQC'],axis=1,inplace=True)

df.drop(['MiscFeature'],axis=1,inplace=True)

df.drop(['Fence'],axis=1,inplace=True)

df.drop(['FireplaceQu'],axis=1,inplace=True)



# Impute continuous var with Mean; Impute categorical var with Mode

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
# Convert categorical features into continuous features in Train Data

from sklearn.preprocessing import LabelEncoder

lencoders = {}

for col in df.select_dtypes(include=['object']).columns:

    lencoders[col] = LabelEncoder()

    df[col] = lencoders[col].fit_transform(df[col])
# Feature Selection using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel



x_t = df.drop('SalePrice', axis=1)

y_t = df['SalePrice']

clf_1 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='log2', max_depth = 4))

clf_2 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='auto', max_depth = 4))

clf_1.fit(x_t, y_t)

clf_2.fit(x_t, y_t)

sel_feat_1 = x_t.columns[(clf_1.get_support())]

sel_feat_2 = x_t.columns[(clf_2.get_support())]

print(sel_feat_1)

print(sel_feat_2)
# Feature Selection using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectFromModel



x_t = df.drop('SalePrice', axis=1)

y_t = df['SalePrice']

clf_3 = SelectFromModel(DecisionTreeClassifier(max_features='log2'))

clf_4 = SelectFromModel(DecisionTreeClassifier(max_features='auto'))

clf_3.fit(x_t, y_t)

clf_4.fit(x_t, y_t)

sel_feat_3 = x_t.columns[(clf_3.get_support())]

sel_feat_4 = x_t.columns[(clf_4.get_support())]

print(sel_feat_3)

print(sel_feat_4)
# Feature Selection using Extra Trees Classifier 

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



x_t = df.drop('SalePrice', axis=1)

y_t = df['SalePrice']

clf_5 = SelectFromModel(ExtraTreesClassifier(max_features='log2'))

clf_6 = SelectFromModel(ExtraTreesClassifier(max_features='auto'))

clf_5.fit(x_t, y_t)

clf_6.fit(x_t, y_t)

sel_feat_5 = x_t.columns[(clf_5.get_support())]

sel_feat_6 = x_t.columns[(clf_5.get_support())]

print(sel_feat_5)

print(sel_feat_6)
# Visualize Relative Feature Importance

from yellowbrick.features import FeatureImportances



clf_1 = RandomForestClassifier()

x_train_1 = df[['LotFrontage', 'LotArea', 'Neighborhood', 'Condition1','Condition2',

                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

                'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual', 'BsmtFinSF1','BsmtFinSF2',

                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

                'FullBath', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',

                'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']] 

y_train = df['SalePrice']

viz_1 = FeatureImportances(clf_1)

viz_1.fit(x_train_1, y_train)

viz_1 = FeatureImportances(clf_1)

viz_1.fit(x_train_1, y_train)

viz_1.poof()



clf_2 = DecisionTreeClassifier()

x_train_2 = df[['LotFrontage', 'LotArea', 'LotShape', 'LotConfig', 'Neighborhood',

                'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

                'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',

                '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr','TotRmsAbvGrd',

                'Fireplaces','GarageType','GarageYrBlt', 'GarageFinish', 'GarageArea', 'WoodDeckSF',

                'OpenPorchSF', 'MoSold', 'YrSold']]

viz_2 = FeatureImportances(clf_2)

viz_2.fit(x_train_2, y_train)

viz_2 = FeatureImportances(clf_2)

viz_2.fit(x_train_2, y_train)

viz_2.poof()



clf_3 = ExtraTreesClassifier()

x_train_3 = df[['MSSubClass', 'LotFrontage', 'LotArea', 'LotShape', 'LotConfig',

       'Neighborhood', 'OverallQual', 'OverallCond', 'YearBuilt',

       'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF',

       'TotalBsmtSF', 'HeatingQC', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'BsmtFullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageFinish', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'MoSold', 'YrSold']]

viz_3 = FeatureImportances(clf_3)

viz_3.fit(x_train_3, y_train)

viz_3 = FeatureImportances(clf_3)

viz_3.fit(x_train_3, y_train)

viz_3.poof()
df_train = df[['Id','MSSubClass','LotFrontage','LotArea','LotShape','LotConfig', 'Neighborhood','HouseStyle',

        'Condition1','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','RoofStyle',

        'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'BsmtExposure','BsmtFinType1',

        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC','1stFlrSF', '2ndFlrSF', 'GrLivArea',

        'FullBath', 'BsmtFullBath','BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional','Fireplaces',

        'GarageType','GarageYrBlt', 'GarageCars', 'GarageFinish', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

        'ScreenPorch','MoSold', 'YrSold','SalePrice']] 

# Included 'Id' and 'SalePrice' to complete the working feature set for training 
# Check correlation matrix for the selected features

import numpy as np

import matplotlib.pyplot as plt

mask = np.zeros_like(df_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr(), mask = mask, vmin = -1, annot = False, cmap = 'RdYlGn')
# Checking skewness of the selected features. If highly skewed, we should either discard or transform it.

skewness_of_features=[]

for col in df_train:

        skewness_of_features.append(df_train[col].skew())

print(skewness_of_features)
# Feature Transformation

# Checking if log-transformation can reduce its skewness

df_train['LotArea']=np.log(df_train['LotArea'])
# Check if skewness have been closer to zero now. If yes, then normality is restored.

print(df_train['LotArea'].skew())
import numpy as np

x_train = df_train[['Id','MSSubClass','LotFrontage','LotArea','LotShape','LotConfig', 'Neighborhood','HouseStyle',

        'Condition1','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','RoofStyle',

        'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'BsmtExposure','BsmtFinType1',

        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC','1stFlrSF', '2ndFlrSF', 'GrLivArea',

        'FullBath', 'BsmtFullBath','BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional','Fireplaces',

        'GarageType','GarageYrBlt', 'GarageCars', 'GarageFinish', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

        'ScreenPorch', 'MoSold', 'YrSold']] 
y_train = df_train['SalePrice']
import xgboost as xgb

from hyperopt import hp, tpe, fmin

from sklearn.model_selection import cross_val_score



valgrid = {'n_estimators':hp.quniform('n_estimators', 1500, 2500, 25),

         'gamma':hp.uniform('gamma', 0.01, 0.05),

         'base_score':hp.uniform('base_score',0.6,0.9),

         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),

         'max_depth':hp.quniform('max_depth', 3,8,1),

         'subsample':hp.uniform('subsample', 0.50, 0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.50, 0.95),

         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.50, 0.95),

         'colsample_bynode':hp.uniform('colsample_bynode', 0.50, 0.95),

         'reg_lambda':hp.uniform('reg_lambda', 1, 20)

        }



def objective(params):

    params = {'n_estimators': int(params['n_estimators']),

             'gamma': params['gamma'],

             'base_score': params['base_score'],

             'learning_rate': params['learning_rate'],

             'max_depth': int(params['max_depth']),

             'subsample': params['subsample'],

             'colsample_bytree': params['colsample_bytree'],

             'colsample_bylevel': params['colsample_bylevel'],

             'colsample_bynode': params['colsample_bynode'],  

             'reg_lambda': params['reg_lambda']}

    

    xb_a= xgb.XGBRegressor(**params)

    score = cross_val_score(xb_a, x_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()

    return -score



bestP = fmin(fn= objective, space= valgrid, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(bestP)
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



total = df_test[['MSSubClass','LotFrontage','LotArea','LotShape','LotConfig', 'Neighborhood','HouseStyle',

        'Condition1','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','RoofStyle',

        'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'BsmtExposure','BsmtFinType1',

        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC','1stFlrSF', '2ndFlrSF', 'GrLivArea',

        'FullBath', 'BsmtFullBath','BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional','Fireplaces',

        'GarageType','GarageYrBlt', 'GarageCars', 'GarageFinish', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

        'ScreenPorch', 'MoSold', 'YrSold']].isnull().sum().sort_values(ascending=False)



# Missing Value analysis for Test Data

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test.head(20)
# Missing Value imputation for Test Data

df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())

df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])

df_test['GarageYrBlt']=df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean())

df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])

df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])

df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])

df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])

df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean()) 

df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mode()[0]) #Imp.

df_test['Functional']=df_test['Functional'].fillna(df_test['Functional'].mode()[0])

df_test['KitchenQual']=df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])

df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())

df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())

df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])

df_test['Exterior1st']=df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())

df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())

df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())

df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['LotArea']=np.log(df_test['LotArea'])
# One-Hot encoding to convert categ. vars to Numeric vars for Test Data

from sklearn.preprocessing import LabelEncoder

df_test_work = df_test[['Id','MSSubClass','LotFrontage','LotArea','LotShape','LotConfig', 'Neighborhood','HouseStyle',

        'Condition1','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','RoofStyle',

        'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'BsmtExposure','BsmtFinType1',

        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC','1stFlrSF', '2ndFlrSF', 'GrLivArea',

        'FullBath', 'BsmtFullBath','BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional','Fireplaces',

        'GarageType','GarageYrBlt', 'GarageCars', 'GarageFinish', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

        'ScreenPorch', 'MoSold', 'YrSold']]



lencoders_test = {}

for colt in df_test_work.select_dtypes(include=['object']).columns:

    lencoders_test[colt] = LabelEncoder()

    df_test_work[colt] = lencoders_test[colt].fit_transform(df_test_work[colt])
# Prediction using XGB model having the best set of hyperparameters

import pandas as pd_out

import xgboost 

from sklearn.model_selection import train_test_split

import numpy as np

reg_final = xgboost.XGBRegressor(base_score=bestP['base_score'], 

            colsample_bylevel=bestP['colsample_bylevel'], colsample_bynode=bestP['colsample_bynode'], 

            colsample_bytree=bestP['colsample_bytree'], #eval_metric='rmse', 

            gamma=bestP['gamma'], learning_rate=bestP['learning_rate'],

            max_depth=int(bestP['max_depth']), n_estimators=int(bestP['n_estimators']), 

            #n_jobs=0,num_parallel_tree=1, objective='reg:squarederror', 

            random_state=42, reg_lambda=bestP['reg_lambda'], subsample=bestP['subsample'])

#X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=123)

#reg_final.fit(X_train,Y_train, early_stopping_rounds=5, eval_set=[(X_test, Y_test)], verbose=False)

reg_final.fit(x_train,y_train)

y_pred = reg_final.predict(df_test_work)

prediction = pd_out.DataFrame(y_pred)

output = pd_out.concat([df_test_work['Id'],prediction], axis=1)

output.columns=['Id','SalePrice']

output.to_csv('HousePrice_submission.csv', index=False)

print("Submission successful")