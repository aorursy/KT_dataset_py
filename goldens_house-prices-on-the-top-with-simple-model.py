import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import cross_val_score



import xgboost as xgb

from hyperopt import hp, tpe, fmin





pd.set_option('display.max_columns', None)
traindf=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

testdf=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
traindf.info()
print(traindf.duplicated().sum())

print(testdf.duplicated().sum())
list_of_numerics=traindf.select_dtypes(include=['float','int']).columns

types= traindf.dtypes

missing= round((traindf.isnull().sum()/traindf.shape[0]),3)*100

overview= traindf.apply(lambda x: [round(x.min()), 

                                 round(x.max()), 

                                 round(x.mean()), 

                                 round(x.quantile(0.5))] if x.name in list_of_numerics else x.unique())



outliers= traindf.apply(lambda x: sum(

                                 (x<(x.quantile(0.25)-1.5*(x.quantile(0.75)-x.quantile(0.25))))|

                                 (x>(x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))))

                                 if x.name in list_of_numerics else ''))





explo = pd.DataFrame({'Types': types,

                      'Missing%': missing,

                      'Overview': overview,

                      'Outliers': outliers}).sort_values(by=['Missing%','Types'],ascending=False)

explo.transpose()
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    traindf[col]=traindf[col].fillna('None')

    testdf[col]=testdf[col].fillna('None')
for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    traindf[col]=traindf[col].fillna(traindf[col].mode()[0])

    testdf[col]=testdf[col].fillna(traindf[col].mode()[0])
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

            'GarageYrBlt','GarageCars','GarageArea'):

    traindf[col]=traindf[col].fillna(0)

    testdf[col]=testdf[col].fillna(0)
traindf['LotFrontage']=traindf['LotFrontage'].fillna(traindf['LotFrontage'].mean())

testdf['LotFrontage']=testdf['LotFrontage'].fillna(traindf['LotFrontage'].mean())
print(traindf.isnull().sum().sum())
fig, axes = plt.subplots(1,2, figsize=(12,5))



ax1= sns.scatterplot(x='GrLivArea', y='SalePrice', data= traindf,ax=axes[0])

ax2= sns.boxplot(x='GrLivArea', data= traindf,ax=axes[1])
#removing outliers recomended by author

traindf= traindf[traindf['GrLivArea']<4000]
plt.figure(figsize=[12,14])

features=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold', 'SalePrice']

n=1

for f in features:

    plt.subplot(10,4,n)

    sns.distplot(traindf[f], kde=False)

    sns.despine()

    n=n+1

plt.tight_layout()

plt.show()
len_traindf=traindf.shape[0]

houses= pd.concat([traindf, testdf], sort=False)



# turning some ordered categorical variables into ordered numerical

# maybe this information about order can help on performance

for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",

            "FireplaceQu","GarageQual","GarageCond","PoolQC"]:

    houses[col]= houses[col].map({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1})





# turning categoric into numeric

houses= pd.get_dummies(houses)



# separating

traindf= houses[:len_traindf]

testdf= houses[len_traindf:]
# x/y split

xtrain= traindf.drop('SalePrice', axis=1)

ytrain= traindf['SalePrice']

xtest= testdf.drop('SalePrice', axis=1)
#ytrain = np.log(ytrain)
space = {'n_estimators':hp.quniform('n_estimators', 1000, 4000, 100),

         'gamma':hp.uniform('gamma', 0.01, 0.05),

         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.025),

         'max_depth':hp.quniform('max_depth', 3,7,1),

         'subsample':hp.uniform('subsample', 0.60, 0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.98),

         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.60, 0.98),

         'reg_lambda': hp.uniform('reg_lambda', 1, 20)

        }



def objective(params):

    params = {'n_estimators': int(params['n_estimators']),

             'gamma': params['gamma'],

             'learning_rate': params['learning_rate'],

             'max_depth': int(params['max_depth']),

             'subsample': params['subsample'],

             'colsample_bytree': params['colsample_bytree'],

             'colsample_bylevel': params['colsample_bylevel'],

             'reg_lambda': params['reg_lambda']}

    

    xb_a= xgb.XGBRegressor(**params)

    score = cross_val_score(xb_a, xtrain, ytrain, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()

    return -score

best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)
print(best)
xb_b = xgb.XGBRegressor(random_state=0,

                        n_estimators=int(best['n_estimators']), 

                        colsample_bytree= best['colsample_bytree'],

                        gamma= best['gamma'],

                        learning_rate= best['learning_rate'],

                        max_depth= int(best['max_depth']),

                        subsample= best['subsample'],

                        colsample_bylevel= best['colsample_bylevel'],

                        reg_lambda= best['reg_lambda']

                       )



xb_b.fit(xtrain, ytrain)
#prediction

preds= xb_b.predict(xtest)

#preds2= np.exp(preds)
#output

output = pd.DataFrame({'Id': testdf.Id,'SalePrice': preds})

output.to_csv('submission.csv', index=False)