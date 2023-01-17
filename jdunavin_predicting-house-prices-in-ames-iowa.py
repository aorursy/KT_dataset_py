import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import matplotlib.ticker as tick

import seaborn as sb

from sklearn import linear_model

from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
train = pd.read_csv("../input/train.csv", header=0)

test = pd.read_csv("../input/test.csv", header=0)
train = train.fillna(0)

test = test.fillna(0)
features = [x for x in train.columns if x not in ['id','SalePrice']]
cat_features = ['MSSubClass', 'MSZoning', 'Street','Alley', 

'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

'HouseStyle','OverallQual','OverallCond',

'Foundation', 'BsmtFinType1','BsmtFinType2', 

'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',

'Heating','CentralAir','Electrical','BsmtFullBath',

'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

'TotRmsAbvGrd','Functional','Fireplaces','GarageType',

'GarageFinish','GarageCars','PavedDrive',

'Fence','MiscFeature','SaleType','SaleCondition']
ord_cat_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',

                   'HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
num_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2',

'BsmtUnfSF','TotalBsmtSF', 'LowQualFinSF','1stFlrSF','2ndFlrSF',

'GrLivArea', 'YearRemodAdd', 'YearBuilt','GarageYrBlt','GarageArea','WoodDeckSF',

'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']
rows_train = len(train)

rows_test = len(test)
def transform_nb(x):

    if x in ("NoRidge", "NridgHt", "StoneBr","Timber","Veenker"):

        return "Hi"

    elif x in ("Mitchel","OldTown","BrkSide","Sawyer","NAmes","IDOTRR","MeadowV","Edwards","NPkVill","BrDale","Blueste"):

        return "Low"

    else:

        return "Mid"
# glue data sets together

train_test = pd.concat((train[features], test[features])).reset_index(drop=True)

# Convert categoricals to codes

for c in range(len(cat_features)):

    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes
#Define a generic function using Pandas replace function

def coding(col, codeDict):

  colCoded = pd.Series(col, copy=True)

  for key, value in codeDict.items():

    colCoded.replace(key, value, inplace=True)

  return colCoded
train_test['ExterQual'] = coding(train_test['ExterQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})

train_test['ExterCond'] = coding(train_test['ExterCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})

train_test['BsmtQual'] = coding(train_test['BsmtQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

train_test['BsmtCond'] = coding(train_test['BsmtCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

train_test['BsmtExposure'] = coding(train_test['BsmtExposure'], {'Gd':4,'Av':3,'Mn':2,'No':1, 0:0})
train_test['HeatingQC'] = coding(train_test['HeatingQC'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})

train_test['KitchenQual'] = coding(train_test['KitchenQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1})

train_test['FireplaceQu'] = coding(train_test['FireplaceQu'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

train_test['GarageQual'] = coding(train_test['GarageQual'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

train_test['GarageCond'] = coding(train_test['GarageCond'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})

train_test['PoolQC'] = coding(train_test['PoolQC'], {'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1, 0:0})
trainX = train_test.iloc[:rows_train,:]

testX = train_test.iloc[rows_train:,:]
fig = plt.figure()

ax = plt.axes()

n,bins,patches=plt.hist(train['SalePrice'], 30, facecolor='dimgrey')

plt.xlabel("Sale price (000s)")

vals=ax.get_xticks()

ax.set_xticklabels(['${:,.0f}'.format(x/1000) for x in vals])

plt.show()
target = np.log(train['SalePrice'])
fig = plt.figure()

ax = plt.axes()

n,bins,patches=plt.hist(target, 30, facecolor='dimgrey')

plt.xlabel("Log(Sale price)")

plt.show()
def doPlots(x, data, ii, fun):

    fig, axes = plt.subplots(len(ii) // 2, ncols = 2)

    fig.tight_layout()

    for i in range(len(ii)):

        fun(x=x[ii[i]], data=data, ax=axes[i // 2, i % 2], color='dimgrey')

    plt.show()
doPlots(cat_features, train, range(0,8), sb.countplot)
doPlots(cat_features, train, range(8,16), sb.countplot)
doPlots(cat_features, train, range(16,24), sb.countplot)
doPlots(cat_features, train, range(24,32), sb.countplot)
doPlots(cat_features, train, range(32,40), sb.countplot)
doPlots(cat_features, train, range(40,44), sb.countplot)
np.average(train['YearBuilt'][train['Foundation']=='PConc'])
np.average(train['YearBuilt'][train['Foundation']=='CBlock'])
def doHistPlots(x, data, ii, fun):

    fig, axes = plt.subplots(len(ii) // 2, ncols = 2)

    fig.tight_layout()

    for i in range(len(ii)):

        fun(data[x[ii[i]]], color='b', ax=axes[i // 2, i % 2], kde=False)

    plt.show()
doHistPlots(x=num_features, data=train, ii=range(0,6), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(6,12), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(12,18), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(18,24), fun=sb.distplot)
train[train['TotalBsmtSF']>4000]
print(train['TotalBsmtSF'].iloc[1298,])

print(train['GrLivArea'].iloc[1298,])

print(train['OverallCond'].iloc[1298,])

print(train['OverallQual'].iloc[1298,])
trainX = trainX.drop(train.index[[1298]])
plt.rcParams['figure.figsize'] = (8.75, 7.0)

ax = plt.axes()

plot1 = sb.boxplot(data=train, x='Neighborhood', y='SalePrice')

ax.set_title("Price distribution by neighborhood")

sb.despine(offset=10, trim=True)

plt.xticks(rotation=90)

plt.show()
def transform_nb(x):

    if x in ("NoRidge", "NridgHt", "StoneBr"):

        return 5 #Over 250

    elif x in ('CollgCr','Veenker','Crawfor','Somerst','Timber','ClearCr'):

        return 4 #200-250

    elif x in ('Mitchel','NWAmes','SawyerW','Gilbert','Blmngtn','SWISU', 'Blueste'):

        return 3 #150-200

    elif x in ('OldTown','BrkSide','Sawyer','NAmes','IDOTRR','Edwards','BrDale', 'NPkVill'):

        return 2 #100 - 150

    elif x in ('MeadowV'):

        return 1

    else:

        return 9 # Catch mistakes
train['NbdClass'] = train['Neighborhood'].apply(transform_nb)

test['NbdClass'] = test['Neighborhood'].apply(transform_nb)

trainX['NbdClass'] = train['NbdClass']

# This is messed up and gives a warning, I couldn't figure out how to make it go away

z = test.loc[:,'NbdClass']

z=np.asarray(z)

testX.loc[:,'NbdClass'] = z
trainX[trainX['NbdClass']==9]
plt.rcParams['figure.figsize'] = (8.75, 7.0)

ax = plt.axes()

plot1 = sb.boxplot(data=train, x='NbdClass', y='SalePrice')

ax.set_title("Price distribution by neighborhood")

sb.despine(offset=10, trim=True)

plt.xticks(rotation=90)

plt.show()
use_features = ['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','GrLivArea','TotalBsmtSF','FullBath','Fireplaces',

           'GarageYrBlt','GarageCars','NbdClass','HouseStyle','Alley','LotShape','LotConfig','Exterior1st',

           'Exterior2nd','ExterQual','ExterCond','Foundation','BsmtExposure','BsmtFinType1','HeatingQC','CentralAir',

           'FireplaceQu','GarageType','GarageFinish']
# Test shrinkage parameters in cross-validation

alphas = np.logspace(-10,-1,num=20)
# Kick out that one weird house

trainY = train['SalePrice']

trainY = trainY.drop(train.index[[1298]])

target = np.log(trainY)
lasso = linear_model.LassoCV(alphas=alphas, tol=0.0001, selection='random', random_state=17, max_iter=1000)

lasso.fit(trainX[use_features], target)
print("Best alpha is",lasso.alpha_)
lasso.pred = lasso.predict(trainX[use_features])

print("Competition training RMSE:", np.sqrt(np.sum(lasso.pred - target)**2))
# Look at what did and didn't get squeezed out

indices = np.arange(len(trainX[use_features].columns))

indices_nz = np.nonzero(lasso.coef_)

indices_z = np.setdiff1d(indices, indices_nz)

set([use_features[i] for i in indices_z])
# see most important and least important features

tmp = [use_features[i] for i in indices_nz[0]]

tmp = np.array(tmp)

tmp2 = lasso.coef_[lasso.coef_ != 0]

lasso.c = pd.DataFrame({'cols':tmp, 'coefs':tmp2})

lasso.c = lasso.c.sort_values(by='coefs',ascending=False)
lasso.c[:5]
lasso.c[-5:]
sollasso = pd.DataFrame({'Id':test['Id'], 'SalePrice':np.exp(lasso.predict(testX[use_features]))})
ntrees = np.arange(100,1001,100)

depths = np.arange(1,8)
# Warning: This takes a bit

scores=[]

X = trainX[use_features]

Y = target

for n in ntrees:

    for d in depths:

        run_tot = 0

        for k in np.arange(0,10):

            

            # Get the kth fold of data

            testXcv = X[k*146:k*146+146]

            trainX_left = X[:k*146]

            trainX_right = X[k*146+146:]

            trainXcv = pd.concat([trainX_left, trainX_right])

            

            testYcv = Y[k*146:k*146+146]

            trainY_left = Y[:k*146]

            trainY_right = Y[k*146+146:]

            trainYcv = np.concatenate([trainY_left, trainY_right])

            

            # Fit a model and make predictions using the kth fold

            rf = RandomForestRegressor(n_estimators=n, max_depth=d, n_jobs=-1)

            rf.fit(trainXcv, trainYcv)

            preds = rf.predict(testXcv)

            

            # Add this fold's score to the previous ones

            run_tot = run_tot + np.sqrt(np.sum((preds - testYcv)**2))

            

        # Now we load the scores table with the parameters we're testing and the 10-fold average score

        scores.append({'ntrees':n, 'depth':d, 'score':run_tot/10.0})

        # Tell me where we're at - Comment this out if you find it annoying

    print("ntrees =",n)

scoredf = pd.DataFrame(scores)

print("Done!")
fig=plt.figure()

ax = plt.axes()

for d in depths:

    x = scoredf['ntrees'][scoredf['depth']==d]

    y = scoredf['score'][scoredf['depth']==d]

    plt.plot(x,y, label='Depth ' + str(d))

plt.legend(loc=9, ncol=len(depths)//2) # upper center

plt.xlabel("Number of trees")

plt.ylabel("Competition RMSE")

ax.set_title("Random forest - Competition RMSE by number of trees and depth")

plt.show()
rf_best = RandomForestRegressor(n_jobs=-1, n_estimators=1000, max_depth=7, random_state=17)

rf_best.fit(trainX[use_features], target)
importances = pd.DataFrame({'Feature':trainX[use_features].columns, 'Importance':rf_best.feature_importances_})

importances = importances.sort_values('Importance',ascending=False).set_index('Feature')

importances[0:10].iloc[::-1].plot(kind='barh',legend=False)

plt.show()
rf.preds = np.exp(rf_best.predict(testX[use_features]))

solrf = pd.DataFrame({'Id':test['Id'], 'SalePrice':rf.preds})

solrf.to_csv("./solrf2.csv", index=False)

solrf.head(3)
rf_all = RandomForestRegressor(n_jobs=-1, n_estimators=1000, max_depth=7, random_state=17)

rf_all.fit(trainX, target)
importances = pd.DataFrame({'Feature':trainX.columns, 'Importance':rf_all.feature_importances_})

importances = importances.sort_values('Importance',ascending=False).set_index('Feature')

importances[0:10].iloc[::-1].plot(kind='barh',legend=False)

plt.show()
rf.preds = np.exp(rf_all.predict(testX))

solrf = pd.DataFrame({'Id':test['Id'], 'SalePrice':rf.preds})

solrf.to_csv("./solrf-full.csv", index=False)

solrf.head(3)
train_X = trainX.iloc[:,1:].as_matrix()

test_X = testX.iloc[:,1:].as_matrix()
gbm = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, target)
importances = pd.DataFrame({'Feature':trainX.iloc[:,1:].columns, 'Importance':gbm.feature_importances_})

importances = importances.sort_values('Importance',ascending=False).set_index('Feature')

importances[0:10].iloc[::-1].plot(kind='barh',legend=False)

plt.title("XGBoost - Feature importance")

plt.show()
xgb_preds = np.exp(gbm.predict(test_X))
solxgb = pd.DataFrame({'Id':test['Id'], 'SalePrice':xgb_preds})

solxgb.to_csv("./solxgb.csv", index=False)

solxgb.head(3)