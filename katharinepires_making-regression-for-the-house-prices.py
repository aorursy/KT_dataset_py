import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
abs(train.corr()['SalePrice']).nlargest(10)
(train)
sns.distplot(train['SalePrice'])
train['SalePrice'] = np.log1p(train['SalePrice'])
n = train.shape[0]
for col in train.columns:
    miss = sum(train[col].isnull())*100.0/n
    if miss > 80.0:
        print('{}: {:0.2f}%'.format(col, miss))
train.drop(['PoolQC', 'MiscFeature', 'Alley','Fence'], axis = 1, inplace = True)
test.drop(['PoolQC', 'MiscFeature', 'Alley','Fence'], axis = 1, inplace = True)
train.corr()['GarageArea']['GarageCars']
train.corr()['GarageArea']['SalePrice']
train.corr()['GarageCars']['SalePrice']
train.drop(['GarageArea'], axis = 1, inplace = True)
test.drop(['GarageArea'], axis = 1, inplace = True)
c = 'OverallQual'
train.corr()[c][abs(train.corr()[c]) > 0.65]
c = 'GarageCars'
train.corr()[c][abs(train.corr()[c]) > 0.65]
c = 'GrLivArea'
train.corr()[c][abs(train.corr()[c]) > 0.65]
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
train = train[train["GrLivArea"] < 4000]
c = 'TotalBsmtSF'
train.corr()[c][abs(train.corr()[c]) > 0.65]
plt.scatter(train['TotalBsmtSF'], train['SalePrice'])
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
train.drop(['2ndFlrSF', 'TotRmsAbvGrd', '1stFlrSF'], axis = 1, inplace = True)
test.drop(['2ndFlrSF', 'TotRmsAbvGrd', '1stFlrSF'], axis = 1, inplace = True)
abs(train.corr()['SalePrice']).nsmallest(10)
train.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', '3SsnPorch', 'MoSold', 'OverallCond', 'MSSubClass'], axis = 1, inplace = True)
test.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', '3SsnPorch', 'MoSold', 'OverallCond', 'MSSubClass'], axis = 1, inplace = True)
train.shape
abs(train.corr()['SalePrice']).nsmallest(10)
train.drop(['PoolArea', 'ScreenPorch', 'EnclosedPorch', 'KitchenAbvGr', 'BedroomAbvGr', 'BsmtUnfSF', 'BsmtFullBath', 'LotArea', 'HalfBath'], axis = 1, inplace = True)
test.drop(['PoolArea', 'ScreenPorch', 'EnclosedPorch', 'KitchenAbvGr', 'BedroomAbvGr', 'BsmtUnfSF', 'BsmtFullBath', 'LotArea', 'HalfBath'], axis = 1, inplace = True)
train.shape
train.head()
def fill_missing_data(df):
    data = df.copy()
    cat = []
    for c,dtype in data.dtypes.items():
        if dtype == 'object':
            cat.append(c)
    # Fill 'None' for the Categorical attribute
    data[cat] = data[cat].fillna('None')
    
    for c in data.columns:
        if c not in cat:
            data[c] = data[c].fillna(0) #Fill 0 for the Numeric attribute
    return data
train = fill_missing_data(train)
test = fill_missing_data(test)
train.columns
def feature_engineering(df):
    data = df.copy()
    
    feature = {
        'categorical':{
            'MSZoning': ['RL', 'RM', 'C (all)', 'FV', 'RH'],
            'LandContour': ['Lvl', 'Bnk', 'Low', 'HLS'],
            'LotConfig': ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'],
            'LandSlope': ['Gtl', 'Mod', 'Sev'],
            'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                            'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
            'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
            'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
            'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
            'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
            'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                           'VinylSd', 'Wd Sdng', 'WdShing'],
             'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                           'VinylSd', 'Wd Sdng', 'WdShing'],
            'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
            'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
            'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
            'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
            'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'None'],
            'GarageFinish': ['Fin', 'RFn', 'Unf', 'None'],
            'PavedDrive': ['Y', 'P', 'N'],
            'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'None'],
            'MoSold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            'YrSold': ['2006', '2007', '2008', '2009', '2010'],
            'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
            'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
        },
        'binary': {
            'Street': ['Pave', 'Grvl'],
            'CentralAir': ['Y', 'N']          
        },
        'ordinal': {
            'LotShape': ['None', 'IR3', 'IR2', 'IR1', 'Reg'],
            'Utilities': ['None', 'NoSeWa', 'NoSewr', 'AllPub'],
            'LandSlope': ['None', 'Sev', 'Mod', 'Gtl'],
            'ExterQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'KitchenQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
        },
    }
    
    selected = []
    for c in data.columns:
        if c in feature['binary']: # Convert the binary attributes to 0/1
            default_value = feature['binary'][c][0]
            feature_name = c + "_is_" + default_value
            selected.append(feature_name)
            data[feature_name] = data[c].apply(lambda x: int(x == default_value))
        elif c in feature['categorical']:
            values = feature['categorical'][c]
            for val in values:
                try:
                    new_name = "{}_{}".format(c, val)
                    selected.append(new_name)
                    data[new_name] = data[c].apply(lambda x: int(x == val))
                except Exception as err:
                    print("One-hot encoding for {}_{}. Error: {}".format(c, val, err))
        elif c in feature['ordinal']: # Convert the Ordinal attributes to a number
            new_name = c + "_ordinal"
            selected.append(new_name)
            data[new_name] = data[c].apply(lambda x: int(feature['ordinal'][c].index(x)))
        else: 
            selected.append(c)
            
    return data[selected]
train = feature_engineering(train)
test = feature_engineering(test)
train
for col in train.columns:
    if any(train[col]) == False:
        train.drop([col], axis = 1, inplace = True)
        test.drop([col], axis = 1, inplace = True)
X = test['Id']
Y = train['SalePrice']
train.drop(['Id', 'SalePrice'], axis = 1, inplace = True)
test.drop(['Id'], axis = 1, inplace = True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train)
train_a = scaler.transform(train)
test_r = scaler.transform(test)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_a, Y, test_size = 0.10, random_state = 1)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 123)
from sklearn.metrics import mean_squared_error
par = {
    "max_depth": 3, 
    "n_estimators": 20000, 
    "learning_rate": 0.02, 
    "subsample": 0.5,
    "colsample_bytree": 1,  
    "min_child_weight": 1.5,
    "reg_alpha": 0.75,
    "reg_lambda": 0.4,
    "seed": 42
}
#XGB Regressor:
import xgboost
xgb = xgboost.XGBRegressor(**par)
par_fit = {
    "eval_metric": "rmse",
    "early_stopping_rounds": 650,
    "verbose": 200,
    "eval_set": [(X_val, Y_val)]
}
xgb.fit(X_train, Y_train, **par_fit)
pred_xgb = xgb.predict(X_test)
mean_squared_error(Y_test, pred_xgb, squared = False)
#Lasso Regression:
from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=0.0007)
Lasso.fit(X_train, Y_train)
pred_Lasso = Lasso.predict(X_test)
mean_squared_error(Y_test, pred_Lasso, squared = False)
#Nearest Neighbors:
from sklearn.neighbors import KNeighborsRegressor
KNR = KNeighborsRegressor()
KNR.fit(X_train, Y_train)
pred_KNR = KNR.predict(X_test)
mean_squared_error(Y_test, pred_KNR, squared = False)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state = 0, min_samples_split = 2, max_depth = 6)  
tree.fit(X_train, Y_train)
pred_tree = tree.predict(X_test)
mean_squared_error(Y_test, pred_tree, squared = False)
#Random Forest:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 550)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)
mean_squared_error(Y_test, pred_rf, squared = False)
#Gradient Boosting Regressor:
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(
    max_depth = 5,
    n_estimators = 2000,
    learning_rate = 0.25
)
gb.fit(X_train, Y_train)
pred_gb = gb.predict(X_test)
mean_squared_error(Y_test, pred_gb, squared = False)
pred = xgb.predict(test_r)
pred = np.exp(pred)
sub = pd.DataFrame({'Id': X, 'SalePrice': pred})
sub.head()
sub.to_csv('submission.csv', index = False)