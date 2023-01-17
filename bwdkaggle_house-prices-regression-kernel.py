import numpy as np 
import pandas as pd 
# the following prevents certain warning messages due to editing copies of dataframes.
pd.options.mode.chained_assignment = None  # default='warn'

train_df = pd.read_csv("../input/train.csv", index_col = "Id")
test_df = pd.read_csv("../input/test.csv", index_col = "Id")
prices = train_df['SalePrice']
all_df = pd.concat([train_df.drop(["SalePrice"], axis=1), test_df])
# useful variable to split all_df up again
switchover = len(train_df) 
# Helper functions for converting qualitative grading systems

# Most quality grading scales fit the following format.
def qual_convert(qu):
    qualities = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}
    if qu in qualities:
        return qualities[qu]
    else: 
        return 0

# The slope is technically a graded scale also.
def slope_convert(slope):
    if slope == "Gtl":
        return 2
    if slope == "Mod":
        return 1
    else: return 0

# Basement qualitites. Debateable if this is the correct order, but seems to fit.
def bsmt_type_convert(typ):
    val = {"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1, "None":0}
    return val[typ]

def central_air_convert(yn):
        if yn=="Y": return 1
        else: return 0

# helper for filling a column with missing values by the mode
def mode_fill(df, col):
    return df[col].fillna(df[col].mode()[0])
# helper for filling a column with missing values by the mean
def mean_fill(df, col):
    return df[col].fillna(df[col].mean()[0])
# main data handling function

def data_handle(df):
    # There's not a clear way to fill the missing GarageYrBlt values so we drop them.
    new = df.drop(['GarageYrBlt'], axis=1) 
    
    # These columns have missing values due to the property simply missing the feature.
    # For these we fill the missing feature with the placeholder "None"
    missing_features = ['MiscFeature', 'Fence', 'BsmtExposure', 'Alley', 'GarageType', 'GarageFinish', 
                        'MasVnrType']

    for col in missing_features:
        new[col] = df[col].fillna("None")
        
    # Missing qualitative data can be replaced by the mode across the column.
    # Missing quantitative/continuous data can be replaced by the mean across the column.
    # Upon investigating some of the missing data, it is apparent we can use a 0-filling
    # e.g. the missing BsmtFinSF data is for a property with no basement.
    
    mode_filled_cols = ['Electrical', 'MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Exterior2nd',
                        'SaleType']
    zero_filled_cols = ['KitchenQual', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                        'GarageArea', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'GarageCars']
    mean_filled_cols = ['LotFrontage']
    
    for col in mode_filled_cols:
        new[col] = mode_fill(df, col)
    for col in zero_filled_cols:
        new[col] = df[col].fillna(0)
    for col in mean_filled_cols:
        new[col] = df[col].fillna(df[col].mean())
        
    
    # Here we convert the graded qualitative features to numerical scales
    # Most of these have a similar quality scale, and can use qual_convert.
    # LandSlope, BsmtFinTypes and CentralAir need to be done separately.    
    
    convert_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
                    'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in convert_cols:
        new[col] = df[col].fillna(0)
        new[col] = new[col].apply(qual_convert)
    
    new['LandSlope'] = df['LandSlope'].apply(slope_convert)
    new['BsmtFinType1'] = df['BsmtFinType1'].fillna("None")
    new['BsmtFinType2'] = df['BsmtFinType2'].fillna("None")
    new['BsmtFinType1'] = new['BsmtFinType1'].apply(bsmt_type_convert)
    new['BsmtFinType2'] = new['BsmtFinType2'].apply(bsmt_type_convert)
    new['CentralAir'] = df['CentralAir'].apply(central_air_convert)
    
    
    # Feature expansion, get the sale date as a relative number of days after some fixed date.
    # The idea is that this should be a more continuous approximation to any time v.s. price correlation than just the year.
    new['YrSold'] = df['YrSold'].apply(str)
    new['MoSold'] = df['MoSold'].apply(str)
    dates = new[['YrSold', 'MoSold']]
    dates['day'] = 1
    dates.columns = ['year', 'month', 'day']
    new['DateSold'] = pd.to_datetime(dates)
    
    # You may want to drop YrSold and MoSold as DayCount will be correlated to them.
    # I'm going to leave them in here.
    # new = new.drop(["YrSold", "MoSold"], axis=1)
    
    new['DayCount'] = (new['DateSold'] - pd.to_datetime('2006-01-01 00:00:00')).dt.days
    new = new.drop(["DateSold"], axis = 1)
    
    # convert MSSubclass to category
    new['MSSubClass'] = df['MSSubClass'].apply(str)
    
    return new
# Convert train and test sets
new_df = data_handle(all_df)
new_train = new_df[:switchover]
new_test = new_df[switchover:]

# Cut off training prices
prices = train_df['SalePrice']

# Concatenate train and test features so that following preprocessing is done on both.
all_features = new_df
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
numerical_cols = all_features.select_dtypes(include = [np.number])
categorical_cols = all_features.select_dtypes(exclude=[np.number])
encoded_cols =  all_features.select_dtypes(exclude=[np.number])

le = LabelEncoder()
for col in categorical_cols :
    trf = le.fit_transform(categorical_cols[col])
    encoded_cols[col] = trf
    
enc = OneHotEncoder(categorical_features = 'all')
onehot = enc.fit_transform(encoded_cols)
onehot = onehot.toarray()
# variance thresholder helper functions
def variance_threshold(df, threshold=0.0):
    from sklearn.feature_selection import VarianceThreshold
    Thresholder = VarianceThreshold(threshold)
    return Thresholder.fit_transform(df)
# be careful not to run this function twice on the same dataframe
# it will double expand the feature set, i.e. log(log), etc.

# adds box cox transformations of features
def expand_numerical_cols(numerical_cols):
    from scipy.special import boxcox1p
    lam = 0.2
    for col in numerical_cols:
        numerical_cols["Log "+ col] = np.log1p(numerical_cols[col])
        numerical_cols["Sqrt "+col] = np.sqrt(numerical_cols[col])
        numerical_cols[col] = boxcox1p(numerical_cols[col], lam)

def expand_features(onehot, numerical):
    expand_numerical_cols(numerical)
    features = np.concatenate((onehot, numerical), axis=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features
# have to reset numerical_cols so we don't expand an already expanded feature set
numerical_cols = all_features.select_dtypes(include = [np.number])
features = expand_features(onehot, numerical_cols)
print(features.shape)
train_features = features[:switchover,:]
test_features = features[switchover:, :]
from sklearn.model_selection import train_test_split
# Split off 1/3 of the data for validation.
X_train, X_valid, y_train, y_valid = train_test_split(train_features, np.log(prices), test_size = 0.3, random_state = 42)
num_features = X_train.shape[1]
num_train = X_train.shape[0]
y_train = y_train
y_valid = y_valid
num_valid = X_valid.shape[0]
# Use Lasso to select 100 features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import warnings #this cell is being awkward
warnings.filterwarnings('ignore')
reg = LassoCV(eps = 0.05, cv = 5, tol = 0.00001, alphas = (0.05,0.01,0.005, 0.001), max_iter = 250)
selector = RFE(reg, n_features_to_select = 100)
selector.fit(X_train, y_train)
ls_train = selector.transform(X_train)
ls_valid = selector.transform(X_valid)
ls_test = selector.transform(test_features)
print(ls_train.shape)
# helper functions

# root mean square error
def rmse(pred, true):
    error = 0
    for x,y in zip(pred,true):
        error+=(x-y)**2
    error/=len(pred)
    return np.sqrt(error)

# helper function to validate models
def validate_model(model, name, x_train, y_train, x_valid, y_valid):
    model.fit(x_train, y_train)
    # could print training error, but we're mostly interested in validation errors
    #print(name , " Training Error: ", rmse(model.predict(x_train), y_train))
    print(name , "Test Error: ", rmse(model.predict(x_valid), y_valid))
models = []
names = []
from sklearn.linear_model import LassoCV

models.append(LassoCV(eps = 0.01, cv = 10, tol = 0.00001, alphas = (1.0, 0.1, 0.01, 0.005, 0.001), max_iter = 1000))
names.append("Lasso")
from sklearn.linear_model import RidgeCV
models.append(RidgeCV(alphas = (0.001, 0.01,0.1, 1, 5, 10), cv = 10))
names.append("Ridge")

from sklearn.ensemble import GradientBoostingRegressor

models.append(GradientBoostingRegressor(loss='lad', learning_rate=0.05, 
                                n_estimators = 200, max_depth = 5))
names.append("Gradient Boosting")
from sklearn.ensemble import RandomForestRegressor
models.append(RandomForestRegressor(criterion = "mse", n_estimators = 15, max_depth = 5))
names.append("Random Forest")

from sklearn.ensemble import BaggingRegressor
models.append(BaggingRegressor(n_estimators=8, max_features=0.8, max_samples = 0.5))
names.append("Bagging")
for model, name in zip(models,names):
    validate_model(model,name, ls_train, y_train, ls_valid, y_valid)

# note, we do XGB separately as it seems to take more arguments
from xgboost import XGBRegressor
reg = XGBRegressor(n_estimators = 1000, learning_rate = 0.08)
reg.fit(ls_train, y_train,verbose = False, eval_set = [(ls_valid, y_valid)], early_stopping_rounds = 20)
predv_xgb = reg.predict(ls_valid)
print("XGB Test Error: ", rmse(predv_xgb,y_valid))
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators = 3000, learning_rate = 0.01)
xgb.fit(ls_train, y_train,verbose = False, eval_set = [(ls_valid, y_valid)], early_stopping_rounds = 500)
predv_xgb = xgb.predict(ls_valid)
print("XGB valid Error: ", rmse(predv_xgb,y_valid))
preds_train = np.zeros((len(ls_train), 3))
preds_valid = np.zeros((len(ls_valid),3))
for i in range(2):
    model = models[i]
    model.fit(ls_train, y_train)
    preds_train[:,i] = model.predict(ls_train)
    preds_valid[:,i] = model.predict(ls_valid)

preds_train[:,2] = xgb.predict(ls_train)
preds_valid[:,2] = predv_xgb
# in fact just averaging lasso and xgb seems to work best.
# round the predicted prices to the nearest 10 as most the data seems to be like this.
pred = np.log(np.rint((np.exp((preds_valid[:,0] +preds_valid[:,2])/2))/10)*10)
print("Stacked model valid error: ", rmse(pred, y_valid))
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import warnings #this cell is being awkward
warnings.filterwarnings('ignore')
reg = LassoCV(eps = 0.05, cv = 5, tol = 0.00001, alphas = (0.05,0.01,0.005, 0.001), max_iter = 250)
selector = RFE(reg, n_features_to_select = 100)
selector.fit(train_features, np.log(prices))
select_train = selector.transform(train_features)
select_test = selector.transform(test_features)

from sklearn.linear_model import LassoCV
reg = LassoCV(eps = 0.01, cv = 10, tol = 0.00001, alphas = (1.0, 0.1, 0.01, 0.005, 0.001), max_iter = 1000)
reg.fit(select_train, np.log(prices))
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators = 3000, learning_rate = 0.01)
xgb.fit(select_train, np.log(prices),verbose = False)

preds = (reg.predict(select_test) + xgb.predict(select_test))/2
preds = 10*np.rint(np.exp(preds)/10)

test_pred_df = pd.DataFrame(preds, index = test_df.index)
test_pred_df.columns = ['SalePrice']
test_pred_df.to_csv("submission.csv")


