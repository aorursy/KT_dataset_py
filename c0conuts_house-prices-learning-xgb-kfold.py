# Imports
%matplotlib inline
%reload_ext autoreload
%autoreload 2
import xgboost as xgb
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from IPython.display import HTML, display
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import KFold
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df["PoolQC"] = train_df["PoolQC"].fillna("None")
train_df["MiscFeature"] = train_df["MiscFeature"].fillna("None")
train_df["Alley"] = train_df["Alley"].fillna("None")
train_df["Fence"] = train_df["Fence"].fillna("None")
train_df["FireplaceQu"] = train_df["FireplaceQu"].fillna("None")
train_df["LotFrontage"] = train_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train_df[col] = train_df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train_df[col] = train_df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train_df[col] = train_df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_df[col] = train_df[col].fillna('None')
train_df["MasVnrType"] = train_df["MasVnrType"].fillna("None")
train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(0)
train_df['MSZoning'] = train_df['MSZoning'].fillna(train_df['MSZoning'].mode()[0])
train_df = train_df.drop(['Utilities'], axis=1)
train_df["Functional"] = train_df["Functional"].fillna("Typ")
train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
train_df['KitchenQual'] = train_df['KitchenQual'].fillna(train_df['KitchenQual'].mode()[0])
train_df['Exterior1st'] = train_df['Exterior1st'].fillna(train_df['Exterior1st'].mode()[0])
train_df['Exterior2nd'] = train_df['Exterior2nd'].fillna(train_df['Exterior2nd'].mode()[0])
train_df['SaleType'] = train_df['SaleType'].fillna(train_df['SaleType'].mode()[0])
train_df['MSSubClass'] = train_df['MSSubClass'].fillna("None")
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']

test_df["PoolQC"] = test_df["PoolQC"].fillna("None")
test_df["MiscFeature"] = test_df["MiscFeature"].fillna("None")
test_df["Alley"] = test_df["Alley"].fillna("None")
test_df["Fence"] = test_df["Fence"].fillna("None")
test_df["FireplaceQu"] = test_df["FireplaceQu"].fillna("None")
test_df["LotFrontage"] = test_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    test_df[col] = test_df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    test_df[col] = test_df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    test_df[col] = test_df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test_df[col] = test_df[col].fillna('None')
test_df["MasVnrType"] = test_df["MasVnrType"].fillna("None")
test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(0)
test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])
test_df = test_df.drop(['Utilities'], axis=1)
test_df["Functional"] = test_df["Functional"].fillna("Typ")
test_df['Electrical'] = test_df['Electrical'].fillna(test_df['Electrical'].mode()[0])
test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df['MSSubClass'] = test_df['MSSubClass'].fillna("None")
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
            'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond', 'Foundation', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
            'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
            'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
contin_vars = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
               'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
               'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
               'PoolArea', 'MiscVal', 'TotalSF']
date_vars = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
objective = ['SalePrice']
# Apply categorical type:
for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()

apply_cats(test_df, train_df)
# Contin_vars as floats:
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
# Feature engineering:

for df in (train_df, test_df):
    df['House_age_when_sold'] = df['YrSold'].astype(int) - df['YearBuilt'].astype(int)

new_contin_vars = ['House_age_when_sold']

# Contin_vars as floats:
for v in new_contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')
# Process the training data using the awesome fastai function proc_df:
train_df = train_df.set_index('Id')
train_df = train_df[cat_vars+contin_vars+new_contin_vars+objective]

df, y, nas, mapper = proc_df(train_df, 'SalePrice', do_scale=True)
# Process the testing data using the awesome fastai function proc_df:
test_df = test_df.set_index('Id')

# Just a dummy column so that the column exists.
test_df['SalePrice'] = 0
test_df = test_df[cat_vars+contin_vars+new_contin_vars+['SalePrice']]

df_test, _, nas, mapper = proc_df(test_df, 'SalePrice', do_scale=True,
                                  mapper=mapper, na_dict=nas)
# Create a K-fold instance
k = 5
kf = KFold(n_splits = k, shuffle = True, random_state=1)
# Initilize a list to gather the k predictions sets:
xgb_preds = []

# Train/validate on 5 different training/validation sets thanks to K-folds and predict on the testing set:
for train_index, test_index in kf.split(np.array(df)):

    train_X, valid_X = np.array(df)[train_index], np.array(df)[test_index]
    train_y, valid_y = y[train_index], y[test_index]
    test_X = np.array(df_test)

    
    xgb_params = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'verbose': False,
        'seed': 27}
    
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(test_X)
    
    model = xgb.train(xgb_params, d_train, num_boost_round = 10000, evals=[(d_valid, 'eval')], verbose_eval=100, 
                     early_stopping_rounds=100)
                        
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
# Average the k predictions sets:
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(k):
        sum+=xgb_preds[j][i]
    preds.append(sum / k)
# Create the results datframe:
df_test['SalePrice'] = np.expm1(np.array(preds).astype(float))
sub = df_test[['SalePrice']].copy()
sub.reset_index(inplace=True, drop=False)
sub.head()
sub.to_csv('sample_submission.csv', index=False)