# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import some necessary librairies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew # for some statistics

# Settings
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = (12, 4)
pd.options.display.max_columns = 500
# Now let's import and put the train and test datasets in  pandas dataframe

#train_df_org = pd.read_csv('house_prices_train.csv')
#test_df_org = pd.read_csv('house_prices_test.csv')

train_df_org = pd.read_csv('../input/train.csv')
test_df_org = pd.read_csv('../input/test.csv')

# train set dimension
print('Train dataset dimension: {} rows, {} columns'.format(train_df_org.shape[0], train_df_org.shape[1]))

# test set dimension
print('Test dataset dimension: {} rows, {} columns'.format(test_df_org.shape[0], test_df_org.shape[1]))

print('First few observations of AMES housing prices train dataset: ')
train_df_org.head()
# Metadata of Titatnic dataset
object_col_names = train_df_org.select_dtypes(include=[np.object]).columns.tolist()
int_col_names = train_df_org.select_dtypes(include=[np.int64]).columns.tolist()
float_col_names = train_df_org.select_dtypes(include=[np.float64]).columns.tolist()
target_var = 'SalesPrice'

num_col_names = int_col_names + float_col_names
total_col_names = object_col_names + int_col_names + float_col_names

if len(total_col_names) == train_df_org.shape[1]:
    print('Number of Features count matching. Train Dataset Features: ', train_df_org.shape[1], ' Features Count: ', len(total_col_names))
else:
    print('Number of Features count not matching. Train Dataset Features: ', train_df_org.shape[1], ' Features Count: ', len(total_col_names))

print('\nTotal number of object features: ', len(object_col_names))
print(object_col_names)

print('\nTotal number of integer features: ', len(int_col_names))
print(int_col_names)

print('\nTotal number of float features: ', len(float_col_names))
print(float_col_names)
train_df_org.describe()
train_df_proc = train_df_org.copy()
test_df_proc = test_df_org.copy()

#Save the 'Id' column
train_ID = train_df_proc['Id']
test_ID = test_df_proc['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df_proc.drop("Id", axis = 1, inplace = True)
test_df_proc.drop("Id", axis = 1, inplace = True)

# train set dimension
print('Size of train dataset after dropping Id: {} rows, {} columns'.format(train_df_proc.shape[0], train_df_proc.shape[1]))

# test set dimension
print('Size of train dataset after dropping Id: {} rows, {} columns'.format(test_df_proc.shape[0], test_df_proc.shape[1]))
# Create the default pairplot
plot_cols1 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'SalePrice']
plot_cols2 = ['GrLivArea','LotArea', 'PoolArea', 'GarageArea', '2ndFlrSF', 'SalePrice']

sns.pairplot(train_df_proc[plot_cols1]);
sns.pairplot(train_df_proc[plot_cols2]);
fig, ax = plt.subplots()
ax.scatter(x = train_df_proc['GrLivArea'], y = train_df_proc['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show();
train_df_proc[(train_df_proc['GrLivArea']>4000) & (train_df_proc['SalePrice']<300000)]
#Deleting outliers
train_df_proc = train_df_proc.drop(train_df_proc[(train_df_proc['GrLivArea']>4000) & (train_df_proc['SalePrice']<300000)].index)
# most correlated features
corrmat = train_df_proc.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_df_proc[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.barplot(train_df_proc.OverallQual,train_df_proc.SalePrice);
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df_proc[cols], size = 2.5)
plt.show();
def check_skewness(col):
    sns.distplot(train_df_proc[col] , fit=norm);
    fig = plt.figure()
    res = stats.probplot(train_df_proc[col], plot=plt)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_df_proc[col])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    
check_skewness('SalePrice')
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df_proc["SalePrice"] = np.log1p(train_df_proc["SalePrice"])

check_skewness('SalePrice')
ntrain = train_df_proc.shape[0]
ntest = test_df_proc.shape[0]
y_train = train_df_proc.SalePrice.values
all_df = pd.concat((train_df_proc, test_df_proc)).reset_index(drop=True)
all_df.drop(['SalePrice'], axis=1, inplace=True)

print('Size of train & test dataset comined: {} rows, {} columns'.format(all_df.shape[0], all_df.shape[1]))
null_feat_df = pd.DataFrame()
null_feat_df['Null Count'] = all_df.isnull().sum().sort_values(ascending=False)
null_feat_df['Null Pct'] = null_feat_df['Null Count'] / float(len(all_df))

null_feat_df = null_feat_df[null_feat_df['Null Pct'] > 0]

total_null_feats = null_feat_df.shape[0]
null_feat_names = null_feat_df.index
print('Total number of features having null values: ', total_null_feats)
print('Name of features having null values: ', null_feat_names)

f, ax = plt.subplots(figsize=(12, 4))
plt.xticks(rotation='90')
sns.barplot(x=null_feat_df.index, y=null_feat_df['Null Pct'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15);
none_col = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'KitchenQual',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in none_col:
    all_df[none_col] = all_df[none_col].fillna('None')

zero_col = ['GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF']
for col in zero_col:
    all_df[col] = all_df[col].fillna(0)

mode_col = ['MSZoning', 'Exterior1st', 'Exterior2nd']
for col in mode_col:
    all_df[col] = all_df[col].fillna(all_df[col].mode()[0])

other_col = ['Functional', 'Utilities', 'Electrical', 'SaleType', 'LotFrontage', 'GarageYrBlt']
all_df['Functional'] = all_df['Functional'].fillna('Typ')
all_df['Utilities'] = all_df['Utilities'].fillna('AllPub')
#all_df['MSSubClass'] = all_df['MSSubClass'].fillna(190)
all_df['Electrical'] = all_df['Electrical'].fillna('SBrkr')
all_df['SaleType'] = all_df['SaleType'].fillna('Oth')
#all_df['GarageYrBlt'] = all_df['GarageYrBlt'].fillna(all_df['YearBuilt'])
all_df['GarageYrBlt'] = all_df['GarageYrBlt'].fillna(0)

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

total_impute_cols = none_col + zero_col + mode_col + other_col

if len(null_feat_names) == len(total_impute_cols):
    print('Number of Null Features count matching. Null Features: ', len(null_feat_names), ' Imputed Features: ', len(total_impute_cols))
    print(set(null_feat_names) - set(total_impute_cols))
else:
    print('Number of Null Features count not matching. Null Features: ', len(null_feat_names), ' Imputed Features: ', len(total_impute_cols))
    print(set(total_impute_cols) - set(null_feat_names))
null_feat_df = pd.DataFrame()
null_feat_df['Null Count'] = all_df.isnull().sum().sort_values(ascending=False)
null_feat_df['Null Pct'] = null_feat_df['Null Count'] / float(len(all_df))

null_feat_df = null_feat_df[null_feat_df['Null Pct'] > 0]

total_null_feats = null_feat_df.shape[0]
null_feat_names = null_feat_df.index
print('Total number of features having null values: ', total_null_feats)
print('Name of features having null values: ', null_feat_names)
# Basic statistics of categorical features
all_df.describe(include=[np.object])
all_df = all_df.drop(['Utilities'], axis=1)
print('Size of dataset after removing Utilities feature: {} rows, {} columns'.format(all_df.shape[0], all_df.shape[1]))
# Basic statistics of categorical features
all_df.describe()
#MSSubClass=The building class
all_df['MSSubClass'] = all_df['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_df['OverallCond'] = all_df['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_df['YrSold'] = all_df['YrSold'].astype(str)
all_df['MoSold'] = all_df['MoSold'].astype(str)

# Additional Attributes
#all_df['OverallQual'] = all_df['OverallQual'].astype(str)
#all_df['YearBuilt'] = all_df['YearBuilt'].astype(str)
#all_df['YearRemodAdd'] = all_df['YearRemodAdd'].astype(str)
#all_df['GarageYrBlt'] = all_df['GarageYrBlt'].astype(str)
from sklearn.preprocessing import LabelEncoder
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'YrSold', 'MoSold',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond'] 
#        'OverallQual', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
# process columns, apply LabelEncoder to categorical features
for c in cols:
    label_enc = LabelEncoder() 
    label_enc.fit(list(all_df[c].values)) 
    all_df[c] = label_enc.transform(list(all_df[c].values))

# shape        
print('Size of dataset after label encoding: {} rows, {} columns'.format(all_df.shape[0], all_df.shape[1]))
# Adding total sqfootage feature 
all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']
numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

sns.barplot(skewness.index,skewness.Skew);
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_df[feat] = boxcox1p(all_df[feat], lam)
all_df = pd.get_dummies(all_df)

# shape        
print('Size of dataset after dummies: {} rows, {} columns'.format(all_df.shape[0], all_df.shape[1]))
final_train_df = all_df[:ntrain]
final_test_df = all_df[ntrain:]

# shape        
print('Size of training dataset: {} rows, {} columns'.format(final_train_df.shape[0], final_train_df.shape[1]))
print('Size of testing dataset: {} rows, {} columns'.format(final_test_df.shape[0], final_test_df.shape[1]))
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#Validation function
n_folds = 5

def kfold_cv_rmsle(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmsle = np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmsle)

def kfold_cv_pred(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    y_pred = cross_val_predict(model, X.values, y, cv=kf)

    return(y_pred)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = kfold_cv_rmsle(KRR, final_train_df, y_train)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = kfold_cv_rmsle(lasso, final_train_df, y_train)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = kfold_cv_rmsle(ENet, final_train_df, y_train)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = kfold_cv_rmsle(GBoost, final_train_df, y_train)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = kfold_cv_rmsle(model_xgb, final_train_df, y_train)
print("XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = kfold_cv_rmsle(model_lgb, final_train_df, y_train)
print("LightGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
LassoMd = lasso.fit(final_train_df.values,y_train)
ENetMd = ENet.fit(final_train_df.values,y_train)
KRRMd = KRR.fit(final_train_df.values,y_train)
GBoostMd = GBoost.fit(final_train_df.values,y_train)
from sklearn.metrics import mean_squared_error

lasso_train_pred = LassoMd.predict(final_train_df.values)
ENet_train_pred = ENetMd.predict(final_train_df.values)
KRR_train_pred = KRRMd.predict(final_train_df.values)
GBoost_train_pred = GBoostMd.predict(final_train_df.values)

avg_train_pred = (lasso_train_pred + ENet_train_pred + KRR_train_pred + GBoost_train_pred) / 4

avg_rmsle = np.sqrt(mean_squared_error(y_train, avg_train_pred))
print("Average Model RMSLE score: {:.4f}".format(avg_rmsle))

avg_train_pred = np.expm1(avg_train_pred)
avg_train_pred
lasso_test_pred = np.expm1(LassoMd.predict(final_test_df.values))
ENet_test_pred = np.expm1(ENetMd.predict(final_test_df.values))
KRR_test_pred = np.expm1(KRRMd.predict(final_test_df.values))
GBoost_test_pred = np.expm1(GBoostMd.predict(final_test_df.values))

finalMd = (lasso_test_pred + ENet_test_pred + KRR_test_pred + GBoost_test_pred) / 4
finalMd
SEED = 42 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
#kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

def get_oof(model, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    kf = KFold(NFOLDS, shuffle=False, random_state=42).split(final_train_df.values)

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        model.fit(x_tr, y_tr)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# Create our OOF train and test predictions. These base results will be used as new features
ENet_oof_train, ENet_oof_test = get_oof(ENet, final_train_df.values, y_train, final_test_df.values)
KRR_oof_train, KRR_oof_test = get_oof(KRR, final_train_df.values, y_train, final_test_df.values)
XGB_oof_train, XGB_oof_test = get_oof(model_xgb, final_train_df.values, y_train, final_test_df.values)
#lasso_oof_train, lasso_oof_test = get_oof(lasso, x_train, y_train, x_test)
base_predictions_train = pd.DataFrame( {'Kernel Ridge': KRR_oof_train.ravel(),
#                                        'Lasso': lasso_oof_train.ravel(),
                                        'Elastic Net': ENet_oof_train.ravel(),
                                        'XGBoost': XGB_oof_train.ravel()
                                       } )
base_predictions_train.head()
x_train = np.concatenate((KRR_oof_train, ENet_oof_train, XGB_oof_train), axis=1)
x_test = np.concatenate((KRR_oof_test, ENet_oof_test, XGB_oof_test), axis=1)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
Stacked_Model = lasso.fit(x_train, y_train)

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)
rmsle_score = np.sqrt(-cross_val_score(Stacked_Model, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))
print("Staked Lasso Model score: {:.4f} ({:.4f})\n".format(rmsle_score.mean(), rmsle_score.std()))

finalMd = Stacked_Model.predict(x_test)
finalMd = np.expm1(finalMd)
finalMd
from mlxtend.regressor import StackingRegressor

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

stregr = StackingRegressor(regressors=[KRR, model_xgb, ENet], 
                           meta_regressor=lasso)

stregr.fit(final_train_df, y_train)
stregr_train_pred = stregr.predict(final_train_df)

stregr_rmsle = np.sqrt(mean_squared_error(y_train, stregr_train_pred))
print("Stacking Regressor Model RMSLE score: {:.4f}".format(avg_rmsle))
print('Stacking Regressor Variance Score: %.4f' % stregr.score(final_train_df, y_train))

stregr_train_pred = np.expm1(stregr_train_pred)
stregr_train_pred
stregr_test_pred = stregr.predict(final_test_df)
finalMd = np.expm1(stregr_test_pred)
finalMd
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = finalMd
sub.to_csv('submission.csv',index=False)