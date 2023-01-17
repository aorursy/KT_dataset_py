import numpy as np

import pandas as pd

import seaborn as sns

from heapq import nlargest 

import scipy.stats as stats

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns; sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



train.head()
categorical = train.select_dtypes(include=['object']).columns

numerical = train.select_dtypes(include=['int64', 'float64']).columns



print("Categorical columns:")

print(categorical)

print("\nNumerical columns:")

print(numerical)



# Correlation matrix



print("\nCorrelations with SalePrice: ")

corr = train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);



Highest_corr = corr.nlargest(20, 'SalePrice')['SalePrice']

print(Highest_corr)
f1, axes = plt.subplots(1, 3, figsize=(15,5))

f1.subplots_adjust(hspace=0.4, wspace=0.8)

sns.boxplot(x=train['OverallQual'], y=train['SalePrice'],orient='v', ax=axes[0])

sns.boxplot(x=train['GarageCars'], y=train['SalePrice'], orient='v', ax=axes[1])

sns.boxplot(x=train['Fireplaces'], y=train['SalePrice'], orient='v', ax=axes[2])
train = train.drop(train[(train['GarageCars']>3) & (train['SalePrice']<300000)].index).reset_index(drop=True)
f2, axes = plt.subplots(1, 3, figsize=(15,5))

f2.subplots_adjust(hspace=0.4, wspace=0.8)

sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'],orient='v', ax=axes[0])

sns.boxplot(x=train['FullBath'], y=train['SalePrice'], orient='v', ax=axes[1])

sns.boxplot(x=train['HalfBath'], y=train['SalePrice'], orient='v', ax=axes[2])
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index).reset_index(drop=True)

train = train.drop(train[(train['GarageArea']>1100) & (train['SalePrice']<200000)].index).reset_index(drop=True)

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)
sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['2ndFlrSF'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['MasVnrArea'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)
sns.jointplot(x=train['YearBuilt'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['YearRemodAdd'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['GarageYrBlt'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)
sns.jointplot(x=train['BsmtFinSF1'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['LotFrontage'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['WoodDeckSF'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)

sns.jointplot(x=train['OpenPorchSF'], y=train['SalePrice'], kind='reg').annotate(stats.pearsonr)
y_train = train.SalePrice

join_data = pd.concat((train, test), sort=False).reset_index(drop=True)

train_size = len(train)

test_size = len(test)



print("Train set size: ", train_size)

print("Test set size: ", test_size)

print("Train+test set size: ", len(join_data))
missings_count = {col:join_data[col].isnull().sum() for col in join_data.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
def fill_missings(data):

    clean_data = data.copy()



    # Replace missing categorical data with None (strings like Gd = Good, Wd = Wood, etc)

    fill_with_nones = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish',

          'GarageQual','GarageCond','BsmtExposure','BsmtQual','BsmtCond', 'BsmtFinType1', 'BsmtFinType2',

          'MasVnrType']

    for col in fill_with_nones:

        clean_data[col] = clean_data[col].fillna("None")



    # Replace some numeric missings with 0

    fill_with_0 = ['MasVnrArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','GarageArea','GarageCars','TotalBsmtSF','GarageYrBlt']

    for col in fill_with_0:

        clean_data[col] = clean_data[col].fillna(0)



    # Replace LotFrontage with neighborhood mean

    clean_data["LotFrontage"] = clean_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



    # When applies or doubting how to replace, use mode

    clean_data['MSZoning'].fillna(clean_data['MSZoning'].mode()[0])

    clean_data['Electrical'].fillna(clean_data['Electrical'].mode()[0])

    clean_data['Exterior1st'].fillna(clean_data['Exterior1st'].mode()[0])

    clean_data['Exterior2nd'].fillna(clean_data['Exterior2nd'].mode()[0])

    clean_data['KitchenQual'].fillna(clean_data['KitchenQual'].mode()[0])

    clean_data['SaleType'].fillna(clean_data['SaleType'].mode()[0])

    clean_data['Utilities'].fillna(clean_data['Utilities'].mode()[0])



    # Ad-hoc replacement of the Functional column

    clean_data['Functional'].fillna('Typ')

    

    return clean_data
def create_additional_features(all_data):

        

    # Flags

    all_data['has_pool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    all_data['has_2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    all_data['has_garage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    all_data['has_fireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    all_data['has_bsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    

    # Combine features

    all_data['Year_BuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

    all_data['Total_SF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'])

    all_data['Total_Bath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

    all_data['Total_porch_SF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])

    

    return all_data
def categorize_data(data):

    categorized_data = data.copy()

    categorized_data['MSSubClass'] = categorized_data['MSSubClass'].apply(str)

    categorized_data['OverallCond'] = categorized_data['OverallCond'].astype(str)

    categorized_data['YrSold'] = categorized_data['YrSold'].astype(str)

    categorized_data['MoSold'] = categorized_data['MoSold'].astype(str)

    

    return categorized_data
def encode_categories(data):



    all_data = data.copy()

    cols_to_encode = ('PoolQC', 'Alley', 'Fence', 'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

            'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 

            'BsmtExposure', 'GarageFinish', 'LandSlope','LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 

            'OverallCond', 'YrSold', 'MoSold')



    # Process columns and apply LabelEncoder to categorical features

    for c in cols_to_encode:

        lbl = LabelEncoder() 

        lbl.fit(list(all_data[c].values)) 

        all_data[c] = lbl.transform(list(all_data[c].values))

        

    return all_data
# Preprocessing

all_data = fill_missings(join_data)

all_data = create_additional_features(all_data)

all_data = categorize_data(all_data)

all_data = encode_categories(all_data)



# Split again train-test data

X = all_data[:train_size]

X_test_full = all_data[train_size:]



print(len(all_data), len(X), len(X_test_full))



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Deal with additional outliers (from an in depth and manual analysis)

outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])



# Drop columns that would lead to overfitting (too much 0s)

overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

X = X.drop(overfit, axis=1)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.99, test_size=0.01,

                                                                random_state=0)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype not in ['int64', 'float64']]



print("Low cardinality columns: ", low_cardinality_cols)



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



print("Numeric columns: ", numeric_cols)



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
skewed_feats = X_train[numeric_cols].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

y_train = y_train.apply(lambda x: np.log(x))

y_valid = y_valid.apply(lambda x: np.log(x))



# Apply log to all columns with |skewness|>0.8

log_col = []

for col in skewed_feats.index:

    if(abs(skewed_feats[col])>0.8): 

        log_col.append(col)

        X_train[col]=X_train[col].apply(lambda x: np.log(x))

        X_valid[col]=X_valid[col].apply(lambda x: np.log(x))

        X_test[col]=X_test[col].apply(lambda x: np.log(x))
def optimize_xgb(all_data): 

    xgb1 = XGBRegressor()

    parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower

                  'objective':['reg:linear'],

                  'learning_rate': [.02, .01, .0075, .005], #so called `eta` value

                  'max_depth': [5, 6, 7],

                  'min_child_weight': [3, 4, 5],

                  'subsample': [0.7],

                  'colsample_bytree': [0.7],

                  'n_estimators': [2500, 5000]}



    xgb_grid = GridSearchCV(xgb1,

                            parameters,

                            cv = 2,

                            n_jobs = 5,

                            verbose=True)



    xgb_grid.fit(X_train, y_train)



    print(xgb_grid.best_score_)

    print(xgb_grid.best_params_)

    

# optimize_xgb(all_data)
# Define model with best MAE

model = XGBRegressor(colsample_bytree=0.7, learning_rate=.01, max_depth=6, min_child_weight=3, n_estimators=3000, 

                     nthread=1, objective='reg:squarederror', subsample=0.7, random_state=21, 

                     early_stopping_rounds = 10, eval_set=[(X_valid, y_valid)], verbose=False)
# Train and test the model



print("Let's the training begin. Plase wait.")



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('model', model)])

my_pipeline.fit(X_train, y_train)



print("Training finished! Now let's predict test values.")



preds_test = my_pipeline.predict(X_test)

preds_test = [np.exp(pred) for pred in preds_test] 



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index+9,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)



print("Everything finished correctly!")



print(output.SalePrice)
print(len(test), len(X_test_full), len(output.SalePrice))
#Check the distribution of SalePrice

sns.distplot(y)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



print("Skewness: %f" % y.skew())

print("Kurtosis: %f" % y.kurt())
# Correlation matrix



corr = all_data.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);



Highest_corr = corr.nsmallest(40, 'SalePrice')['SalePrice']

print(Highest_corr)
lightgbm = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

lgb_model_full_data = lightgbm.fit(X_train, y_train)
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



xgb_model_full_data = xgboost.fit(X_train, y_train)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def blend_models_predict(X):

    return ((1 * xgb_model_full_data.predict(X)) + \

            (0 * lgb_model_full_data.predict(X)))



print('RMSLE score on train data:')

print(rmsle(y_valid, blend_models_predict(X_valid)))