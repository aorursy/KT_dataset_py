# Data preprocessing and linear algebra
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn import preprocessing
from sklearn import utils

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from graphviz import Source
from IPython.display import SVG, display, HTML
style = "<style>svg{width: 70% !important; height: 60% !important;} </style>"

# Tools for model stecking, cross-validation, error calculation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, RidgeCV
import sklearn.linear_model as linear_model
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
### Data load ###
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
### Data exploration ###
train.head()
test.head()
train.shape
test.shape
# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']
# Now we can drop it
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
# Exploring number of unique observations in Target variable 
len(train.SalePrice.unique())
### NB
# It has 663 unique values. Difference between this competition and well-known Titanic is that previous 
# had only 2 unique values in target variable: 0 and 1.
###
# Draw histogram and boxplot of SalePrice distr

### NB
# In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of 
# numerical data through their quartiles.
###

# If you prefer simple histogram.
# sns.distplot(train['SalePrice']);

f,ax = plt.subplots(1,2,figsize=(16,6))
sns.distplot(train['SalePrice'],fit=norm,ax=ax[0])
sns.boxplot(train['SalePrice'])
plt.show()

# Calculate and print skewness and kurtosis
print("Skewness: {}".format(train['SalePrice'].skew()))
print("Kurtosis: {}".format(train['SalePrice'].kurt()))
print("--------------------------------------")
print(train['SalePrice'].describe())
# Draw probability plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
### Data Visualization ###
# Check correlation between target variable and independent variables
# Draw correlation matrix
plt.figure(figsize=(16,14))
sns.heatmap(train.corr(),annot=False)
# Draw SalePrice' correlation matrix (zoomed heatmap)
#saleprice correlation matrix
corrmat = train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Order variables by correaltion with target variable:
corr = train.select_dtypes(include='number').corr()
plt.figure(figsize=(16,6))
corr['SalePrice'].sort_values(ascending=False)[1:].plot(kind='bar')
plt.tight_layout()
# Analyse 'GrLivArea'
f,ax = plt.subplots(1,2,figsize=(16,4))
sns.boxplot(train['GrLivArea'],ax=ax[0])
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel('GrLiveArea')
plt.ylabel('SalePrice')
plt.show()
# We can see at the bottom right two with extremely large 'GrLivArea' that are of a low price. 
# These values are huge oultliers. Therefore, we can safely delete them.

train.drop(train[train['GrLivArea']>4500].index,axis=0,inplace=True)
# Check again after delete
f,ax = plt.subplots(1,2,figsize=(16,4))
sns.boxplot(train['GrLivArea'],ax=ax[0])
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel('GrLiveArea')
plt.ylabel('SalePrice')
plt.show()
# Analyse 'OverallQual'
f,ax = plt.subplots(1,2,figsize=(16,4))
sns.boxplot(train['OverallQual'],ax=ax[0])
plt.scatter(train['OverallQual'],train['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.show()
# Lets study relationsheeps between variables

# As we mentioned above while our first look at datasets, data in columns has different types: there are
# both numeric and string.

# Find numeric features
numeric_cols = train.select_dtypes(exclude='object').columns
numeric_cols_length = len(numeric_cols)  

fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)

# Skip 'Id' and 'SalePrice' features
for i in range(1,numeric_cols_length-1):
    feature = numeric_cols[i]
    plt.subplot(numeric_cols_length, 3, i)
    sns.scatterplot(x=feature, y='SalePrice', data=train)
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
           
plt.show()
f,ax = plt.subplots(1,3,figsize=(16,4))
sns.scatterplot('GrLivArea','TotRmsAbvGrd',data=train,ax=ax[0])
sns.scatterplot('TotalBsmtSF','1stFlrSF',data=train,ax = ax[1])
sns.scatterplot('GarageCars','GarageArea',data=train,ax = ax[2])
plt.show()
f,ax = plt.subplots(1,3,figsize=(16,4))
sns.scatterplot(x='TotalBsmtSF', y='SalePrice',data=train,ax=ax[0])
sns.scatterplot(x='LotArea', y='SalePrice',data=train,ax=ax[1])
sns.scatterplot(x='OverallQual', y='SalePrice',data=train,ax=ax[2])
plt.show()
# After removing the two outliers, we see that skewness is reduced. 
# But of course still Saleprice is not normally distributed.
f,ax = plt.subplots(1,2,figsize=(16,4))
sns.distplot(train['SalePrice'],ax=ax[0],fit=norm)
stats.probplot(train['SalePrice'],plot=plt)
plt.show()

# Calculate and print skewness and kurtosis
print("Skewness: {}".format(train['SalePrice'].skew()))
print("Kurtosis: {}".format(train['SalePrice'].kurt()))
# We need to normalize the distribution.
# Lets follow a well known approach - Log-transformation [5].

y = np.log1p(train['SalePrice'])

f,ax = plt.subplots(1,2,figsize=(16,4))
sns.distplot(y,fit=norm,ax=ax[0])
stats.probplot(y,plot=plt)
plt.show()

# Calculate and print skewness and kurtosis
print("Skewness: {}".format(y.skew()))
print("Kurtosis: {}".format(y.kurt()))
# Move on. To remove overfitted features lets use iterative approach suggested in [3]
def remove_overfit_features(df,weight):
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > weight:
            overfit.append(i)
    overfit = list(overfit)
    return overfit
# Call a function with our train data as argument
overfitted_features = remove_overfit_features(train,99)
train.drop(overfitted_features,inplace=True,axis=1)
test.drop(overfitted_features,inplace=True,axis=1)
# Lets form training subset 'y' used in ML fit later
train_labels = y
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
train.head()
# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset.

### NB
# This approach used in [2] and [3]. Of course, as some guys mensioned in comments, doing so may cause
# data leakage (as well as applying Box Cox on combined data).
###
# ... so we combine them.
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape
# Check missing data
# Count missing data
total = all_features.isnull().sum().sort_values(ascending=False)
percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
# Visualize missing data
sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
# And in test data
sns.heatmap(test.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
# Then visualise combined data

### NB
# If we do it later after feature engneering, we can check if there are left some missings
###
sns.heatmap(all_features.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
### Data preprocessing ###
# Count amount of missing data before we begin to fill them
print("Total No. of missing value {} before imputation".format(sum(all_features.isnull().sum())))
# Begin fill them according to [3]
# In [3] author uses 'Timber' value to fill 'Electrical'. Since there is only one cell with missing value
# it's not critically important, so lets just left [3] original style

def fill_missing_values():
 
    fillSaleType = all_features[all_features['SaleCondition'] == 'Normal']['SaleType'].mode()[0]
    all_features['SaleType'].fillna(fillSaleType,inplace=True)

    fillElectrical = all_features[all_features['Neighborhood']=='Timber']['Electrical'].mode()[0]
    all_features['Electrical'].fillna(fillElectrical,inplace=True)

    exterior1_neighbor = all_features[all_features['Exterior1st'].isnull()]['Neighborhood'].values[0]
    fillExterior1 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior1st'].fillna(fillExterior1,inplace=True)

    exterior2_neighbor = all_features[all_features['Exterior2nd'].isnull()]['Neighborhood'].values[0]
    fillExterior2 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior2nd'].fillna(fillExterior2,inplace=True)

    bsmtNeigh = all_features[all_features['BsmtFinSF1'].isnull()]['Neighborhood'].values[0]
    fillBsmtFinSf1 = all_features[all_features['Neighborhood'] == bsmtNeigh]['BsmtFinSF1'].mode()[0]
    all_features['BsmtFinSF1'].fillna(fillBsmtFinSf1,inplace=True)

    kitchen_grade = all_features[all_features['KitchenQual'].isnull()]['KitchenAbvGr'].values[0]
    fillKitchenQual = all_features[all_features['KitchenAbvGr'] == kitchen_grade]['KitchenQual'].mode()[0]
    all_features['KitchenQual'].fillna(fillKitchenQual,inplace=True)
    
    # Groupby MSSubClass and fill in missing value by 0 MSZoning of all theMSSuClass
    all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: 
                                                                                        x.fillna(x.mode()[0]))
    
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: 
                                                                                        x.fillna(x.median()))
    
    
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2','PoolQC']:
        all_features[col] = all_features[col].fillna('None')
    
    categorical_cols =  all_features.select_dtypes(include='object').columns
    all_features[categorical_cols] = all_features[categorical_cols].fillna('None')
    
    numeric_cols = all_features.select_dtypes(include='number').columns
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)
    
    all_features['Shed'] = np.where(all_features['MiscFeature']=='Shed', 1, 0)
    
    # GarageYrBlt: missing values are for building which has no Garage. Imputing 0 makes 
    # huge difference with other buildings, imputing mean doesn't make sense since there is no Garage. 
    # We can drop it.
    all_features.drop(['GarageYrBlt','MiscFeature'],inplace=True,axis=1)
    
    all_features['QualitySF'] = all_features['GrLivArea'] * all_features['OverallQual']

# Call a function
fill_missing_values()
# Check after imputing
print("Total No. of missing value {} after imputation".format(sum(all_features.isnull().sum())))
# And finally drop 'PoolQC'
all_features = all_features.drop(['PoolQC',], axis=1)
# Fixing skewed features
# Start with convertion of numeric features to string format
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)
# And continue with filling the skewed features
numeric = all_features.select_dtypes(include='number').columns
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
# Normalize skewed features using BoxCox transform [4]
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']
all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])

all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
def booleanFeatures(columns):
    for col in columns:
        all_features[col+"_bool"] = all_features[col].apply(lambda x: 1 if x > 0 else 0)
booleanFeatures(['GarageArea','TotalBsmtSF','2ndFlrSF','Fireplaces','WoodDeckSF',
                 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'])  
def logs(columns):
    for col in columns:
        all_features[col+"_log"] = np.log(1.01+all_features[col])  

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','MiscVal','YearRemodAdd','TotalSF']

logs(log_features)
def squares(columns):
    for col in columns:
        all_features[col+"_sq"] =  all_features[col] * all_features[col]

squared_features = ['GarageCars_log','YearRemodAdd', 'LotFrontage_log', 
                    'TotalBsmtSF_log', '2ndFlrSF_log', 'GrLivArea_log' ]

squares(squared_features)
# After new featres created we can pass to work with dummy variables.
# We have to turn them into int format since ML algorhytms don't accept strings

# Get dummies
all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.head()
# So now after all we are ready to form our 'X' subset used in ML fit
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
train_labels = train_labels.drop(y.index[outliers])
### Machine Learning ###
# Split dataset into k consecutive folds (without shuffling by default).
kf = KFold(n_splits=12, random_state=42, shuffle=True)

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
# Define basic models
### NB
# We will use RobustScaler method in case of models wwhich are very sensitive to  otliers. Make then more robust.
###
# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)
# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)
# Lasso Regressor
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
# StackingCVRegressor 
stackReg = StackingCVRegressor(regressors=(lasso, svr, ridge, lightgbm),
                                meta_regressor=gbr,
                                use_features_in_secondary=True,random_state=42)
# Base models score
model_score = {}

score = cv_rmse(lightgbm)
lgb_model_full_data = lightgbm.fit(X, train_labels)
print("lightgbm: {:.4f}".format(score.mean()))
model_score['lgb'] = score.mean()
score = cv_rmse(lasso)
lasso_model_full_data = lasso.fit(X, train_labels)
print("lasso: {:.4f}".format(score.mean()))
model_score['lasso'] = score.mean()
score = cv_rmse(svr)
svr_model_full_data = svr.fit(X, train_labels)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
model_score['svr'] = score.mean()
score = cv_rmse(ridge)
ridge_model_full_data = ridge.fit(X, train_labels)
print("ridge: {:.4f}".format(score.mean()))
model_score['ridge'] =  score.mean()
score = cv_rmse(gbr)
gbr_model_full_data = gbr.fit(X, train_labels)
print("gbr: {:.4f}".format(score.mean()))
model_score['gbr'] =  score.mean()
# Stack models
stack_reg_model = stackReg.fit(np.array(X), np.array(train_labels))
# Make blended meta-model
def blended_predictions(X,weight):
    return ((weight[0] * ridge_model_full_data.predict(X)) + \
            (weight[1] * svr_model_full_data.predict(X)) + \
            (weight[2] * gbr_model_full_data.predict(X)) + \
            (weight[3] * lasso_model_full_data.predict(X)) + \
            (weight[4] * lgb_model_full_data.predict(X)) + \
            (weight[5] * stack_reg_model.predict(np.array(X))))
# According to competition rules evaluation metod is RMSLE:
# "Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted 
# value and the logarithm of the observed sales price. (Taking logs means that errors in predicting 
# expensive houses and cheap houses will affect the result equally.)"
### NB
# Root Mean Squared Error (RMSE) and Root Mean Squared Logarithmic Error (RMSLE) 
# both are the techniques to find out the difference between the values predicted by your machine 
# learning model and the actual values.
###
# Blended model predictions
blended_score = rmsle(train_labels, blended_predictions(X,[0.15,0.2,0.1,0.15,0.1,0.3]))
print("blended score: {:.4f}".format(blended_score))
model_score['blended_model'] =  blended_score
### NB
# Weignhts are empirycally choosen according to errors rate by using this logic: the higher weights is given to model with
# less error. Inspired by [3] approach.
###
pd.Series(model_score).sort_values(ascending=True)
# Predictions
# Read sample submission csv
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Add our predictions
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test,[0.15,0.2,0.1,0.15,0.1,0.3])))
sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Write result to csv
submission.to_csv('my_submission.csv', index=False)
# And finally, let’s take a tired but satisfied look at our result
my_submission_check = pd.read_csv('my_submission.csv')
my_submission_check
# Literature
# [1] https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 
# [2] https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/comments
# [3] https://www.kaggle.com/johnwill225/extensive-exploratory-data-analysis/notebook
# [4] https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation
# [5] https://en.wikipedia.org/wiki/Data_transformation_(statistics)
