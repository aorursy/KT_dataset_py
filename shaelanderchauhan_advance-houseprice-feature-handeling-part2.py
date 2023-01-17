# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Refrence Material
import numpy as np # Linear Algebera
import pandas as pd # data preprocessiog

import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew # A data is called as skewed when curve appears distorted or skewed either to the left or 
#to the right, in a statistical distribution

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV ,LassoCV ,RidgeCV # Cv (cross Valadation)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold ,cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import missingno as msno

 


pd.pandas.set_option('display.max_columns',None) # view all columns
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
def missing_percentage(df):
            # get columns with null values in decending order , 
            # [its done like we do for groupby] second cond Include only values which have null exclude Not null eg id
        
    total_missing   = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent_missing = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[df.isnull().sum().sort_values(ascending = False) != 0]
    
    total_present = df.notnull().sum().sort_values(ascending = True)[df.notnull().sum().sort_values(ascending = True)!= len(df)]
    percent_present = round(df.notnull().sum().sort_values(ascending = True)/len(train) *100,2)[df.notnull().sum().sort_values(ascending = True) != len(df) ]
    print('Total features',len(total_missing))
    return pd.concat([total_missing, percent_missing,total_present,percent_present], axis=1, keys=['Total Missing','Percent Missing','Total Present','Percent Present'])
train["SalePrice"] = np.log1p(train["SalePrice"])
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train,'SalePrice')
target_train = train['SalePrice']
df_train = train.drop('SalePrice', axis=1) 
df_test = test
all_data = pd.concat([df_train, df_test]).reset_index(drop=True)
missing_percentage(all_data)
all_data.shape
categorical_features = [feature for feature in all_data.columns if all_data[feature].isnull().sum() >=1 and all_data[feature].dtypes == 'O']
print(categorical_features,'\nLength :  ',len(categorical_features))
def replace_missing_cat(dataset,features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    
    return data

all_data = replace_missing_cat(all_data,categorical_features)
all_data.head()
all_data.shape
numerical_nan = [feature for feature in all_data.columns if all_data[feature].isnull().sum() >=1 and all_data[feature].dtypes != 'O']
for feature in numerical_nan:
    meadian_val = all_data[feature].median()
    
    # create a new feature for refrence and to get more information that where a particular feature had a nan value
    # replacing nan value with 1 and where it had a value replace it with 0
    all_data[feature + '__nan'] = np.where(all_data[feature].isnull(),1,0)
    all_data[feature].fillna(meadian_val,inplace = True)
all_data.shape
all_data.head()
missing_percentage(all_data)
numerical_features = [feature for feature in all_data.columns if all_data[feature].dtype != 'O']
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    all_data[feature] = all_data['YrSold'] - all_data[feature]
all_data.head()
discreate_feature = [feature for feature in numerical_features if len(all_data[feature].unique())< 25 and feature not in year_feature + ['Id'] ] # matlab values not more that 25 unique values 
len(discreate_feature)

continous_feature = [feature for feature in numerical_features if feature not in discreate_feature+year_feature+['Id']]
continous_feature
for feature in continous_feature:
    if 0 in all_data[feature].unique(): # because log (0) will give infinity value so we want to escape that
        pass
    else:
        all_data[feature] = np.log1p(all_data[feature])
#         plt.scatter(train[feature],train['SalePrice'])
#         plt.xlabel(feature)
#         plt.ylabel('SalePrice')
#         plt.show()
       
Categorical_all = [feature for feature in all_data.columns if all_data[feature].dtype == 'O']
for feature in Categorical_all:
    labels_ordered=all_data.groupby([feature])['Id'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    #print(labels_ordered)
    all_data[feature]=all_data[feature].map(labels_ordered)

all_data.head(50)
feature_scale=[feature for feature in all_data.columns if feature not in ['Id']]
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(all_data[feature_scale])
scaler.transform(all_data[feature_scale])
data = pd.concat([all_data[['Id']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(all_data[feature_scale]), columns=feature_scale)],
                    axis=1)
missing_percentage(data)
data.head(10)
X = data.iloc[:len(target_train), :]
X_test = data.iloc[len(target_train):, :]

X.shape, target_train.shape, X_test.shape
X.head()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
# drop Id
X = X.drop(['Id'],axis=1)
feature_select_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 0))
feature_select_model.fit(X,target_train)
feature_select_model.get_support()
selected_features = X.columns[(feature_select_model.get_support())]
selected_features

print('total features: {}'.format((X.shape[1])))
print('selected features: {}'.format(len(selected_features)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(feature_select_model.estimator_.coef_ == 0)))
X = X[selected_features]
X.shape
X.tail(100)
X_test = X_test[selected_features]
X_test.shape
# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, target_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor


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

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())
score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(target_train))
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X,target_train)
print('xgboost')
xgb_model_full_data = xgboost.fit(X, target_train)
print('Svr')
svr_model_full_data = svr.fit(X, target_train)
print('Ridge')
ridge_model_full_data = ridge.fit(X, target_train)
print('RandomForest')
rf_model_full_data = rf.fit(X, target_train)
print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, target_train)
# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.2 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.05 * rf_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))

# Get final precitions from the blended model
blended_score = rmse(target_train, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# Read in sample_submission dataframe
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape


# Append predictions from blended models
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test)))
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)


# Scale predictions
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression2.csv", index=False)
