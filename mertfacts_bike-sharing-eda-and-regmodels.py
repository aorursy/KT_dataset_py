#Basics
import numpy as np 
import pandas as pd 
import calendar
from datetime import datetime

#Plots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#Models
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

#Misc
from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score,train_test_split, GridSearchCV,KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,make_scorer
from sklearn.pipeline import make_pipeline

#Stats
from scipy.stats import skew,norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#Location path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Reading data set as train & test
train=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

train.shape, test.shape
train.head()
test.head()
#Let's look closer to our target, 'count'
sns.set_style("white")
sns.set_color_codes(palette="deep")
f, ax=plt.subplots(figsize=(4,5))

#Lets check the distribution of it
sns.distplot(train['count'],color='b');
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='count')
ax.set(title='Bike Count Distribution')
sns.despine(trim=True, left=True)
plt.show()
#Lets look at Skewness and Kurtosis
print("Skewness: %f" % train['count'].skew())
print("Kurtosis: %f" % train['count'].kurt())
#Finding first numerical features - p1
numeric_dtypes=train.dtypes[train.dtypes != "object"].index
train.columns
#Finding numerical features-2
numeric_dtypes=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric=[]

for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in [ 'count']:
            pass
        else:
            numeric.append(i)

#Visualization more features
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 30))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if(feature=='count'):
        break
    plt.subplot(len(list(numeric)), 2, i)
    sns.scatterplot(x=feature, y='count', hue='count', palette='Blues', data=train)

    plt.xlabel('{}'.format(feature), size=12,labelpad=12.5)
    plt.ylabel('count', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()
#Correlation matrix with features
corr=train.corr()
plt.subplots(figsize=(10,12))
sns.heatmap(corr,vmax=0.8,cmap="BuPu",square=True)
df_vis2=pd.concat([train['count'],train['atemp']],axis=1)
f, ax=plt.subplots(figsize=(18,6))
fig=sns.boxplot(x=train['atemp'],y=train['count'],data=df_vis2)
fig.axis(ymin=0);
plt.xticks(rotation=60);
df_vis=pd.concat([train['count'],train['season']],axis=1)
f, ax=plt.subplots(figsize=(18,6))
fig=sns.boxplot(x=train['season'],y=train['count'],data=df_vis)
fig.axis(ymin=0);
plt.xticks(rotation=60);
df_vis3=pd.concat([train['count'],train['windspeed']],axis=1)
f, ax=plt.subplots(figsize=(18,6))
fig=sns.boxplot(x=train['windspeed'],y=train['count'],data=df_vis3)
fig.axis(ymin=0);
plt.xticks(rotation=60);
# Split features and labels
train_labels = train['count'].reset_index(drop=True)
train_features = train.drop(['count'], axis=1)
test_features = test
test_features
# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape
all_features.head()
#Unnecessary variables omitted
all_features["year"] = [t.year for t in pd.DatetimeIndex(all_features.datetime)]

all_features["date"] = all_features.datetime.apply(lambda x : x.split()[0])
all_features["hour"] = all_features.datetime.apply(lambda x : x.split()[1].split(":")[0])
all_features["weekday"] = all_features.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
all_features["month"] = all_features.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
all_features.drop(["datetime","date","hour"],axis=1,inplace=True)
all_features.head()
all_features.info()
# log(1+x) transform
train["count"] = np.log1p(train["count"])
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['count'] , fit=norm, color="b");

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['count'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution & you can see new-black version
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="count")
ax.set(title="Count distribution")
sns.despine(trim=True, left=True)

plt.show()
# Remove outliers - ?
for col in all_features.dtypes[all_features.dtypes != "object"].index:
    percentiles = all_features[col].quantile([0.01,0.99]).values
    all_features[col] = np.clip(all_features[col], percentiles[0], percentiles[1])
#missing data - why count still come out? - NO :)
Total = all_features.isnull().sum().sort_values(ascending=False)
Percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([Total, Percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#Fetch all numeric features
numeric_dtypes=['int64','float64']
numeric=[]
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)
# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
# Find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(30)
#Let's normalize skewed features
for i in skew_index:
    all_features[i]=boxcox1p(all_features[i],boxcox_normmax(all_features[i]+1))
# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
# Find new version of skewed numerical features- 2 variables stayed same but why ? 
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(30)
#Fetch all numeric features
numeric_dtypes=['int64','float64']
log_features=[]
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        log_features.append(i)
print(log_features)
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res
log_features=['atemp']
all_features = logs(all_features, log_features)
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['weather', 'atemp', ]
all_features = squares(all_features, squared_features)
#Be careful about row numbers should be larger than column numbers - log & squared transformations can cause this situation
all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape
# Remove any duplicated column names
all_features = all_features.loc[:,~all_features.columns.duplicated()]
all_features.shape
all_features.columns
#Is here is a shape difference problem? 

X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape
train_labels.head()
X.head()
X_test.head()
# Setup cross validation folds
kf = KFold(n_splits=10, random_state=42, shuffle=True)
# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=200,
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
                       n_estimators=200,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
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
gbr = GradientBoostingRegressor(n_estimators=200,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200,
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
score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())
score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, train_labels)
print('xgboost')
xgb_model_full_data = xgboost.fit(X, train_labels)
print('svr')
svr_model_full_data = svr.fit(X, train_labels)
print('ridge')
ridge_model_full_data = ridge.fit(X, train_labels)
print('rf')
rf_model_full_data = rf.fit(X, train_labels)
print('gbr')
gbr_model_full_data = gbr.fit(X, train_labels)
# RANDOM SELECTION - Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.15 * rf_model_full_data.predict(X)) + \
            (0.35 * stack_gen_model.predict(np.array(X))))
# get rf rmsle
blended_score_rf = rmsle(train_labels, rf_model_full_data.predict(X))
scores['blended'] = (blended_score_rf, 0)
print('RMSLE score on train data:')
print(blended_score_rf)
# get ridge rmsle
blended_score_ridge = rmsle(train_labels, ridge_model_full_data.predict(X))
scores['blended'] = (blended_score_ridge, 0)
print('RMSLE score on train data:')
print(blended_score_ridge)
# Get final precitions from the blended model
blended_score = rmsle(train_labels, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)
# Get final precitions from the blended model
blended_score_rf = rmsle(train_labels, rf_model_full_data.predict(X))
scores_rf['blended'] = (blended_score_rf, 0)
print('RMSLE score on train data:')
print(blended_score_rf)
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
# X & train_labels came from:
    #X = all_features.iloc[:len(train_labels), :]
    #X_test = all_features.iloc[len(train_labels):, :]
    #X.shape, train_labels.shape, X_test.shape
x_train, x_test, y_train, y_test=train_test_split(X,train_labels,test_size=0.25,random_state=42)
ridge_model=Ridge(alpha=0.1).fit(x_train,y_train)
y_pred=ridge_model.predict(x_train)
rmse=np.sqrt(mean_squared_error(y_train,y_pred))
rmse
def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
rmsle(y_train,y_pred)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
