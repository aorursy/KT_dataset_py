# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib.pyplot import figure
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
#figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split


import warnings
# Ignore useless warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Avoid runtime error messages
pd.set_option('display.float_format', lambda x:'%f'%x)

np.random.seed(87)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath = '../input/graduate-admissions/Admission_Predict_Ver1.1.csv'
data = pd.read_csv(filepath)
data.head(5)
data.describe()
#check for missing values:
print(data.isnull().sum())
#Serial No seems like a useless column, lets save it incase we need it and drop it
data_serialNo = data['Serial No.']
data.drop('Serial No.', axis=1,inplace=True)
print("Dropped Serial No.")
print(data.shape)
print(data.head(5))
print(data.dtypes)
# Looks like SOP, LOR, Research are actually categorical variables, Lets convert them to categorical variables
temp = ["SOP","LOR ","Research","University Rating"]
for label in temp:
    data[label] = data[label].astype('object')
print("Converted to Category")
#picking up numerical and categorical features:
numerical_features = data.dtypes[data.dtypes != "object"].index
categorical_features = data.dtypes[data.dtypes == "object"].index

print(numerical_features)
print(categorical_features)
for i,feature in enumerate(numerical_features,1):
    print (i, feature, sep=":")

for j,feature in enumerate(categorical_features,1):
    print (j, feature, sep="-")

#print(np.ceil((len(data.columns))/3))
#plotting all numeric features against GRE Score
fig, ax = plt.subplots(figsize=(20,15))
sns.set(style='darkgrid')
length = len(numerical_features)
for i,feature in enumerate(numerical_features,1):
    plt.subplot(np.ceil(length/2), 2, i) #nrows,ncols,index
    sns.scatterplot(x = feature, y = 'GRE Score',data=data,alpha=0.7,hue="GRE Score", palette ='nipy_spectral_r',linewidth=0.5, edgecolor='white')
    #sns.regplot(x = feature, y="GRE Score",data=data,color='orange')
    plt.ylabel('GRE Score', fontsize=13)
    plt.xlabel(feature, fontsize=13)
plt.show()
#plotting distributions of all numeric features
fig, ax = plt.subplots(figsize=(20,20))
sns.set(style='darkgrid')
length = len(data.columns)
for i,feature in enumerate(numerical_features,1):
    plt.subplot(np.ceil(length/2), 2, i) #nrows,ncols,index
    sns.distplot(data[feature], rug=True,color='green')
    plt.xlabel(feature, fontsize=13)
    plt.title("Distribution of {} variable".format(feature))
plt.show()
#plotting distributions of all categorical features
fig, ax = plt.subplots(figsize=(20,20))
sns.set(style='darkgrid')
length = len(data.columns)
for i,feature in enumerate(categorical_features,1):
    plt.subplot(np.ceil(length/2), 2, i) #nrows,ncols,index
    sns.scatterplot(x=feature,y="CGPA",data=data,alpha=0.7,hue="Research", palette ='nipy_spectral_r',linewidth=0.8, edgecolor='white')
    #sns.regplot(x = feature, y="GRE Score",data=data,color='orange')
    plt.xlabel(feature, fontsize=13)
    plt.title("{} variable".format(feature))
plt.show()
#plotting distributions of all categorical features
fig, ax = plt.subplots(figsize=(20,20))
sns.set(style='darkgrid')
length = len(data.columns)
for i,feature in enumerate(categorical_features,1):
    plt.subplot(np.ceil(length/2), 2, i) #nrows,ncols,index
    sns.scatterplot(x=feature,y="GRE Score",data=data,alpha=0.7,hue="Research", palette ='nipy_spectral_r',linewidth=0.8, edgecolor='white')
    #sns.regplot(x = feature, y="GRE Score",data=data,color='orange')
    plt.xlabel(feature, fontsize=13)
    plt.title("{} variable".format(feature))
plt.show()

#Lets look at Correltion matrix between variables
data_temp = data.astype('float64')
#print(data_temp)
corr_mat = data_temp.corr()
self_corr = np.zeros_like(corr_mat)
self_corr[np.triu_indices_from(self_corr)] = True
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap (corr_mat, square = True,cmap="viridis_r",alpha=0.7,linewidth=0.5,edgecolor='white',fmt=".2f", annot=True,mask=self_corr)
corr_mat['Chance of Admit '].sort_values(ascending=False)
print(data[numerical_features])
#Log transforming all numerical features to avoid any skewness

from scipy.stats import skew
skewed_feats = data[numerical_features].apply(lambda x: skew(x.dropna())) #compute skewness
print(skewed_feats)
skewed_feats = skewed_feats.index
skewed_feats
#Applying log transform to all the skewed variables in the full dataset
data[skewed_feats] = np.log1p(data[skewed_feats])
print("Transformed all Numerical Features")
#creating Dummy variables from the categorical Data
data = pd.get_dummies(data)
print("created dummy variables")
data.head(10)
#Splitting data
#chance of Admit is our target variable
X = data.drop('Chance of Admit ',axis=1)
y = data['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, shuffle=False)
print("X_train: ",X_train.shape,"X_test: ",X_test.shape,"y_train: ",y_train.shape,"y_test: ",y_test.shape)
#Import Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

#setup KFold with 10 splits
kf = KFold(n_splits=10,random_state=23, shuffle=True)
#Define our RMSE function: Root Mean Squared Error
def rmse_cross_val (model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv = kf))
    return (rmse)

#we have already scaled back our outliers, even then lets define our RMSLE function: Root Mean Squared log Error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y_train,y_pred))
print("Defined RMSE function")
#Linear Regression
lreg = LinearRegression()

#LightGBM Regressor
lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=6,
                         learning_rate=0.01,
                         n_estimators=7000,
                         max_bin=200,
                         bagging_fractions=0.8,
                         bagging_freq=4,
                         feature_fractions=0.2,
                         feature_fraction_seed=8,
                         min_sum_hessian_in_leaf=11,
                         verbose=-1,
                         random_state=23)
#XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective="reg:squarederror",
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=22,
                       reg_alpha=0.00006,
                       random_state=23)
                         
    
#RidgeRegressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(),RidgeCV(alphas=ridge_alphas, cv=kf))


#supportvectorregressor
svr = make_pipeline(RobustScaler(),SVR(C=20,epsilon=0.008,gamma=0.0003))

#gradientboostregressor
gbr = GradientBoostingRegressor(n_estimators = 6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=23)
#random forest regressor
rf = RandomForestRegressor(n_estimators=1200,
                           max_depth=5,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features=None,
                           oob_score=True,
                           random_state=23)

#Stack all Models

stack_gen = StackingCVRegressor(regressors=(lreg,xgboost,lightgbm,svr,ridge,gbr,rf),
                                meta_regressor=svr,
                                use_features_in_secondary=True)
scores={}
score = rmse_cross_val(lreg)
print("Liner regression: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['lreg'] = (score.mean(),score.std())
score = rmse_cross_val(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['lgb'] = (score.mean(),score.std())
score = rmse_cross_val(xgboost)
print("XGBoost: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['xgboost'] = (score.mean(),score.std())
score = rmse_cross_val(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['svr'] = (score.mean(),score.std())
score = rmse_cross_val(ridge)
print("Ridge: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['ridge'] = (score.mean(),score.std())
score = rmse_cross_val(rf)
print("RandomForest: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['rf'] = (score.mean(),score.std())
score = rmse_cross_val(gbr)
print("Gradientboostingregressor: {:.4f} ({:.4f})".format(score.mean(),score.std()))
scores['gbr'] = (score.mean(),score.std())

print("Linear Regression")
lreg_model_full_data=lreg.fit(X_train,y_train)

print("lightgbm")
lgbm_model_full_data=lightgbm.fit(X_train,y_train)

print("xgboost")
xgbm_model_full_data=xgboost.fit(X_train,y_train)

print("svr")
svr_model_full_data=svr.fit(X_train,y_train)

print("Ridge")
ridge_model_full_data=ridge.fit(X_train,y_train)

print("RandomForest")
rf_model_full_data=rf.fit(X_train,y_train)

print("gradientBoosting")
gbr_model_full_data=gbr.fit(X_train,y_train)

print("stacked models")
stack_gen_model_full_data=stack_gen.fit(X_train,y_train)
def blended_predictions(X_train):
    return ((0.1 * ridge_model_full_data.predict(X_train))+
            (0.1 * svr_model_full_data.predict(X_train))+
            (0.1 * gbr_model_full_data.predict(X_train))+
            (0.1 * xgbm_model_full_data.predict(X_train))+
            (0.1 * lgbm_model_full_data.predict(X_train))+
            (0.1 * rf_model_full_data.predict(X_train))+
            (0.1 * lreg_model_full_data.predict(X_train))+
            (0.30 * stack_gen_model_full_data.predict(np.array(X_train))))
#Getting final predictions
blended_score = rmsle(y, blended_predictions(X_train))
scores['blended'] = (blended_score,0)
print("RMSLE Score on training data:")
print(blended_score)
#Identifying best model
f, ax = plt.subplots(figsize=(20, 15))
ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'],color='green')
for i, score in enumerate(scores.values()):
    ax.text(i,score[0]-0.001,'{:.6f}'.format(score[0]),horizontalalignment='left',size='large',color='purple',weight='semibold')
plt.ylabel('Score(RMSE)',size=20)
plt.xlabel('Model',size=20)
plt.title("Scores of Models",size=20)
plt.show()

#Generating final predictions
blendPred = blended_predictions(X_test)
submission = pd.DataFrame()
submission['actual'] = y_test
submission['predictions'] = blendPred
submission['residuals'] = submission['actual']-submission['predictions']
submission['rmsescore']= np.absolute(submission['residuals']/submission['actual']*100)
submission = submission.reset_index()
submission.drop('index',axis=1,inplace=True)
pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '% 2f' % x)
submission.sort_values(by = ['rmsescore'])