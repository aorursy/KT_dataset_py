# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
% matplotlib inline
# Read and load Data
train_data = pd.read_csv("../input/train.csv")

# check index of dataframe
train_data.columns

# Describe Stastistics Data
train_data.describe()
#PLot Histogram for 'SalePrice'
sns.distplot(train_data['SalePrice'])
# Skewness and Kurtosis
print("Skewness : %f" % train_data['SalePrice'].skew())
print("Kurtosis : %f" % train_data['SalePrice'].kurt())
# Correlation Matrix(Heat map)
corrmat = train_data.corr()
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(corrmat,cmap = "Greens",vmax = 0.8, square = True)

correlations = train_data.corr()
correlations = correlations["SalePrice"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations.head(10)
#'SalePrice' Correlation Matrix
k = 10
cols = corrmat.nlargest(k , 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale = 1.00)
hm = sns.clustermap(cm , cmap = "Greens",cbar = True,square = True,
                 yticklabels = cols.values, xticklabels = cols.values)
# SCATTER PLOT
sns.set()
expensive = train_data['SalePrice'].quantile(0.9) # 90th percentile
train_data['expensive'] = train_data['SalePrice'].apply(lambda x: 'Expensive' if x > expensive else 'Not-expensive')
sns.pairplot(train_data[["expensive", "SalePrice", "OverallQual", "TotalBsmtSF", "GarageArea","GrLivArea","YearBuilt" ]]
                ,size = 2.5, kind = 'scatter', hue = 'expensive') 
plt.show()

# Missing Data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
# Cleaning Missing Data
train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
# Check there is no missing data
train_data.isnull().sum().max() 
# Standaradising data
SalePrice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][: , np.newaxis]);
low = SalePrice_scaled[SalePrice_scaled[:,0].argsort()][: 10]
high = SalePrice_scaled[SalePrice_scaled[:,0].argsort()][-10 :]
print("low_range:", low)
print("high_range:", high)
#Analyse SalePrice/GrLiveArea
data = pd.concat([train_data['SalePrice'], train_data['GrLivArea']], axis = 1)
data.plot.scatter(x ='GrLivArea', y= 'SalePrice', ylim = (0,800000)); #, alpha=0.3);
# Deleting points
train_data.sort_values(by = 'GrLivArea', ascending  = False)[: 2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
# Histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit = norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'],plot = plt)
#Applying log transformation
train_data['SalePrice'] = np.log(train_data['SalePrice'])
#Transformed histogram and normal probability plot

sns.distplot(train_data['SalePrice'], fit = norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'],plot = plt)
#pairplot top 6 correlated columns with 'SalePrice'
sns.set()
sns.pairplot(train_data[["SalePrice", "OverallQual","TotalBsmtSF","GarageArea","GrLivArea","YearBuilt" ]],
                diag_kind = 'kde' ,size = 2.5) 
plt.show()

y= train_data['SalePrice']

features_name =["OverallQual","TotalBsmtSF", "GarageArea","GrLivArea","YearBuilt"] 
X= train_data[features_name]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

#Lets Normalize the train data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.model_selection import KFold, cross_val_score

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X_train,y_train,
                                   scoring="neg_mean_squared_error", 
                                   cv = kfolds))
    return(rmse)
# Linear Regression
from sklearn import linear_model
from sklearn.metrics import r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print('The accuracy of the Linear Regression is',r2_score(y_test,y_pred))
import xgboost as xgb
xg_reg = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
print('The accuracy of the xgboost is',r2_score(y_test,preds))
print(" rmse : ",cv_rmse(xg_reg).mean(),'\n')
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
params = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

# Initialisation de XGB et GridSearch
xgb = xgb.XGBRegressor(nthread=1, learning_rate=0.02, n_estimators=600) 
grid = GridSearchCV(xgb, params)

# Lancer la gridsearch
grid.fit(X_train, y_train)

# Prendre le modèle avec les meilleurs paramètres
clf = grid.best_estimator_

# Entrainer le meilleur algorithme à notre jeu de données
clf.fit(X_train, y_train)

# Afficher le score R2
print(r2_score(y_train, clf.predict(X_train))) 
print(" rmse : ",cv_rmse(clf).mean(),'\n')
from sklearn import linear_model
from sklearn import svm

classifiers = [
    svm.SVR(),
    linear_model.BayesianRidge(),
    linear_model.ARDRegression(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]
trainingData    = X_train
trainingScores  = y_train
predictionData  = X_test

for item in classifiers:
    print(item)
    clf = item
    clf.fit(trainingData, trainingScores)
    y_predict = clf.predict(predictionData)
    print('Accuracy are',r2_score(y_test,y_predict), '\n')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
model1 =xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
model2 = linear_model.LinearRegression()
model3 = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], 
                                   l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000)


model_1 = model1.fit(X_train,y_train)
model_2 = model2.fit(X_train,y_train)
model_3 = model3.fit(X_train,y_train)


pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)


score1 =  model1.score(X_test,pred1)
score2 =  model2.score(X_test,pred2)
score3 =  model3.score(X_test,pred3)

finalpred=(pred1+pred2+pred3)/3
finalscore = (score1 + score2 +score3)/3
print(" XGB: ",cv_rmse(model_1).mean(),'\n')
print("LR rmse : ",cv_rmse(model_2).mean(),'\n')
print("en rmse : ",cv_rmse(model_3).mean(),'\n')
print('Accuracy are',r2_score(y_test,finalpred), '\n')
print('Final score is',finalscore)
import  lightgbm as lgb
import time
parameters = {
                'max_depth': 1,'min_data_in_leaf': 85,'feature_fraction': 0.80,'bagging_fraction':0.8,'boosting_type':'gbdt',
                'learning_rate': 0.1, 'num_leaves': 30,'subsample': 0.8,'lambda_l2': 4,'objective': 'regression_l2',
                'application':'regression','num_boost_round':5000,'zero_as_missing': True,
                'early_stopping_rounds':100,'metric': 'mae','seed': 2
             }
train_data = lgb.Dataset(X_train, y_train, silent=False)
test_data = lgb.Dataset(X_test, y_test, silent=False)
lgb_model = lgb.train(parameters, train_set = train_data,verbose_eval=500, valid_sets=test_data)
df_test = pd.read_csv("../input/test.csv")
df_test =df_test.reset_index()
df_test[features_name].isnull().sum() #detect missing values
 #complete missing TotalBsmtSF with median
df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].median(), inplace = True)
 #complete missing GarageArea  with median
df_test['GarageArea'].fillna(df_test['GarageArea'].median(), inplace = True)

test_X = df_test[features_name]

pred=np.expm1(lgb_model.predict(test_X))
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

