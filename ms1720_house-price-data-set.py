import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
print('shape of train is: ', train.shape)
print('shape of test is: ', test.shape)
print(train.info())
x = pd.DataFrame(train.dtypes[train.dtypes == int]).reset_index()
x
if train.columns[6] == x['index'][1]:
    print(train[train.columns[1]].isnull().sum())
train.select_dtypes(exclude = ['object']).isnull().sum()
train.select_dtypes(include = ['object']).isnull().sum()
(train.select_dtypes(exclude = ['object']).isnull().sum()).plot(kind = 'bar')
plt.show()
(train.select_dtypes(include = ['object']).isnull().sum()).plot(kind = 'bar')
plt.show()
(train.YearRemodAdd == train.YearBuilt).sum()
(train.YearRemodAdd != train.YearBuilt).sum()
pd.DataFrame([(train.YearRemodAdd == train.YearBuilt).sum(), (train.YearRemodAdd != train.YearBuilt).sum()]).plot(kind ='bar')
dict = pd.DataFrame({'yes' : ((train.YearRemodAdd != train.YearBuilt).sum()),
'no' : ((train.YearRemodAdd == train.YearBuilt).sum())}, index = [0])
dict.unstack().plot(kind = 'bar')
plt.xlabel('Remodel Status')
plt.ylabel('Count')
plt.xticks([0,1],['Yes', 'No'])
plt.show()
train.describe()
(train.isnull().sum().sum())/(train.shape[0]*train.shape[1])
(test.isnull().sum().sum())/(test.shape[0]*test.shape[1])
train.duplicated().sum()
train.MSZoning.value_counts()
train.select_dtypes(include = ['object']).columns[0]
def cat_barplot():
    for n in range(0, 43):
        plt.subplot(11, 4, n+1)
        train.select_dtypes(include = ['object']).iloc[:,n].value_counts().plot(kind = 'bar')
        plt.xlabel(train.select_dtypes(include = ['object']).iloc[:,n].name)
plt.figure(figsize = (25, 60))
cat_barplot()
plt.show()
def num_densityplot():
    for n in range(0, 38):
        plt.subplot(10, 4, n+1)
        train.select_dtypes(exclude = ['object']).iloc[:,n].plot.kde()
        plt.xlabel(train.select_dtypes(exclude = ['object']).iloc[:,n].name)
plt.figure(figsize = (25, 60))
num_densityplot()
plt.show()
train.boxplot("SalePrice", "Neighborhood", figsize=(15, 8), rot = 90)
plt.show()
#train.select_dtypes(exclude = ['object']).corr()
plt.figure(figsize = (12,10))
sns.heatmap(train.select_dtypes(exclude = ['object']).corr())
plt.show()
train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = False)[1:9]
train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = True)[0:4]
#list(pd.DataFrame(train.select_dtypes(exclude = ['object']).corr()['SalePrice'].sort_values( axis = 0, ascending = True)[0:4]).index)
list = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond']
sns.pairplot(train, x_vars = list[0: 4], y_vars = ['SalePrice'], kind = 'reg')
sns.pairplot(train, x_vars = list[4: 8], y_vars = ['SalePrice'], kind = 'reg')
sns.pairplot(train, x_vars = list[8: 12], y_vars = ['SalePrice'], kind = 'reg')
plt.show()
(train.select_dtypes(include = ['object']).columns)[1]
def cat_Sales():
    for n in range(0, 43):
        plt.subplot(11, 4, n+1)
        sns.boxplot(y = train["SalePrice"], x = train.select_dtypes(include = ['object']).iloc[:,n])
        plt.xticks(rotation = 90)
        #plt.show()
plt.figure(figsize=(20, 60))
cat_Sales()
plt.show()
train_num = train.select_dtypes(exclude = ['object'])
train_cat = train.select_dtypes(include = ['object'])

test_num = test.select_dtypes(exclude = ['object'])
test_cat = test.select_dtypes(include = ['object'])
train_num = train_num.fillna(0)
test_num = test_num.fillna(0)
train_cat = train_cat.fillna('Unavailable')
test_cat = test_cat.fillna('Unavailable')
train_cat1 = pd.get_dummies(train_cat)
test_cat1 = pd.get_dummies(test_cat)
train_cat1.head()
from sklearn.preprocessing import StandardScaler
train_num.iloc[:, :-1] = StandardScaler().fit_transform(train_num.iloc[:, :-1])
test_num.iloc[:, :] = StandardScaler().fit_transform(test_num.iloc[:, :])
train_num.SalePrice = np.log(train_num.SalePrice)
train_X = pd.concat([train_num, train_cat1], axis = 1)
test_X = pd.concat([test_num, test_cat1], axis = 1)
X = train_X.drop('SalePrice', axis = 1)
y = train_X.SalePrice
X = X[pd.DataFrame([set(X.columns).intersection(set(test_X.columns))]).T[0]]
test_X = test_X[pd.DataFrame([set(test_X.columns).intersection(set(X.columns))]).T[0]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
y.hist(bins = 50)
plt.show()
simple_model = sm.OLS(y_train, X_train)
simple_results = simple_model.fit()
simple_results.summary()
score_ols = simple_results.rsquared
print(score_ols)
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred_lr = model.predict(X_val)
score_lr = model.score(X_val, y_val)
print(mean_squared_error(y_pred_lr, y_val))
mse_lr = mean_squared_error(y_pred_lr, y_val)
score_lr
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth = 5, random_state = 1)
model = regressor.fit(X_train, y_train)
y_pred_dt = model.predict(X_val)
score_dt = model.score(X_val, y_val)
print(mean_squared_error(y_pred_dt, y_val))
mse_dt = mean_squared_error(y_pred_dt, y_val)
score_dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
list = []
for i in range(1,10): 
    regr = RandomForestRegressor(n_estimators = 15, max_depth=i,
                             random_state=1)
    model = regr.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    list.append(model.score(X_val, y_val))
pd.DataFrame(list)
list = []
for i in range(15,25): 
    regr = RandomForestRegressor(n_estimators=i, max_depth=9,
                             random_state=1)
    model = regr.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    list.append(model.score(X_val, y_val))
list = pd.DataFrame(list)
list
regr = RandomForestRegressor(n_estimators= 20, max_depth=9,
                             random_state=1)
model = regr.fit(X_train, y_train)
y_pred_ranf = model.predict(X_val)
score_ranf = model.score(X_val, y_val)
print(mean_squared_error(y_pred_ranf, y_val))
mse_ranf = mean_squared_error(y_pred_ranf, y_val)
score_ranf
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=30)
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_val)
score_xgb = xgb.score(X_val, y_val)
print(mean_squared_error(y_pred_xgb, y_val))
mse_xgb = mean_squared_error(y_pred_xgb, y_val)
print(score_xgb)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=79, random_state=1)
gbr.fit(X_train,y_train)
y_pred_gbr = gbr.predict(X_val)
score_gbr = gbr.score(X_val, y_val)
print(mean_squared_error(y_pred_gbr, y_val))
mse_gbr = mean_squared_error(y_pred_gbr, y_val)
print(score_gbr)
scores_list = ['score_ols','score_lr','score_dt','score_ranf','score_xgb','score_gbr']
mse = ['NA', mse_lr,mse_dt,mse_ranf,mse_xgb,mse_gbr]
scores = [score_ols,score_lr,score_dt,score_ranf,score_xgb,score_gbr]
score_df = pd.DataFrame([scores_list, scores, mse]).T
score_df.index = score_df[0]
del score_df[0]
score_df
test_prediction_r = gbr.fit(X_train,y_train).predict(test_X)
test_prediction_r = np.exp(test_prediction_r)
plt.subplot(1, 2, 1)
train.SalePrice.hist(bins = 100, figsize = (15,5))
plt.title('Original Data Set Sales Prices')
plt.subplot(1,2,2)
plt.hist(test_prediction_r, bins = 100)
plt.title('Test Data Set Sales Prices')
plt.show()
import h2o
from h2o.automl import H2OAutoML
h2o.init()
# Load data into H2O
df = h2o.H2OFrame(train_X)
df.describe()
y = "SalePrice"
splits = df.split_frame(ratios = [0.8], seed = 1)
train_aml = splits[0]
val = splits[1]
aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "Housing Data Analysis")
aml.train(y = y, training_frame = train_aml, leaderboard_frame = val)
aml.leaderboard.head()
pred = aml.predict(val)
pred.head()
perf = aml.leader.model_performance(val)
perf
#house_test = train_X[pd.DataFrame([set(X.columns).intersection(set(train_X.columns))]).T[0].tolist()]
#X = X[pd.DataFrame([set(X.columns).intersection(set(train_X.columns))]).T[0].tolist()]
np.exp(train_X['SalePrice']).describe()
bins = [30900, 130000, 163000, 214000, 800000]
names = [1, 2, 3, 4]
names1 = ['Cheap','Lower Range', 'Mid-Range', 'Expensive'] 
train_X['SalePrice'] = np.exp(train_X['SalePrice'])
train_X['SalePriceRange'] = train_X['SalePrice']
train_X.SalePriceRange = pd.cut(train_X['SalePriceRange'], bins, labels = names)
X_c = train_X.drop(['SalePrice','SalePriceRange'], axis = 1)
y_c = train_X.SalePriceRange
X_c = X_c[pd.DataFrame([set(X_c.columns).intersection(set(test_X.columns))]).T[0]]
test_X = test_X[pd.DataFrame([set(test_X.columns).intersection(set(X_c.columns))]).T[0]]
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size = 0.2, random_state = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier

from yellowbrick.classifier import ConfusionMatrix
def model_fit(x):
    x.fit(X_train_c, y_train_c)
    y_pred = x.predict(X_val_c)
    model_fit.accuracy = accuracy_score(y_pred, y_val_c)
    print('Accuracy Score',accuracy_score(y_pred, y_val_c))
    print(classification_report(y_pred, y_val_c))
    #print('Confusion Matrix \n',confusion_matrix(y_pred, y_val_c))
    
    classes = names1
    
    model_cm = ConfusionMatrix(
    x, classes = classes,
    label_encoder = {1 : 'Cheap', 2 : 'Lower Range', 3 : 'Mid-Range', 4 : 'Expensive'})
    
    model_cm.fit(X_train_c, y_train_c)
    model_cm.score(X_val_c, y_val_c)
    
    model_cm.poof()    
model_fit(KNeighborsClassifier(n_neighbors = 4))
KNN = model_fit.accuracy
from sklearn.linear_model import LogisticRegression
model_fit(LogisticRegression())
Logistic = model_fit.accuracy
from sklearn.naive_bayes import GaussianNB
model_fit(GaussianNB())
Gaussian = model_fit.accuracy
from sklearn import tree
model_fit(tree.DecisionTreeClassifier())
Tree = model_fit.accuracy
from sklearn.ensemble import RandomForestClassifier
model_fit(RandomForestClassifier(n_estimators = 100, max_depth =10, random_state = 1))
RandomForest = model_fit.accuracy
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'max_depth' : [1, 5, 10],
              'learning_rate' : [0.01, 0.1], 
              'n_estimators' :[5, 10, 15]}

xgb = XGBClassifier()
xgb_cv = RandomizedSearchCV(xgb, param_grid, cv = 5)
xgb_cv.fit(X_train_c, y_train_c)
print(xgb_cv.best_params_)
print(xgb_cv.best_score_)
model_fit(XGBClassifier(max_depth=25, learning_rate=0.1, n_estimators=1800, silent=True, 
                        objective='multi:softprop', booster='gbtree', n_jobs=2, 
                        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, 
                        seed=None, missing=None))
XGBClf = model_fit.accuracy
scores_list_1 = ['KNN','Logistic','Gaussian','Tree','RandomForest','XGBClassifier']
scores_1 = [KNN, Logistic, Gaussian, Tree, RandomForest, XGBClf]
score_df_classification = pd.DataFrame([scores_list_1, scores_1]).T
score_df_classification.index = score_df_classification[0]
del score_df_classification[0]
score_df_classification
test_prediction = XGBClassifier(max_depth=25, learning_rate=0.1, n_estimators=1800, silent=True, 
                        objective='multi:softprop', booster='gbtree', n_jobs=2, 
                        nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, 
                        seed=None, missing=None).fit(X_train_c,y_train_c).predict(test_X)
test_prediction
plt.subplot(1,2,1)
pd.DataFrame(test_prediction)[0].value_counts().plot.bar(figsize = (15,5))
plt.xticks([0, 1, 2, 3], names1)
plt.title('Prediction on Class Distribution on Sale Price Range')

plt.subplot(1,2,2)
train_X.SalePriceRange.value_counts().plot.bar()
plt.xticks([0, 1, 2, 3], names1)
plt.title('Class Distribution on Sale Price Range')
plt.show()