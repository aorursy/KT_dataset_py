import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ###################################################################
# data loding
data = pd.read_csv('../input/diamonds.csv',index_col=0)
# ###################################################################
# data check
data.head(10)
data.info()
data.isnull().sum()
# ###################################################################
# Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='RdBu_r', annot=True, linewidths=0.5, center=0)
# 1. carat
sns.distplot(data['carat'])
data['carat'].ix[np.where(data['carat'] < 2)].count()
#   1-1. 대부분의 다이아몬드(96%)는 (0~2) 사이의 Carat값을 가진다.
bins = [0, 1, 2, 3, 4, 10]
label = ['Carat0','Carat1','Carat2','Carat3','Carat4']
data['Categoricalcarat'] = pd.cut(data['carat'], bins, labels=label)
sns.barplot(x='Categoricalcarat', y='price', data=data)
#   1-2. Carat이 커질수록 평균가격 또한 증가하는 추세를 보임.
#   1-3. Carat의 증가폭은 (0~1) - (1~2), (1~2) - (2~3)일때 가장 크게 증가한다. 대략 $5000 증가한다.
sns.lmplot('carat','price', data=data)
#   1-4. Carat has a significant positive correlation.
# 2. Create Feature volume = x * y * z
#   2-1. Outlier Delete for x,y,z = 0
#        x,y,z can not have a value of 0 and it is judged to be an invalid data input
print(np.where(data['x'] == 0))
print(np.where(data['y'] == 0))
print(np.where(data['z'] == 0))
data = data.drop(data.index[np.where(data['x'] == 0)])
data = data.drop(data.index[np.where(data['y'] == 0)])
data = data.drop(data.index[np.where(data['z'] == 0)])
data = data.reset_index(drop=True)
# Create Volume
data['volume'] = data['x'] * data['y'] * data['z']
#   2-2. Volume Outlier Delete
sns.distplot(data['volume'])
data['carat'].ix[np.where(data['volume'] > 200)].count()
#   2-3. 다이아몬드의 대부분의 volume은 200이하이며, 200이상의 volume을 가지는 다이아몬드는 전체 다이아몬드의 0.15% 이다.
# In the case of ix 24058, the y value is 58 (y average value 5.73), which is very large and is classified as outliers.
np.where(data['volume'] > 1000)
data.ix[np.where(data['volume'] > 1000)]
print('Diamiond Y Mean : ',data['y'].mean())
data = data.drop(data.index[np.where(data['volume'] >= 1000)])
data = data.reset_index(drop=True)
#   2-4. Categorize the all Volume value and See the trend
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000]
label = ['Vol0','Vol100','Vol200','Vol300','Vol400','Vol500','Vol600','Vol700','Vol800']
data['Categoricalvolume'] = pd.cut(data['volume'], bins, labels=label)
sns.barplot(x='Categoricalvolume', y='price', data=data)
#   2-5. The larger the volume, the increase the average price.
#   2-6. However, the average price of the (800 ~) volume is very low, which is likely to be the wrong data.
data.ix[np.where(data['volume'] >= 800)]
#   2-7. y,z values are excessively larger than the average, and Compared with carat, you can see that the y and z values are entered incorrectly.
print('Diamiond Y Mean : ',data['y'].mean())
print('Diamiond Z Mean : ',data['z'].mean())
data = data.drop(data.index[np.where(data['volume'] >= 800)])
data = data.reset_index(drop=True)
sns.lmplot('volume','price', data=data)
#   2-8. Volume has a significant positive correlation.
# Encoding cut, color, clarity
# 3. cut : Fair, Good, Very Good, Premium, Ideal
sns.countplot(data['cut'], order=['Fair','Good','Very Good','Premium','Ideal'])
data['carat'].ix[np.where((data['cut'] == 'Premium') | (data['cut'] == 'Ideal'))].count()
#   3-1. Many Diamond  are in higher grade (Premium, Ideal)
#   3-2. Premium and Ideal grades, account for 65% of all diamonds.
sns.barplot(x='cut', y='price', order=['Fair','Good','Very Good','Premium','Ideal'], data=data)
#   3-3. The average price is similar for all grades.
#   3-4. In particular, the highest grade Ideal, has the lowest average price among all grades.
#   3-5. From observations 3-4, we can be deduced that cut has a small impact on average prices.
# Encoding cut
data.loc[data['cut'] == 'Fair', 'cut'] = 1
data.loc[data['cut'] == 'Good', 'cut'] = 2
data.loc[data['cut'] == 'Very Good', 'cut'] = 3
data.loc[data['cut'] == 'Premium', 'cut'] = 4
data.loc[data['cut'] == 'Ideal', 'cut'] = 5
# 4. color : J(Worst) I H G F E D(Best)
sns.countplot(data['color'], order=['J','I','H','G','F','E','D'])
#   4-1. Many Diamond  are in middle grade (G, F)
sns.barplot(x='color', y='price', order=['J','I','H','G','F','E','D'], data=data)
#   4-2. For color, the higher the grade, the lower the average price.
print('Color J Mean Price : ', data['price'].ix[np.where(data['color'] == 'J')].mean())
print('Color D Mean Price : ', data['price'].ix[np.where(data['color'] == 'D')].mean())
#   4-3. Especially, the average price for the lowest grade J($ 5323) is 1.68 times higher than the average price for the highest grade D($ 3168).
#   4-4. In observations 4-3, the average price is more influenced by other features than Color.
# Encoding color
data.loc[data['color'] == 'J', 'color'] = 1
data.loc[data['color'] == 'I', 'color'] = 2
data.loc[data['color'] == 'H', 'color'] = 3
data.loc[data['color'] == 'G', 'color'] = 4
data.loc[data['color'] == 'F', 'color'] = 5
data.loc[data['color'] == 'E', 'color'] = 6
data.loc[data['color'] == 'D', 'color'] = 7
# 5. clarity : I1(Worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(Best)
sns.countplot(data['clarity'], order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
#   5-1. Many Diamond  are in lower grade (SI2, SI1, VS2)
sns.barplot(x='clarity', y='price', order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], data=data)
#   5-2. Overall, the higher the grade, the lower the average price.
print('Clatity SI2 Mean Price : ', data['price'].ix[np.where(data['clarity'] == 'SI2')].mean())
print('Clatity IF Mean Price : ', data['price'].ix[np.where(data['clarity'] == 'IF')].mean())
#   5-3. the average price for the lower grade SI2($ 5059) is 1.78 times higher than the average price for the highest grade IF($ 2864) too.
# Encoding clarity
data.loc[data['clarity'] == 'I1', 'clarity'] = 1
data.loc[data['clarity'] == 'SI2', 'clarity'] = 2
data.loc[data['clarity'] == 'SI1', 'clarity'] = 3
data.loc[data['clarity'] == 'VS2', 'clarity'] = 4
data.loc[data['clarity'] == 'VS1', 'clarity'] = 5
data.loc[data['clarity'] == 'VVS2', 'clarity'] = 6
data.loc[data['clarity'] == 'VVS1', 'clarity'] = 7
data.loc[data['clarity'] == 'IF', 'clarity'] = 8
# 6. depth
sns.distplot(data['depth'])
data['depth'].ix[np.where((data['depth'] > 60) & (data['depth'] < 65))].count()
#   6-1. Most diamonds(88%) have depth values between (60 ~ 65)
bins = [0, 50, 55, 60, 65, 70, 75, 80]
label = ['depth0','depth50','depth55','depth60','depth65','depth70','depth75']
data['Categoricaldepth'] = pd.cut(data['depth'], bins, labels=label)
sns.barplot(x='Categoricaldepth', y='price', data=data)
#   6-2. depth has a similar average price as a whole, but the average price is lowered by more than half in a certain interval (50 ~ 55) and (70 ~ 75
sns.lmplot('depth','price', data=data)
#   6-3. depth has a weak negative correlation.
# 7. table
sns.distplot(data['table'])
data['table'].ix[np.where((data['table'] > 55) & (data['table'] < 60))].count()
#   7-1. Most diamonds(84%) have depth values between (60 ~ 65)
bins = [0, 50, 55, 60, 65, 70, 75, 100]
label = ['table0','table50','table55','table60','table65','table70','table75']
data['Categoricaltable'] = pd.cut(data['table'], bins, labels=label)
sns.barplot(x='Categoricaltable', y='price', data=data)
#   7-2. The table also has a similar average price as a whole, but the average price is lowered by more than half in a certain interval (70 ~ 75).
#   7-3. (75 ~) interval has a large variance, but the number of diamonds in the interval is very small, so there is no big influence.

sns.lmplot('table','price', data=data)
#   7-4. table has a weak positive correlation.
# Data Cleaning
data = data.drop(['x','y','z'], axis=1)
data = data.drop(['Categoricalvolume','Categoricalcarat','Categoricaldepth','Categoricaltable'], axis=1)
data.head(10)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='RdBu_r', annot=True, linewidths=0.5, center=0)
# ###################################################################
# Explanatory variable X, Response variable y
X = data.drop(['price'], axis=1)
y = data['price']
# ###################################################################
# Scale Standardization
# Scale the data to be between -1 and 1
sc = StandardScaler()
X = sc.fit_transform(X)
# ###################################################################
# Data Split
seed = 940224
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
# ###################################################################
# Visualization, Report Ver.2
# Functions for visualizing results
def pred_vis(name, y_test_vis, y_pred_vis):
    if y_test_vis.shape[0] > 200:
        y_test_vis = y_test_vis[:200]
        y_pred_vis = y_pred_vis[:200]
        
    y_test_m_vis = y_test_vis.as_matrix()
    plt.figure(figsize=(12,5))
    plt.title('%s Prediction' %name)
    plt.plot(y_test_m_vis, c='steelblue', alpha=1)
    plt.plot(y_pred_vis, c='darkorange', alpha=2)
    legend_list = ['y_test', 'y_pred']
    plt.xlabel('Var')
    plt.ylabel('Output')
    plt.legend(legend_list, loc=1, fontsize='10')
    plt.grid(True)
    plt.show()

    
# GridSearchCV, RandomizedSearchCV Report Function
# -> by. scikit-learn.org "Comparing randomized search and grid search for hyperparameter estimation"
def report(results, n_top=3):
    lcount = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            if lcount > 2:
                break
            lcount += 1


def model_scores(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    global X_test
    adj_r2 = 1 - (1 - r2)*float(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

    print('MSE    : %0.2f ' % mse)
    print('MAE    : %0.2f ' % mae)
    print('RMSE   : %0.2f ' % rmse)
    print('R2     : %0.2f ' % r2)
    print('Adj_R2 : %0.2f ' % adj_r2)
    return {'mse':[mse], 'rmse':[rmse], 'r2':[r2]}


def result_vis(r2_results_vis, names_vis):
    fig =plt.figure(figsize=(6,6))
    fig.suptitle('Algorithm Comparison - R2')
    ax = fig.add_subplot(111)
    plt.barh(np.arange(len(names_vis)), sum(r2_results_vis, []), align="center")
    ax.set_yticks(np.arange(len(names_vis)))
    ax.set_yticklabels(names_vis)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.grid(True)
models = []
models.append(('RandomForest ', RandomForestRegressor(n_estimators=1000)))
models.append(('AdaBoost     ', AdaBoostRegressor(n_estimators=1000)))
models.append(('GBM          ', GradientBoostingRegressor(n_estimators=1000)))
models.append(('XGBoost      ', xgb.XGBRegressor(booster='gbtree',objective='reg:linear',n_estimators=1000)))
#Hyperparameter Grid Ver.1
# param_grid = {
#        'RandomForest'  : {'n_estimators'       : st.randint(500, 1000),
#                           'max_features'       : ['auto','sqrt','log2'],
#                           'max_depth'          : st.randint(1, 20),
#                           'min_samples_split'  : st.randint(2, 50),    
#                           'min_samples_leaf'   : st.randint(1, 50),
#                           'criterion'          : ['mse', 'mae']},
#                           
#        'Adaboost'      : {'n_estimators'       : st.randint(500, 1000),
#                           'learning_rate'      : st.uniform(0.001, 0.1),
#                           'loss'               : ['linear', 'square', 'exponential']},
#                           
#        'GBM'           : {'n_estimators'       : st.randint(1000, 5000),
#                           'max_depth'          : st.randint(1, 20),
#                           'learning_rate'      : st.uniform(0.001, 0.1),
#                           'min_samples_split'  : st.randint(2, 50),
#                           'min_samples_leaf'   : st.randint(2, 50)},
#
#        'XGB'           : {'n_estimators'       : st.randint(1000, 5000),
#                           'max_depth'          : st.randint(1, 20),
#                           'learning_rate'      : st.uniform(0.001, 0.1),
#                           'colsample_bytree'   : st.beta(10, 1),
#                           'subsample'          : st.beta(10, 1),
#                           'gamma'              : st.uniform(0, 10),
#                           'min_child_weight'   : st.expon(0, 10)}
#}

# MODELS - param_grid
# models = []
# models.append(('RandomForest ', RandomizedSearchCV(RandomForestRegressor(), param_grid['RandomForest'], scoring='r2', cv=Kfold, n_jobs=-1, n_iter=100, random_state=seed)))
# models.append(('AdaBoost     ', RandomizedSearchCV(AdaBoostRegressor(), param_grid['AdaBoost'], scoring='r2', cv=Kfold, n_jobs=-1, n_iter=100, random_state=seed)))
# models.append(('GBM          ', RandomizedSearchCV(GradientBoostingRegressor(), param_grid['GBM'], scoring='r2', cv=Kfold, n_jobs=-1, n_iter=100, random_state=seed)))
# models.append(('XGBoost      ', RandomizedSearchCV(xgb.XGBRegressor(booster='gbtree',objective='reg:linear'), param_grid['XGB'], scoring='r2', cv=Kfold, n_jobs=-1, n_iter=100, random_state=seed)))
r2_results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print('')
    print('## %s ##################################' % name)
    print('Test score : %.4f' % model.score(X_test, y_test))
    results = model_scores(y_test, y_pred)

    pred_vis(name, y_test, y_pred)
    
    r2_results.append(results['r2'])
    names.append(name.replace(' ', ''))
result_vis(r2_results, names)
sns.factorplot('name','r2', data=pd.DataFrame({'name':names, 'r2':[round(i, 4) for i in sum(r2_results, [])]}), size=6)