# Load python libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, learning_curve, validation_curve, GridSearchCV
from skopt import BayesSearchCV
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import sys
sys.path.append('/Users/minjielu/anaconda3/envs/python/lib/python3.5/site-packages')

import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')
%%time
# Load data
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# take a look at train and test data
print('train data size: {}*{}'.format(data.shape[0],data.shape[1]))
print('test data size: {}*{}'.format(test.shape[0],test.shape[1]))
data.sample(10)
test.sample(10)
# Take out magic features discovered by olivier
magic_features = ['f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1']

features = [f for f in data.columns if f not in ['target', 'ID']]
magic_features_loc = [features.index(x) for x in magic_features]
%%time
# Statistic features and magic features are used as inputs for machine learning algortihms
features = [f for f in data.columns if f not in ['target', 'ID']]

def to_hist_func(row):
    count = row[row != 0].shape[0]
    hist = []
    hist.extend(row[magic_features_loc]) # Add Santander 46 magic features.
    # Replace 0 values with null, this procedure seems to improve the performance of regressions.
    # hist[hist == 0] = np.nan
    # When statistic features are calculated, zero values are removed.
    row = row[row != 0]
    hist.append(np.min(row)) # Add the minimum
    # hist.append(np.percentile(row,10)) # Add percentiles
    # hist.append(np.percentile(row,20))
    hist.append(np.percentile(row,25))
    # hist.append(np.percentile(row,30))
    # hist.append(np.percentile(row,40))
    hist.append(np.percentile(row,50))
    # hist.append(np.percentile(row,60))
    # hist.append(np.percentile(row,70))
    hist.append(np.percentile(row,75))
    # hist.append(np.percentile(row,80))
    # hist.append(np.percentile(row,90))
    hist.append(np.max(row)) # Add the maximum
    # hist.append(np.mean(row)) # Add the mean
    # hist.append(np.median(row)) # Add the median
    # hist.append(np.sum(row)) # Add sum
    # Add fine histogram.
    # for x in np.arange(8,17,0.2):
        # hist.append(row[(row < x+1) & (row >= x)].shape[0])
    # Add coarse histogram.
    # for x in np.arange(8,17,1):
        # hist.append(row[(row < x+2) & (row >= x)].shape[0])
    # hist.append(row[(row < 23) & (row >= 20)].shape[0])
    hist.append(count)  # Add the number of nonzero features
    hist.append(skew(row)) # Add the skewness
    hist.append(kurtosis(row)) # Add the kurtosis
    '''
    # One observation is that there are lots of repeated values
    # Therefore, statistic features are also extracted after these repeated values are removed
    row_unique = np.unique(row)
    hist.append(np.min(row_unique))
    hist.append(np.percentile(row_unique,10))
    hist.append(np.percentile(row_unique,20))
    hist.append(np.percentile(row_unique,25))
    hist.append(np.percentile(row_unique,30))
    hist.append(np.percentile(row_unique,40))
    hist.append(np.percentile(row_unique,50))
    hist.append(np.percentile(row_unique,60))
    hist.append(np.percentile(row_unique,70))
    hist.append(np.percentile(row_unique,75))
    hist.append(np.percentile(row_unique,80))
    hist.append(np.percentile(row_unique,90))
    hist.append(np.max(row_unique))
    for x in np.arange(8,17,0.2):
        hist.append(row_unique[(row_unique < x+1) & (row_unique >= x)].shape[0])
    for x in np.arange(8,17,1):
        hist.append(row_unique[(row_unique < x+2) & (row_unique >= x)].shape[0])
    hist.append(row_unique[(row_unique < 23) & (row_unique >= 20)].shape[0])
    hist.append(len(row_unique)) # Add the number of unique values.
    hist.append(skew(row_unique))
    hist.append(kurtosis(row_unique))
    '''
    pdrow = pd.Series(row)
    # Add the three most frequent values. If there is not enough unique values, zeroes or nans are used instead
    unique_values = pdrow.value_counts()
    hist.append(unique_values.index[0])
    if unique_values.shape[0] == 1:
        hist.extend([0,0]) 
        # hist.extend([np.nan,np.nan])
        return hist
    hist.append(unique_values.index[1])
    if unique_values.shape[0] == 2:
        hist.extend([0])
        # hist.extend([np.nan])
        return hist
    hist.append(unique_values.index[2])
    return hist


# Generate statistic features for train data
hist_data = np.apply_along_axis(
    func1d=to_hist_func, 
    axis=1, 
    arr=(np.log1p(data[features])).astype(float)) 
%%time
# Generate statistic features for test data
hist_test = np.apply_along_axis(
    func1d=to_hist_func, 
    axis=1, 
    arr=(np.log1p(test[features])).astype(float))
# Define a score function that returns mean squared log error, since the built-in mean_squared_log_error of sklearn somehow doesn't work
def my_own_score(ground_truth,predictions):
    return (mean_squared_error(ground_truth,predictions) ** .5)
    
score = make_scorer(my_own_score,greater_is_better=False)
def plot_learning_curve(regressor, title, x, y, score):
    train_sizes, train_scores, test_scores = learning_curve(regressor, x, y, scoring=score, train_sizes = np.linspace(0.1,1.0,7), cv = 5)
    plt.figure()
    plt.title(title)
    plt.xlabel('Number of samples')
    plt.ylabel('score')
    plt.grid()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt
# Define regressors using the best parameters found by grid search or Bayesian search
myET = ExtraTreesRegressor(n_estimators=1000, max_features=.8, max_depth=10, min_samples_leaf=5, random_state=3, n_jobs=-1)
myKNN = KNeighborsRegressor(n_neighbors=20, weights='distance', algorithm='kd_tree', n_jobs=-1)
myDT = DecisionTreeRegressor(criterion='friedman_mse', splitter='random', min_samples_leaf=15)
myRF = RandomForestRegressor(n_estimators=2000, max_features=.3, max_depth=10, min_samples_leaf=5,random_state=3,n_jobs=-1)
myADBoost = AdaBoostRegressor(n_estimators=100, learning_rate=0.05, loss='linear', random_state=3)
# mylgb = lgb.LGBMRegressor(num_leaves=58,subsample=.4,colsample_bytree=.4,max_depth=10,learning_rate=0.05,objective='regression',random_state=3,boosting_type='gbdt',seed=3,min_child_weight=np.power(10,-0.1477),reg_lambda=np.power(10,1.7570),reg_alpha=np.power(10,-2.2887),min_split_gain=np.power(10,-2.5988))
# For data with leak items added
# mylgb = lgb.LGBMRegressor(objective='regression', random_state=3, learning_rate=0.0353, max_bin=386, max_depth=19, min_child_samples=19,min_child_weight=4, min_split_gain=0.464, n_estimators=124, num_leaves=51, reg_alpha=0.000246, reg_lambda=0.0001696, subsample_freq=4)
# myxgb = xgb.XGBRegressor(objective='reg:linear',booster='gbtree',seed=3,colsample_bylevel=0.477,colsample_bytree=0.1,gamma=0.119,learning_rate=0.0602,max_depth=9,min_child_weight=100,n_estimators=150,reg_lambda=1e-9,subsample=1.0)
# For data with train leak removed
# mylgb = lgb.LGBMRegressor(objective='regression', random_state=3, learning_rate=0.0237, max_bin=100, max_depth=50, min_child_samples=7,min_child_weight=2, min_split_gain=0.001, n_estimators=150, num_leaves=10, reg_alpha=1e-9, reg_lambda=3.12e-8, subsample_freq=0)
# myxgb = xgb.XGBRegressor(objective='reg:linear',booster='gbtree',seed=3,colsample_bylevel=0.117,colsample_bytree=0.524,gamma=0.00885,learning_rate=0.147,max_depth=6,min_child_weight=76,n_estimators=86,reg_lambda=37.56,subsample=0.948)
mylgb = lgb.LGBMRegressor(num_leaves=60,subsample=.4,colsample_bytree=.6,max_depth=2,learning_rate=0.1,objective='regression',random_state=3,boosting_type='gbdt',seed=3,min_child_weight=np.power(10,-0.1477),reg_lambda=np.power(10,1.7570),reg_alpha=np.power(10,-2.2887),min_split_gain=np.power(10,-2.5988))
myxgb = xgb.XGBRegressor(objective='reg:linear',booster='gbtree',seed=3,colsample_bylevel=0.44,colsample_bytree=0.53,gamma=1.98e-3,learning_rate=0.0355,max_depth=44,min_child_weight=79,n_estimators=144,reg_lambda=0.0355,subsample=1.0)

# Plot learning curves for all regressors
plot_learning_curve(myET,'ExtraTrees regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(myKNN,'KNN regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(myDT,'DecisionTree regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(myRF,'RandomForest regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(myADBoost,'AdaBoost regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(mylgb,'LightGBM regressor learning curve',hist_data,np.log1p(data['target']),score)
plot_learning_curve(myxgb,'XGBoost regressor learning curve',hist_data,np.log1p(data['target']),score)
# Generate meta features using selected regressors
def generate_meta_features(regressor,x,y,z):
    folds = KFold(n_splits=5,shuffle=True,random_state=1)
    oof_preds = np.zeros(x.shape[0])
    test_preds = np.zeros(z.shape[0])
    
    for n_fold, (trn_, val_) in enumerate(folds.split(x)):
        regressor.fit(x[trn_],y[trn_])
        oof_preds[val_] = regressor.predict(hist_data[val_])
        test_preds += regressor.predict(z) / folds.n_splits
        
    return oof_preds,test_preds
ET_meta,ET_test_meta = generate_meta_features(myET,hist_data,np.log1p(data['target']),hist_test)
KNN_meta,KNN_test_meta = generate_meta_features(myKNN,hist_data,np.log1p(data['target']),hist_test)
DT_meta,DT_test_meta = generate_meta_features(myDT,hist_data,np.log1p(data['target']),hist_test)
RF_meta,RF_test_meta = generate_meta_features(myRF,hist_data,np.log1p(data['target']),hist_test)
ADBoost_meta,ADBoost_test_meta = generate_meta_features(myADBoost,hist_data,np.log1p(data['target']),hist_test)
LGB_meta,LGB_test_meta = generate_meta_features(mylgb,hist_data,np.log1p(data['target']),hist_test)
XGB_meta,XGB_test_meta = generate_meta_features(myxgb,hist_data,np.log1p(data['target']),hist_test)
ET_meta = pd.Series(ET_meta,name='ET_meta')
KNN_meta = pd.Series(KNN_meta,name='KNN_meta')
DT_meta = pd.Series(DT_meta,name='DT_meta')
RF_meta = pd.Series(RF_meta,name='RF_meta')
ADBoost_meta = pd.Series(ADBoost_meta,name='ADBoost_meta')
LGB_meta = pd.Series(LGB_meta,name='LGB_meta')
XGB_meta = pd.Series(XGB_meta,name='XGB_meta')
ET_test_meta = pd.Series(ET_test_meta,name='ET_meta')
KNN_test_meta = pd.Series(KNN_test_meta,name='KNN_meta')
DT_test_meta = pd.Series(DT_test_meta,name='DT_meta')
RF_test_meta = pd.Series(RF_test_meta,name='RF_meta')
ADBoost_test_meta = pd.Series(ADBoost_test_meta,name='ADBoost_meta')
LGB_test_meta = pd.Series(LGB_test_meta,name='LGB_meta')
XGB_test_meta = pd.Series(XGB_test_meta,name='XGB_meta')

train_meta=pd.concat([ET_meta,KNN_meta,DT_meta,RF_meta,ADBoost_meta,LGB_meta,XGB_meta],axis=1) # Meta features for train data
test_meta=pd.concat([ET_test_meta,KNN_test_meta,DT_test_meta,RF_test_meta,ADBoost_test_meta,LGB_test_meta,XGB_test_meta],axis=1) # Meta features for test data

for column in train_meta.columns:
    print('Cross validation score for ' + column + ': {}'.format(mean_squared_error(np.log1p(data['target']),train_meta[column]) ** .5))
g = sns.heatmap(test_meta[["ET_meta","KNN_meta","DT_meta","RF_meta","ADBoost_meta",'LGB_meta','XGB_meta']].corr(),cmap="BrBG",annot=True)
'''
for i in np.arange(0,1,0.05):
    oof_preds = LGB_meta*i+XGB_meta*(1-i)
    print(str(i)+':'+str(mean_squared_error(oof_preds,np.log1p(data['target'])) ** .5)
'''
    
# sub_preds = (RF_test_meta+LGB_test_meta)/2
# sub_preds = (ET_test_meta+LGB_test_meta)/2
# Generate the final result using the average prediction from LightGBM and XGBoost regressor
sub_preds = (LGB_test_meta+XGB_test_meta)/2
# Observe correlation between XGBoost and LightGBM result
g = sns.regplot(x='LGB_meta',y='XGB_meta',data=test_meta,fit_reg=False)
_ = g.set_title('XGBoost result versus LightGBM result')
#customerid = data['ID']
#result = pd.Series(gbm.predict(test_x),name='target')
#result = pd.Series(oof_preds,name='target')
#result = pd.concat([customerid,result],axis=1)
#min_value = train_y.min()
#result.loc[result['target'] < min_value,'target'] = min_value
#result.to_csv('Santander_train_2.csv',index=False)


test['target'] = np.expm1(sub_preds)
test[['ID', 'target']].to_csv('lgb_xgb.csv', index=False)