%load_ext autoreload

%autoreload 2



%matplotlib inline
!pip install xgboost==0.90



 !pip install fastai==0.7.0
from fastai.imports import *

from fastai.structured import *

import seaborn as sns

import matplotlib.pyplot as plt

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
df_raw = pd.read_csv(os.path.join("../input","train.csv"))
df_raw['Age'] = df_raw['YrSold'] - df_raw['YearBuilt']
df_raw['Remodel Age'] = df_raw['YrSold'] - df_raw['YearRemodAdd']
df_raw.drop(['YrSold'], axis=1,inplace=True)

df_raw.drop(['YearBuilt'], axis=1,inplace=True)

df_raw.drop(['YearRemodAdd'], axis=1,inplace=True)

df_raw.drop(['GarageArea'], axis=1,inplace=True)

df_raw['Quality'] = df_raw['OverallCond']+df_raw['OverallQual']

df_raw.drop(['OverallCond'], axis=1,inplace=True)

df_raw.drop(['OverallQual'], axis=1,inplace=True)

df_raw['Avg area of room'] = df_raw['GrLivArea']/df_raw['TotRmsAbvGrd']
train_cats(df_raw)
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=30)

df_trn2.drop(['Id'], axis=1,inplace=True)

def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 350  

n_trn = len(df_trn2)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df_trn2, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)

X_train.shape, y_train.shape, X_valid.shape



n_test = 100

def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train),

           rmse(m.predict(X_valid), y_valid),

           m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)


m = RandomForestRegressor(n_estimators=120, min_samples_leaf=1, 

                      max_features=0.6, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train) 

print_score(m)
#Understanding how number of estimators will effect the perfomance

preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:,0], np.mean(preds[:,0]), y_valid[0]

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.055, n_estimators=850,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 6, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(X_train, y_train) 

print_score(model_lgb)
from xgboost import XGBRegressor

clf3= XGBRegressor(max_depth=9,learning_rate=0.07,subsample=.8,min_child_weight=3,colsample_bytree=.6,scale_pos_weight=1,

gamma=10,reg_alpha=6,reg_lambda=1.1)

# n_estimators = 100 (default)

# max_depth = 3 (default)

clf3.fit(X_train,y_train)

print_score(clf3)
dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_valid, label=y_valid)



params = {

    'booster': 'gbtree', 

    'objective': 'reg:squarederror', # regression task

    'subsample': 0.8, # 80% of data to grow trees and prevent overfitting

    'colsample_bytree': 0.89, # 89% of features used

    'eta': 0.1,

    'gamma':0,

    'max_depth': 10,

    'seed': 42} # for reproducible results

params['eval_metric'] = "rmse"

num_boost_round = 999

model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'rmse'},

    early_stopping_rounds=10

)

cv_results
params['eval_metric'] = "rmse"

gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(5,12)

    for min_child_weight in range(1,8)

]



min_mae = float("Inf")

best_params = None



min_rmse = float("Inf")

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))



    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight



    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'rmse'},

        early_stopping_rounds=10

    )

    

      # Update best RMSLE

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = (max_depth,min_child_weight)



print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))  
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(5,10)]

    for colsample in [i/10. for i in range(5,10)]

]



min_rmse = float("Inf")

best_params = None



# We start by the largest values and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))



    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample



    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'rmse'},

        early_stopping_rounds=10

    )



    # Update best score

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = (subsample,colsample)



print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))
min_rmse = float("Inf")

best_params = None



for eta in [.022, .025, .02, .027, .028, .021]:

    print("CV with eta={}".format(eta))



    # We update our parameters

    params['eta'] = eta



    # Run and time CV

    cv_results = xgb.cv(params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=['rmse'],

            early_stopping_rounds=10

          )



    # Update best score

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds\n".format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = eta



print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
min_rmse = float("Inf")

best_params = None



for reg_alpha in [0, 0.001, 0.005, 0.01, 0.05]:

    print("CV with reg_alpha ={}".format(reg_alpha ))



    # We update our parameters

    params['reg_alpha '] = reg_alpha 



    # Run and time CV

    cv_results = xgb.cv(params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=['rmse'],

            early_stopping_rounds=10

          )



    # Update best score

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print("\tRMSE {} for {} rounds\n".format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = reg_alpha 



print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
params = {

    'booster': 'gbtree', 

    'objective': 'reg:squarederror', # regression task

    'subsample': 0.8, # 80% of data to grow trees and prevent overfitting

    'colsample' : 0.8, # 80% of features used

    'eta': 0.02,

    'max_depth': 6,

    'min_child_weight' : 5,

    'seed': 42} # for reproducible results

params['eval_metric'] = "rmse"

num_boost_round = 999

model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)
#Plotting a graph between the scores obtaibned from the validation sets of our models vs the score obtained

#from kaggle

x = [

0.12694616827013327,0.13292792672788983,

0.23365237301751454]

y = [

0.13052,0.14462,

0.26203]

import matplotlib.pyplot as plt

plt.plot(x, y, linewidth=3)



plt.show()
t=m.estimators_[0].tree_

fi = rf_feat_importance(m, df_trn2) 

fi.plot('cols', 'imp', figsize=(10,6), legend=False);
#Calculating feature importance by bar chart

def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
from scipy.cluster import hierarchy as hc

to_keep = fi[fi.imp>0.001].cols; len(to_keep)

df_keep = df_trn2[to_keep].copy()

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, 

      orientation='left', leaf_font_size=16)

plt.show()
from pdpbox import pdp # 

from plotnine import *

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(m, df_trn2,feature=feat_name,model_features=df_trn2.columns)

    return pdp.pdp_plot(p, feat_name, plot_lines=True, 

                        cluster=clusters is not None, 

                        n_cluster_centers=clusters)



plot_pdp('Age')
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(m, df_trn2,feature=feat_name,model_features=df_trn2.columns)

    return pdp.pdp_plot(p, feat_name, plot_lines=True, 

                        cluster=clusters is not None, 

                        n_cluster_centers=clusters)



plot_pdp('GrLivArea')
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(m, df_trn2,feature=feat_name,model_features=df_trn2.columns)

    return pdp.pdp_plot(p, feat_name, plot_lines=True, 

                        cluster=clusters is not None, 

                        n_cluster_centers=clusters)



plot_pdp('Quality')
df_raw['SalePrice'] = np.expm1(df_raw['SalePrice'])

f, ax = plt.subplots(figsize=(16, 8))

corr = df_raw.corr()

fig = sns.heatmap(corr)
fig = sns.distplot(df_raw['SalePrice'],color='darkcyan')
fig = sns.jointplot(x=df_raw["SalePrice"], y=df_raw["GrLivArea"], kind='scatter',s=200, color='pink', edgecolor="white", linewidth=2)
sns.jointplot(x=df_raw["SalePrice"], y=df_raw["Age"], kind='hex', color='skyblue',gridsize=13)
fig = sns.jointplot(x=df_raw["SalePrice"], y=df_raw["GarageCars"], kind='reg',color = 'limegreen')
f, ax = plt.subplots(figsize=(10, 8))

fig = sns.boxplot(x="Quality", y="SalePrice", data=df_raw)

fig.axis(ymin=0, ymax=800000)
f, ax = plt.subplots(figsize=(10, 8))

sns.swarmplot(x='TotRmsAbvGrd', y='SalePrice', hue='GrLivArea',

              data =df_raw,color = 'red',alpha=0.8)
f, ax = plt.subplots(figsize=(10, 8))

sns.lineplot(x='Avg area of room', y='SalePrice', data=df_raw, hue='BedroomAbvGr')