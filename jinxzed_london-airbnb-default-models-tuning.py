import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, log_loss, make_scorer
from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel,\
                                        f_regression, mutual_info_regression, univariate_selection
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,\
                                    validation_curve, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler,\
                                    FunctionTransformer, PowerTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option("display.max_columns", None)
def drop_cols(data, cols):
    if len(cols)==0:
        return df
    return data.drop(cols, axis=1)

def missing(data):
    if data.isna().sum().sum()==0:
        return "all missing values treated"
    data = data.isna().sum()/data.shape[0]
    data[data>0].plot(kind='bar', figsize=(16,7))
    all_miss = list(data[data==1].index)
    print("These columns have all the values missing",all_miss)
    plt.title("Missing value plot")
    plt.tight_layout()
    plt.xlabel("Column")
    plt.ylabel("Missing data in %")
    plt.xticks(rotation=90)
    plt.show()

def adj_r2(R2):
    return (1- ((1-R2) * (x_train.shape[0]-1)/(x_train.shape[0]-x_train.shape[1]-1)))

def mape(a, p):
    a = np.array(a).reshape(-1,1)
    p = np.array(p).reshape(-1,1)
    return np.mean( np.abs((a - p)/ a) )*100

def RMSLE(a, p):
    a = np.array(a).reshape(-1,1)
    p = np.array(p).reshape(-1,1)
    assert len(p) == len(a)
    return np.sqrt(np.mean( (np.log(p+1) - np.log(a+1))**2 ))

def RMSE(a, p):
    a = np.array(a).reshape(-1,1)
    p = np.array(p).reshape(-1,1)
    return np.sqrt(np.mean((p-a)**2))

rmsle_score = make_scorer(RMSLE, greater_is_better=False)
mape_score = make_scorer(mape, greater_is_better=False)
rmse_score = make_scorer(RMSE, greater_is_better=False)
def treat_skew(data, exclude=None, threshold = 1 ):
    cols = list(data.skew()[abs(data.skew())>threshold].index)    
    cols = [ col for col in cols if col not in exclude]
    return cols
df = pd.read_csv("/kaggle/input/londonairbnb/vishal_final.csv")
df = drop_cols(df, ["country","lastreviewdays", "firstreviewdays"])
new = pd.read_csv("/kaggle/input/airbnb-new-features/new_features.csv")
new = drop_cols(new, ['neighbourhood_cleansed','modeshare','accidents','cycling','name_pos','desc_pos','neig_pos','inte_pos'])
df = pd.concat((df, new), axis=1)
df.columns
df.shape
df.groupby('neighbourhood_cleansed')['review_scores_rating'].mean().plot(kind='bar')
plt.legend()
plt.show()
df.groupby('property_type')['review_scores_rating'].mean().plot(kind='bar')
plt.show()
df.boxplot(by='neighbourhood_cleansed',column=['distance'], figsize=(16,6), showmeans=True)#.plot(kind='kde')
plt.xticks(rotation=90)
plt.legend()
plt.show()
df.groupby("property_type")['price'].mean().plot(kind='bar')
plt.show()

original_price = df["price"]
train, test = train_test_split(df, train_size=0.7, random_state=1)
train.shape, test.shape

def neigh_feat(df):
    neigbourhood_reviews = df.groupby(['neighbourhood_cleansed','property_type'])['review_scores_rating'].mean().reset_index()
    neigbourhood_reviews.columns = ['neighbourhood_cleansed','property_type','mean_rating']
    df = df.merge(neigbourhood_reviews, on =['neighbourhood_cleansed','property_type'], how='left')
    df["rating_mean"] = (df["review_scores_rating"]/df['mean_rating'])
    df.drop('mean_rating', axis=1, inplace=True)

    neigbourhood_distance = df.groupby(['neighbourhood_cleansed','property_type'])['distance'].mean().reset_index()
    neigbourhood_distance.columns = ['neighbourhood_cleansed','property_type','mean_distance']
    df = df.merge(neigbourhood_distance, on =['neighbourhood_cleansed','property_type'], how='left')
    df["distance_mean"] = (df["distance"]/df['mean_distance'])
    df.drop('mean_distance', axis=1, inplace=True)
    return df

train = neigh_feat(train)
test = neigh_feat(test)
def feature_engineering(df, aggregate = None, feature = None, funcs =None):
    if funcs==None:
        funcs = ['mean', 'median', 'min', 'max']
    data = df.groupby(feature)[aggregate].agg(funcs)
    group = "_".join(feature)+"_"+aggregate
    data.columns = [group+"_"+func for func in funcs]
    return data

funcs = ['mean','median','min','max']
cols = ['neighbourhood_cleansed','property_type']

data_price = feature_engineering(train, 'price', cols)
#train = train.merge(data_price, on=cols, how='left')
#test = test.merge(data_price, on=cols, how='left')

data_security = feature_engineering(train, 'security_deposit', cols)
data_clean = feature_engineering(train, 'cleaning_fee', cols)
data_extra = feature_engineering(train, 'extra_people', cols)

#train = train.merge(data_security, on=cols, how='left')
#train = train.merge(data_clean, on=cols, how='left')
#train = train.merge(data_extra, on=cols, how='left')

data_security = feature_engineering(test, 'security_deposit', cols)
data_clean = feature_engineering(test, 'cleaning_fee', cols)
data_extra = feature_engineering(test, 'extra_people', cols)

#test = test.merge(data_security, on=cols, how='left')
#test = test.merge(data_clean, on=cols, how='left')
#test = test.merge(data_extra, on=cols, how='left')
x_train = train.drop(['id','price'], axis=1)
y_train = np.log(train['price'])
x_test = test.drop(['id','price'], axis=1)
y_test = np.log(test['price'])
x_train = x_train.replace({np.Inf:0})
x_test = x_test.replace({np.Inf:0})
def feature_select_rf(train=None, test=None, y=y_train):
    columns = train.columns
    selector_rf.fit(train, y)
    columns_rf = [col for col, flag in zip(columns,selector_rf.get_support()) if flag]
    x_select = train[columns_rf]
    print(f"Feature shape selected : {x_select.shape}")
    test_select = test[columns_rf]
    return (x_select, test_select)

def feature_select_lgbm(train=None, test=None, y=y_train):
    columns = train.columns
    selector_lgbm.fit(train, y)
    columns_lgbm = [col for col, flag in zip(columns,selector_lgbm.get_support()) if flag]
    x_select = train[columns_lgbm]
    print(f"Feature shape selected : {x_select.shape}")
    test_select = test[columns_lgbm]
    return (x_select, test_select)

def preprocess_onehot(train, test=None, y=y_train, select=False, flag='rf'):
    ct_onehot.fit(train)
    onehot_cols = list(ct_onehot.named_transformers_['onehot'].get_feature_names(dummy))
    all_columns = skew + passthru + onehot_cols
    x_transform = pd.DataFrame(scaler.fit_transform(ct_onehot.transform(train)), columns=all_columns)
    x_transform = pd.DataFrame(numeric_skew.fit_transform(imputer_br.fit_transform(x_transform)), columns=all_columns)
    test_transform = pd.DataFrame(scaler.transform(ct_onehot.transform(test)), columns=all_columns)
    test_transform = pd.DataFrame(numeric_skew.transform(imputer_br.transform(test_transform)), columns=all_columns)
    print("Original Data Shape", x_transform.shape, test_transform.shape)
    if select==False:
        return x_transform, test_transform
    print("Selecting the final features...")
    if flag=='rf':
        print("Feature selection using Random Forest")
        x_transform, test_transform = feature_select_rf(x_transform, test_transform, y_train)
    elif flag=='lgbm':
        print("Feature selection using Lightgbm")
        x_transform, test_transform = feature_select_lgbm(x_transform, test_transform, y_train)
    return (x_transform, test_transform)
x_train.columns
numeric = ["host_acceptance_rate","accommodates","bathrooms","bedrooms","beds","security_deposit","cleaning_fee",
       "guests_included","extra_people","availability_30","availability_60","availability_90","availability_365",
       "number_of_reviews","number_of_reviews_ltm","review_scores_rating","reviews_per_month","amenity_sum","distance","premium",
      'hostsincedays','firstreviewdays','lastreviewdays','walking','cycletrack','rating_mean', 'distance_mean',
          'review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value']
pt = PowerTransformer(method='yeo-johnson')
treatment = pd.DataFrame(x_train[numeric].skew())
treatment['PT'] = pd.DataFrame(pt.fit_transform(x_train[numeric]), columns = numeric).skew()

skew = treat_skew(x_train[numeric], ['host_acceptance_rate'])
#treatment
categorical = list(set(x_train.columns).difference(set(numeric)))

dummy = ["experiences_offered","property_type","room_type","bed_type","cancellation_policy","host_response_time"]
passthru = list(set(x_train.columns).difference(set(dummy+skew)))
pt = PowerTransformer(method='yeo-johnson')

impute_br = BayesianRidge()
imputer_br = IterativeImputer(impute_br, skip_complete=True)

select_rf = RandomForestRegressor(random_state=1, n_jobs=4)
select_lgbm = LGBMRegressor(n_jobs=4, random_state=1)

onehot = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

ct_onehot = ColumnTransformer([('pass1','passthrough', skew),
                               ('pass2','passthrough', passthru),
                               ('onehot', onehot,   dummy)    ], remainder='drop')

numeric_skew = ColumnTransformer([('skew', pt, np.arange(len(skew)))],
                                  remainder='passthrough')
selector_rf = SelectFromModel(select_rf, threshold='0.5*median')
selector_lgbm = SelectFromModel(select_lgbm, threshold='0.5*median')
x_transform, test_transform = preprocess_onehot(x_train, x_test, select=False)
x_transform.columns

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
vif_val = pd.DataFrame(index = x_transform[numeric].columns)
vif_val["VIF"]= [vif(x_transform[numeric].values, i) for i in range(x_transform[numeric].shape[1])]
vif_val

x_select_rf, test_select_rf = preprocess_onehot(x_train, x_test, select=True, flag='rf')
x_select_rf.columns
x_select_lgb, test_select_lgb = preprocess_onehot(x_train, x_test, select=True, flag='lgbm')
x_select_lgb.columns
#pd.concat((x_select, y_train.reset_index(drop=True)), axis=1).to_csv("train.csv")
#pd.concat((test_select, y_test.reset_index(drop=True)), axis=1).to_csv("test.csv")
def plot_importance(col, model):
    plt.figure(figsize=(16,5))
    plt.bar(col, model.feature_importances_)
    plt.title("Feature Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
def model_results(model, x_train, y_train, x_test, y_test, return_pred = False, fit=True, data=('metrics','imp')):
    if fit:
        model.fit(x_train, y_train)
    trainpred = model.predict(x_train).reshape(-1,1)
    testpred = model.predict(x_test).reshape(-1,1)
    r2train = r2_score(y_train, trainpred)
    r2test = r2_score(y_test, testpred)
    if data[0] =='metrics':
        print("Train R2 : %.4f Test R2 : %.4f"%( r2train, r2test))
        print("Train RMSE : %.4f Test RMSE : %.4f"%(RMSE(y_train, trainpred), RMSE(y_test, testpred)))
        print("Train adj R2 : %.4f Test adj R2 : %.4f"%(adj_r2(r2train), adj_r2(r2test)))
        print("Train RMSLE : %.4f Test RMSLE : %.4f"%(RMSLE(y_train, trainpred), RMSLE(y_test, testpred)))
    if data[1] =='imp':
        plot_importance(x_train.columns, model)
    if return_pred:
        return (trainpred, testpred)

dt = DecisionTreeRegressor(random_state=1)
rf= RandomForestRegressor(n_jobs=4, random_state=1)
hist = HistGradientBoostingRegressor(random_state=1)
gb_default = GradientBoostingRegressor(random_state=1, n_estimators=1000, learning_rate=0.03)
rf_default = RandomForestRegressor(n_jobs=4, random_state=1)
xgb_default = XGBRegressor(n_jobs=4, random_state=1, verbosity=0)
lgb_default = LGBMRegressor(n_jobs=4, random_state=1)
cat_default = CatBoostRegressor(random_state=1, verbose=0)

models = []
models.append(("GBR",gb_default))
models.append(("RF",rf_default))
models.append(("XGB",xgb_default))
models.append(("LGB",lgb_default))
models.append(("CAT",cat_default))
r2 = []
names = []
cv = ShuffleSplit(n_splits=3, train_size=0.7, random_state=1)

#def feature_imp(models, x, y, cv=cv):
imp = pd.DataFrame({"Col":x_transform.columns})
fea_imp = []
for name, model in models[1:]:
    feat = []
    print(f"Training model {name}...")
    for train_index, test_index in cv.split(x_transform, y_train):
        xtrain, xtest = x_transform.iloc[train_index,:], x_transform.iloc[test_index,:]
        ytrain, ytest = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(xtrain, ytrain)
        feat.append(model.feature_importances_)
        names.append(name)
        r2.append(r2_score(ytest, model.predict(xtest)))
    imp[name] = np.mean(feat, axis=0)
    fea_imp.append(feat)
#return imp
imp.sort_values('CAT', ascending=False).T
imp.to_csv("feature_importance.csv", index=False)
plt.boxplot(np.array(r2).reshape(3, -1))
plt.xticks(np.arange(1,5), ['RF','XGB','LGB','CAT'])
plt.title("Model Comparison")
plt.xticks(rotation=90)
plt.show()

for name, model in models[1:]:
    print(f"Training model {name}...")
    model_results(model, x_select_lgb, y_train, test_select_lgb, y_test, data=('metrics','no'))
    print()
for name, model in models[1:]:
    print(f"Training model {name}...")
    model_results(model, x_select_rf, y_train, test_select_rf, y_test, data=('metrics','no'))


gb_tune = GradientBoostingRegressor(random_state=1, learning_rate=0.02364633, n_estimators = 906, subsample=0.6120154, criterion='friedman_mse',
                                    max_depth=13, min_samples_split=17, min_samples_leaf = 18, min_impurity_decrease=0.000529976)
model_results(gb_tune, x_select_rf, y_train, test_select_rf, y_test)

rf_tune = RandomForestRegressor(n_jobs=4, random_state=1, n_estimators = 2291, max_depth=12, max_leaf_nodes=389,
                                max_features='auto',min_samples_leaf=2, min_samples_split=6)
model_results(rf_tune, x_select_rf, y_train, test_select_rf, y_test)

xgb_tune = XGBRegressor(n_jobs=4, random_state=1, verbosity=0, n_estimators=793, max_depth=10, reg_lambda=7,reg_alpha=7,
                        colsample_bynode=0.8, gamma=0.001, colsample_bytree=0.75, learning_rate=0.08, subsample=0.7)
model_results(xgb_tune, x_select_rf, y_train, test_select_rf, y_test)

lgb_tune = LGBMRegressor(n_jobs=4, random_state=1, n_estimators=200, max_depth=12, num_leaves=85, learning_rate=0.08,
                         lambda_l1 = 6.154617e-2, lambda_l2 = 9.02934605e-3,feature_fraction=0.75, bagging_fraction=0.9,
                         bagging_freq=1, min_child_samples=50)
model_results(lgb_tune, x_select_rf, y_train, test_select_rf, y_test)

cat_tune = CatBoostRegressor(random_state=1, verbose=0, metric_period=100, eval_metric='R2',iterations=2000,
                             #learning_rate=0.05, depth=7, reg_lambda=0.1, #bagging_temperature=0.5, subsample=0.6, colsample_bylevel=0.45,
                             #od_type='Iter', od_wait=3#, min_data_in_leaf=100,#grow_policy='Depthwise' 
                            )
cat_tune.fit(x_select_lgb, y_train, eval_set=(test_select_lgb, y_test), early_stopping_rounds=10, use_best_model=True, verbose=True, plot=True)
model_results(cat_tune, x_select_rf, y_train, test_select_rf, y_test, fit=False)



run = [rf_default, xgb_default, lgb_default, cat_default]
trainpreds = []
testpreds = []
for model in run:
    trainpred, testpred = model_results(model, x_select_rf, y_train, test_select_rf, y_test, return_pred=True)
    trainpreds.append(trainpred)
    testpreds.append(testpred)
    
stacktrain = np.mean(trainpreds, axis=0)
stacktest = np.mean(testpreds, axis=0)

print("Train R2 : %.4f Test R2 : %.4f"%(r2_score(y_train, stacktrain), r2_score(y_test, stacktest)))
print("Train RMSE : %.4f Test RMSE : %.4f"%(RMSE(y_train, stacktrain), RMSE(y_test, stacktest)))
print("Train MAPE : %.4f Test MAPE : %.4f"%(mape(y_train, stacktrain), mape(y_test, stacktest)))
print("Train RMSLE : %.4f Test RMSLE : %.4f"%(RMSLE(y_train, stacktrain), RMSLE(y_test, stacktest)))
plt.scatter(np.exp(y_train), np.exp(stacktrain))
plt.show()

plt.scatter(np.exp(y_test), np.exp(stacktest))
plt.show()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, RepeatedKFold, ShuffleSplit
kfold = KFold(n_splits = 10, shuffle=True, random_state=1)
repeat = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
split = ShuffleSplit(n_splits=10, train_size=0.7, random_state=1)
def tune_model(model, X, Y, params, cv=5, iters = 100, metric=rmse_score, grid=False):
    if grid:
        print("Tuning the Parameters using full search space...")
        search = GridSearchCV(model, params, scoring=metric, n_jobs=4, cv=cv, verbose=1, return_train_score=True)
        search.fit(X, Y)
        return search
    else:
        print("Tuning using Randomized Parameters search space...")
        search = RandomizedSearchCV(model, params, scoring=metric, n_jobs=4, cv=cv, verbose=1, return_train_score=True, n_iter=iters)
        search.fit(X, Y)
        return search
repeat = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

xgb_default = XGBRegressor(n_jobs=4, random_state=1, verbosity=0)
lgb_default = LGBMRegressor(n_jobs=4, random_state=1)
cat_default = CatBoostRegressor(random_state=1, verbose=0)

models = []
models.append(("XGB",xgb_default))
models.append(("LGB",lgb_default))
models.append(("CAT",cat_default))
r2 = []
names = []
for name, model in models:
    scores = cross_val_score(model, x_select_lgb, y_train, scoring='r2', n_jobs=4, cv=repeat)
    r2.append(scores)
    names.append(name)
plt.boxplot(r2, labels=names)
plt.title("Model comparisons")
plt.show()
r2
#LGBM PARAMS
#'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1,
#'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': 4, 'num_leaves': 31, 'objective': None,
#'random_state': 1, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0
def objective_lgb(trial):
    params= {'n_estimators':trial.suggest_int('n_estimators',1200, 1800), 'n_jobs':4, 'random_state':1,
            'boosting_type':'gbdt',#trial.suggest_categorical('boosting_type',['gbdt','rf']),
            'max_depth':trial.suggest_int('max_depth',10, 20), 'num_leaves': trial.suggest_int('num_leaves', 150, 300), 
            'learning_rate':trial.suggest_uniform('learning_rate',0.01, 0.3),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1),'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 0.6),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 2), 
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 40) }
    model = LGBMRegressor(**params)
    #train, test, ytrain, ytest = train_test_split(x_select, y_train, train_size=0.7, random_state=1)
    split = ShuffleSplit(n_splits=3, train_size=0.7, random_state=1)
    scores = cross_val_score(model, x_select, y_train, cv=split, n_jobs=4, scoring='r2')
    #model.fit(train, ytrain)
    #preds = model.predict(test)
    #return r2_score(ytest, preds)
    return np.mean(scores)
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=50)
print("Number of finished trials: {}".format(len(study_lgb.trials)))
print("Best trial:")
trial_lgb = study_lgb.best_trial
print("  Value: {}".format(trial_lgb.value))
print("  Params: ")
for key, value in trial_lgb.params.items():
    print("    {}: {}".format(key, value))

#LGBM PARAMS
#'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1,
#'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': 4, 'num_leaves': 31, 'objective': None,
#'random_state': 1, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0
def objective_gbr(trial):
    params= {'n_estimators':trial.suggest_int('n_estimators',200, 1000), 'random_state':1, 'subsample': trial.suggest_uniform('subsample', 0.3, 1), 
            'criterion':trial.suggest_categorical('criterion',['mse','friedman_mse']),
            'max_depth':trial.suggest_int('max_depth',3, 20), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 40),
            'learning_rate':trial.suggest_uniform('learning_rate',0.01, 0.7),'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 40),
            'min_impurity_decrease':trial.suggest_uniform('min_impurity_decrease',1e-8,0.01)
            }
    model = GradientBoostingRegressor(**params)
    train, test, ytrain, ytest = train_test_split(x_select, y_train, train_size=0.7, random_state=1)
    #split = ShuffleSplit(n_splits=3, train_size=0.7, random_state=1)
    #scores = cross_val_score(model, x_select, y_train, cv=split, n_jobs=4, scoring='r2')
    model.fit(train, ytrain)
    preds = model.predict(test)
    return r2_score(ytest, preds)
    #return np.mean(scores)
study_gbr = optuna.create_study(direction="maximize")
study_gbr.optimize(objective_gbr, n_trials=100)
print("Number of finished trials: {}".format(len(study_gbr.trials)))
print("Best trial:")
trial_gbr = study_gbr.best_trial
print("  Value: {}".format(trial_gbr.value))
print("  Params: ")
for key, value in trial_gbr.params.items():
    print("    {}: {}".format(key, value))

# RF PARAMS
#'bootstrap': True, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None,
#'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0,
#'n_estimators': 100, 'n_jobs': 4, 'oob_score': False, 'random_state': 1, 'verbose': 0, 'warm_start': False
def objective_rf(trial):
    params={'n_estimators':trial.suggest_int('n_estimators',500, 2500), 'max_depth':trial.suggest_int('max_depth',3, 20),
        'n_jobs':4, 'random_state':1, #'criterion':trial.suggest_categorical('criterion',['mse','mae']),
        'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',50,400),
            'max_features':trial.suggest_categorical('max_features',['auto','sqrt','log2']),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,10),'min_samples_split':trial.suggest_int('min_samples_split',2, 20)
           }
    model = RandomForestRegressor(**params)
    train, test, ytrain, ytest = train_test_split(x_select, y_train, train_size=0.7, random_state=1)
    #split = ShuffleSplit(n_splits=3, train_size=0.7, random_state=1)
    #scores = cross_val_score(model, x_select, y_train, cv=split, n_jobs=4, scoring='r2')
    model.fit(train, ytrain)
    preds = model.predict(test)
    return r2_score(ytest, preds)
    #return np.mean(scores)
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=100)
print("Number of finished trials: {}".format(len(study_rf.trials)))
print("Best trial:")
trial_rf = study_rf.best_trial
print("  Value: {}".format(trial_rf.value))
print("  Params: ")
for key, value in trial_rf.params.items():
    print("    {}: {}".format(key, value))
data_rf_study = study_rf.trials_dataframe()
data_rf_study.to_csv('study_rf.csv', index=False)

#from optuna.distributions import UniformDistribution, IntUniformDistribution, DiscreteUniformDistribution, LogUniformDistribution
#from optuna.integration import OptunaSearchCV
#params= {'n_estimators':IntUniformDistribution(1200, 1800),'max_depth':IntUniformDistribution(10, 20),
#         'num_leaves': IntUniformDistribution(150, 300),'learning_rate':UniformDistribution(0.01, 0.3),
#         'lambda_l1': LogUniformDistribution( 1e-8, 1),'lambda_l2': LogUniformDistribution(1e-8, 1),
#         'feature_fraction': UniformDistribution( 0.2, 0.6),'bagging_fraction': UniformDistribution(0.6, 1.0),
#         #'bagging_freq': IntUniformDistribution( 1, 3),
#         'min_child_samples': IntUniformDistribution( 2, 40) }

#tuner = OptunaSearchCV(LGBMRegressor(n_jobs=4, random_state=1,boosting_type='gbdt',bagging_freq=1), params, n_jobs=4, 
#                       n_trials=100, random_state=1, scoring='r2', verbose=1, cv=None)
#tuner.fit(x_select, y_train)

