# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re as re

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')
# loading the dataset
train = pd.read_csv('../input/train.csv', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', index_col='PassengerId')
train.shape, test.shape
Y = train['Survived']
train = train.drop('Survived', axis=1)
train['Training_set'] = True
test['Training_set'] = False
df_comb = pd.concat([train, test])
df_comb.isnull().sum()
# Feature Engineering
##1. Family Size as combination of SibSb + Parch - from Sina
df_comb['FamilySize'] = df_comb['SibSp'] + df_comb['Parch'] + 1
# check if columns have duplicate ticket number. Duplicate ticket numbers symbolize family/friends
tks = df_comb['Ticket']
dataset_dup = df_comb[tks.isin(tks[tks.duplicated()])]
df_comb['TkDup'] = df_comb.Ticket.isin(dataset_dup['Ticket'])
df_comb.head(5)
df_comb[df_comb['Ticket'] == '113803']
# 2. IsAlone using FamilySize and DupTicket
df_comb['IsAlone'] = 0
dataset_filter = (df_comb['FamilySize'] == 1) & (df_comb['TkDup'] == False)
df_comb['IsAlone'] = np.where(dataset_filter, 1, 0)
df_comb.head(5)
# 3. Cabin
df_comb.loc[df_comb.Cabin.notnull() & df_comb.Cabin.str.contains('F'), 'Cabin']
deck_list = list(map(lambda x: x[0], df_comb.Cabin.dropna().tolist()))
print(deck_list)
# Deck as the initial of the Cabin. If no cabin, use 'X'
crit = df_comb['Cabin'].isnull()
df_comb['Deck'] = df_comb['Cabin'].astype(str).str[0].where(~crit, other='X')
df_comb.head()
df_comb.drop('Cabin', axis=1, inplace=True)
# some passengers booked more than a single cabin
df_comb.loc[[28, 76, 89, 129]]
df_comb.head(3)
# 4. Title derived from Name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name) # regex starting with space ending with .
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

df_comb['Title'] = df_comb['Name'].apply(get_title)

print(pd.crosstab(df_comb['Title'], df_comb['Sex']))
df_comb['Title'] = df_comb['Title'].replace(['Mlle','Ms', 'Countess', 'Lady', 'Dona'], 'Miss')
df_comb['Title'] = df_comb['Title'].replace('Mme', 'Mrs')
df_comb['Title'] = df_comb['Title'].replace(['Capt','Col', 'Don', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
print(pd.crosstab(df_comb['Title'], df_comb['Sex']))
# 5., 6. setting age and fare to median value in its class
df_comb.loc[df_comb.Age.isnull(), 'Age'] = df_comb.groupby(['Title', 'Pclass']).Age.transform('mean')
df_comb.loc[df_comb.Fare.isnull(), 'Fare'] = df_comb.groupby(['Title', 'Pclass']).Fare.transform('mean')
#7. Embarked
df_comb[df_comb.Embarked.isnull()]
df_comb[df_comb['Fare'].between(75, 85) & (df_comb.Pclass==1) & (df_comb.Sex=='female') & (df_comb.Deck.str.startswith('B'))]
# cannot identify which Embark, so marking to mode
df_comb['Embarked']= df_comb['Embarked'].fillna(value=df_comb['Embarked'].value_counts().index[0])
df_comb.loc[[62, 830]]
#8. Time travelled S(11), C(5), Q(4) in Days before hitting iceberg (15th April 1912) 
# calculated based on embarkment port route of  S(11), C(5), Q(4) using info 
# from https://discovernorthernireland.com/things-to-do/attractions/titanic/titanic-sailing-route-map/

df_comb['TimeTravelled'] = 11 # default for S
dataset_filter = (df_comb['Embarked'] == 'C')
dataset_filter_1 = (df_comb['Embarked'] == 'Q')
df_comb['TimeTravelled'] = np.where(dataset_filter, 5, df_comb['TimeTravelled'].values)
df_comb['TimeTravelled'] = np.where(dataset_filter_1, 4, df_comb['TimeTravelled'].values)
df_comb.head(3)
df_comb.drop(["Name", "TkDup", "Ticket"], axis=1, inplace=True)
df_comb.head(3)
#oneHotEncoding Sex, Embarked, Title,Deck
df_comb = df_comb.join(pd.get_dummies(df_comb[['Sex', 'Embarked','Title', 'Deck']]))
df_comb.drop(['Sex','Embarked','Title', 'Deck'], axis=1, inplace=True)
df_comb.head(3)
df_comb.info()
# solving dummy variable drop by dropping one categorical value in each encong
df_comb.drop(['Sex_female','Embarked_C','Title_Dr', 'Deck_X'], axis=1, inplace=True)
df_comb.info()
df_comb.head()
train = df_comb[df_comb['Training_set'] == True]
train.drop('Training_set', axis=1, inplace=True)
test = df_comb[df_comb['Training_set'] == False]
test.drop('Training_set', axis=1, inplace=True)
train.shape, test.shape
train.head()
train_surv = pd.concat([train, Y], axis =1)
train_surv.head()
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_surv.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train_surv[[u'Survived', u'Pclass', u'Sex_male', u'Age', u'Parch', u'Fare', u'Embarked_S', u'IsAlone', u'TimeTravelled',
       u'FamilySize', u'Deck_A', u'Deck_B', u'Deck_F', u'Deck_G', u'Deck_T']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
train_bfe = train.copy() # duplicating
test_bfe = test.copy() # duplicating
train_bfe.insert(0,'Bias',1) # adding bias for statsmodel
test_bfe.insert(0,'Bias',1)
train_bfe.head(5)
import statsmodels.formula.api as sm
# new regressor from sm
regressor_OLS = sm.OLS(endog=Y, exog=train_bfe).fit()
regressor_OLS.summary()
# automated backward elimination with Adjusted R Square
import statsmodels.formula.api as sm
def backwardElimination(X, Y, SL):
    x = X.values
    y = Y.values
    index_r = np.arange(X.shape[1])
    index_r = np.reshape(index_r, (1,index_r.shape[0]))
    index_d = index_r
    print(index_r)
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp = x
                    x = np.delete(x, j, 1)
                    index_r_temp = index_r
                    index_r = np.delete(index_r, j, 1)
                    print(index_r)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = temp
                        index_rollback = index_r_temp
                        print (regressor_OLS.summary())
                        deleted = np.append(np.setdiff1d(index_d, index_rollback),0)
                        return x_rollback, deleted
                    else:
                        continue
    regressor_OLS.summary()
    return x
SL = 0.05
X_Modeled, index_deleted = backwardElimination(train_bfe, Y, SL)
print(index_deleted)
train.head(2)
train_bfe.drop(columns=train_bfe.columns[index_deleted]).head(3)
test_bfe.drop(columns=test_bfe.columns[index_deleted]).head(3)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

print(train.shape,test.shape,Y.shape)
type(Y)
# cross validation stratgey 
# Splitting the Training dataset into further Training set and Cross validation Set, this will allow us to validate the final model
# Stack layer - 0
SEED = 0 # for reproducibility

X_train = train_bfe.values
Y_train = Y.values
x_train, x_cv, y_train, y_cv = train_test_split(X_train, Y_train, test_size = 0.15, random_state = SEED)
x_test = test_bfe.values
print("Train %s, CV %s, Test %s"%(x_train.shape, x_cv.shape, x_test.shape))
#Validation function
NFOLDS = 10

kf_i = KFold(NFOLDS, shuffle=True, random_state=42)
kf = kf_i.get_n_splits(x_train)

def f1s_cv(model):
    f1s= cross_val_score(model, x_train, y_train, scoring="f1", cv = kf_i)
    return(f1s)

# Making the Confusion Matrix and f1score function

def f1s_e(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    precision = (cm[0,0]/(cm[0,0]+cm[0,1]))*100
    recall = (cm[0,0]/(cm[0,0]+ cm[1,0]))*100
    FS = (2*(precision*recall))/(precision+recall)
    return FS

def f1s_ec(y, y_pred_ec, threshold = .5):
        #convert into binary values
        for i in range(0,y_pred_ec.shape[0]):
            if y_pred_ec[i]>=threshold:       # setting threshold to .5
                y_pred_ec[i]=1
            else:  
                y_pred_ec[i]=0
        cm = confusion_matrix(y, y_pred_ec)
        precision = (cm[0,0]/(cm[0,0]+cm[0,1]))*100
        recall = (cm[0,0]/(cm[0,0]+ cm[1,0]))*100
        FS = (2*(precision*recall))/(precision+recall)
        return FS
    
# to time randomized or grid search
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# Feature Scaling all three sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_cv_scaled = sc.transform(x_cv)
x_test_scaled = sc.transform(x_test)
# testing various classification models
pipelines = []

pipelines.append(('ScaledSAG', Pipeline([('Scaler', StandardScaler()),('SAG', LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / x_train.shape[0]))])))
pipelines.append(('ScaledXGBC', Pipeline([('Scaler', RobustScaler()),('XGBC', xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05))])))
pipelines.append(('ScaledKNC', Pipeline([('Scaler', StandardScaler()),('KNC', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))])))
pipelines.append(('ScaledSVC', Pipeline([('Scaler', StandardScaler()),('SVC', SVC(kernel = 'rbf'))])))
pipelines.append(('ScaledGNB', Pipeline([('Scaler', StandardScaler()),('GNB', GaussianNB())])))
pipelines.append(('DTC', DecisionTreeClassifier(criterion = 'entropy')))
pipelines.append(('RFC', RandomForestClassifier(n_estimators = 100)))
pipelines.append(('RobustABC', Pipeline([('Robust', RobustScaler()),('ABC', AdaBoostClassifier())])))
pipelines.append(('ScaledGBC', Pipeline([('Scaler', StandardScaler()),('GBC', GradientBoostingClassifier())])))
pipelines.append(('ScaledETR', Pipeline([('Scaler', StandardScaler()),('ETR', ExtraTreesClassifier())])))
pipelines.append(('ScaledPAC', Pipeline([('Scaler', StandardScaler()),("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0))])))
pipelines.append(('ScaledGPC', Pipeline([('Scaler', StandardScaler()),('GPC', GaussianProcessClassifier(1.0 * RBF(1.0)))])))
pipelines.append(('ScaledMLPC', Pipeline([('Scaler', StandardScaler()),('MLPC', MLPClassifier(alpha=1))])))

results = []
names = []
for name, model in pipelines:
    cv_results = f1s_cv(model)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Class to extend the Sklearn classifier
class SklearnWrapper(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
# Class to extend the Sklearn classifier for XGB

class XgbWrapper(object):
    def __init__(self,params=None):
        self.param = params
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x, threshold = .5):
        y_pred_xgb=self.gbdt.predict(xgb.DMatrix(x))
        #convert into binary values
        for i in range(0,x.shape[0]):
            if y_pred_xgb[i]>=threshold:       # setting threshold to .5
                y_pred_xgb[i]=1
            else:  
                y_pred_xgb[i]=0
        return y_pred_xgb
# Class to extend the Sklearn classifier for LGBM

class LgbWrapper(object):

    def __init__(self,params=None):
        self.param = params

    def train(self, x, y, tsize = .2):
        lgb_x_train, lgb_x_cv, lgb_y_train, lgb_y_cv = train_test_split(x, y, test_size = tsize, random_state = SEED)
        d_train = lgb.Dataset(lgb_x_train, lgb_y_train)
        d_valid = lgb.Dataset(lgb_x_cv, lgb_y_cv)

        self.lgbt = lgb.train(self.param,
                d_train, 
                100000,
               valid_sets=[d_valid],
               early_stopping_rounds=100,
                verbose_eval=1000)

    def predict(self, x, threshold = .5):
        y_pred_lgb=self.lgbt.predict(x)
        #convert into binary values
        for i in range(0,x.shape[0]):
            if y_pred_lgb[i]>=threshold:       # setting threshold to .5
                y_pred_lgb[i]=1
            else:  
                y_pred_lgb[i]=0
        return y_pred_lgb
def get_oof(clf, o_x_train, o_y_train, o_x_cv, o_x_test):
    oof_train = np.zeros((o_x_train.shape[0],))
    oof_cv = np.zeros((o_x_cv.shape[0],))
    oof_cv_skf = np.empty((NFOLDS, o_x_cv.shape[0]))
    oof_test = np.zeros((o_x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, o_x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf_i.split(o_x_train)):
        x_tr = o_x_train[train_index]
        y_tr = o_y_train[train_index]
        x_te = o_x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)

        oof_cv_skf[i, :] = clf.predict(o_x_cv)
        oof_test_skf[i, :] = clf.predict(o_x_test)

    oof_cv[:] = oof_cv_skf.mean(axis=0)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_cv.reshape(-1, 1), oof_test.reshape(-1, 1)
# knc_grid = GridSearchCV(
#   estimator = KNeighborsClassifier(),
#     param_grid = {
#         'n_neighbors':[5,15,20],
#         'leaf_size':[2,5,8],
#         'p':[1,2]},
#     cv = kf,
#     scoring = "f1"    
# )
# knc_grid.fit(x_train_scaled,y_train)
# knc_params = knc_grid.best_params_
# print("KNC Params%s, Score%s"%(knc_grid.best_params_,knc_grid.best_score_))
# svc_grid = GridSearchCV(
#   estimator = SVC(kernel = 'rbf'),
#     param_grid = {
#         'C':[0.1, 1, 2, 3, 10],
#         'gamma':[0.01, 0.02, 0.03, 0.05, .1, .2, .3]
#     },
#     cv = kf,
#     scoring = "f1"    
# )
# svc_grid.fit(x_train_scaled,y_train)
# svc_params = svc_grid.best_params_
# print("SVC Params%s, Score%s"%(svc_grid.best_params_,svc_grid.best_score_))

# et_grid = GridSearchCV(
#   estimator = ExtraTreesClassifier(),
#     param_grid = {
#         "n_estimators": [10, 100, 450,500,550, 575],
#         "max_depth": [1, 10, 100, 500, 1000],
#         "min_samples_leaf": [1,3,4,5, 10], 
#         'max_features': ('auto', 'sqrt','log2')
#     },
#     n_jobs=-1,
#     cv = kf,
#     scoring = "f1"    
# )
# et_grid.fit(x_train_scaled,y_train)
# et_params = et_grid.best_params_
# print(et_grid.grid_scores_)
# print("ET Params%s, Score%s"%(et_grid.best_params_,et_grid.best_score_))
# mlp_grid = GridSearchCV(
#   estimator = MLPClassifier(),
#     param_grid = {'activation':['relu'], 'solver': ['lbfgs'], 'max_iter': [500], 
#                   'alpha': [.001], 
#                   'hidden_layer_sizes':np.arange(9, 11), 'random_state':[3]}, 
#     cv = kf,
#     scoring = "f1",
#     n_jobs = -1
# )
# mlp_grid.fit(x_train,y_train)
# mlp_params = mlp_grid.best_params_
# print("MLP Params%s, Score%s"%(mlp_grid.best_params_,mlp_grid.best_score_))
# ada_grid = GridSearchCV(
#   estimator = AdaBoostClassifier(),
#     param_grid = {'n_estimators': [1, 10, 50, 100, 200, 500],
#                   'learning_rate': [0.01, 0.03, 0.05, .1, 1],
#                   'algorithm': ['SAMME', 'SAMME.R']},
#     cv = kf,
#     scoring = "f1"    
# )
# ada_grid.fit(x_train_scaled,y_train)
# print(ada_grid.cv_results_)
# ada_params = ada_grid.best_params_
# print("ADA Params%s, Score%s"%(ada_grid.best_params_,ada_grid.best_score_))
# # ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")

# ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)

# # ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))

# kernel_list = [ker_rq]

# gpc_grid = GridSearchCV(
#   estimator = GaussianProcessClassifier(),
#     param_grid = {"kernel": kernel_list,
#               "optimizer": ["fmin_l_bfgs_b"],
#               "n_restarts_optimizer": [1],
#               "copy_X_train": [True]},
#     cv = kf,
#     scoring = "f1"    
# )

# gpc_grid.fit(x_train_scaled,y_train)
# print(gpc_grid.cv_results_)
# gpc_params = gpc_grid.best_params_
# print("GPC Params%s, Score%s"%(gpc_grid.best_params_,gpc_grid.best_score_))
# rf_grid = GridSearchCV(
#   estimator = RandomForestClassifier(warm_start=True,max_features='sqrt'),
#     param_grid = {
#         "n_estimators": [100, 200],
#         "max_depth": [2,3,4,5,6],
#         "min_samples_leaf": [1,2,3], 
#         'max_features': ['log2']
#     },
#     cv = kf,
#     scoring = "f1"    
# )
# rf_grid.fit(x_train,y_train)
# rf_params = rf_grid.best_params_
# print("RF Params%s, Score%s"%(rf_grid.best_params_,rf_grid.best_score_))
# from scipy.stats import expon

# C_distr = [0.01, 0.05, 0.09, 0.1, 0.9, 1]

# lr_grid = GridSearchCV(
#     estimator = LogisticRegression(),
#     param_grid = {'penalty': ['l1'], 'solver': [ 'saga'], 'C': C_distr},
#     cv = kf,
#     scoring = "f1"    
# )
# lr_grid.fit(x_train_scaled,y_train)
# lr_params = lr_grid.best_params_
# print("LR Params%s, Score%s"%(lr_grid.best_params_,lr_grid.best_score_))
# from sklearn.model_selection import StratifiedKFold

# skf_xgb = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)


# params_xgb = {
#          'min_child_weight': [1, 2,3, 5],
#         'subsample': [0.7, 0.8, 1.0],
#         'colsample_bytree': [0.7, 0.8, 0.9],
#         'max_depth': [3, 4, 5, 7],
#         }

# xgb_model_l1 = xgb.XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
#                     silent=True, nthread=1)

# xgb_l1_grid = RandomizedSearchCV(xgb_model_1, param_distributions=params_xgb,scoring='f1', 
#                                     cv=skf_1.split(x_train_scaled,y_train), verbose=3, random_state=1001)

# start_time = timer(None) # timing starts from this point for "start_time" variabl
# xgb_l1_grid.fit(x_train_scaled,y_train)
# timer(start_time) # timing ends here for "start_time" variable
# print(xgb_l1_grid.cv_results_)
# xgb_l1_params = xgb_l1_grid.best_params_
# print("XGB1 Params%s, Score%s"%(xgb_l1_grid.best_params_,xgb_l1_grid.best_score_))
knc_params = {'leaf_size': 2, 'n_neighbors': 15, 'p': 1}
svc_params = {'C': 2, 'gamma': 0.01}
et_params = {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 4, 'n_estimators': 500}
mlp_params = {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 9, 'max_iter': 500, 'random_state': 3, 'solver': 'lbfgs'}
ada_params = {'algorithm': 'SAMME.R', 'learning_rate': 0.05, 'n_estimators': 500}
gpc_params = {'copy_X_train': True, 'kernel': 1**2 * RationalQuadratic(alpha=0.1, length_scale=1), 'n_restarts_optimizer': 1, 'optimizer': 'fmin_l_bfgs_b'}
rf_params = {'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 2, 'n_estimators': 100}
lr_params = {'solver':'sag', 'tol':1e-1, 'C': 1.e4 / x_train.shape[0]}
xgb_l1_params = {'subsample': 1.0, 'min_child_weight': 5, 'max_depth': 4, 'colsample_bytree': 0.7}
# Stack Layer 1
knc = SklearnWrapper(clf=KNeighborsClassifier, params=knc_params)
svc = SklearnWrapper(clf=SVC, params=svc_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, params=et_params)
mlp = SklearnWrapper(clf=MLPClassifier, params=mlp_params)
ada = SklearnWrapper(clf=AdaBoostClassifier, params=ada_params)
gpc = SklearnWrapper(clf=GaussianProcessClassifier, params=gpc_params)
rf = SklearnWrapper(clf=RandomForestClassifier, params=rf_params)
lr = SklearnWrapper(clf=LogisticRegression, params=lr_params)
xg_l1 = XgbWrapper(params=xgb_l1_params)

knc_oof_train_l1f1,knc_oof_cv_l1f1,knc_oof_test_l1f1 = get_oof(knc,x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
svc_oof_train_l1f1,svc_oof_cv_l1f1,svc_oof_test_l1f1 = get_oof(svc,x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
et_oof_train_l1f1,et_oof_cv_l1f1,et_oof_test_l1f1 = get_oof(et, x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
mlp_oof_train_l1f1,mlp_oof_cv_l1f1,mlp_oof_test_l1f1 = get_oof(mlp, x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
ada_oof_train_l1f1,ada_oof_cv_l1f1,ada_oof_test_l1f1 = get_oof(ada, x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
gpc_oof_train_l1f1,gpc_oof_cv_l1f1,gpc_oof_test_l1f1 = get_oof(gpc,x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
rf_oof_train_l1f1,rf_oof_cv_l1f1,rf_oof_test_l1f1 = get_oof(rf,x_train, y_train, x_cv, x_test)
lr_oof_train_l1f1,lr_oof_cv_l1f1,lr_oof_test_l1f1 = get_oof(lr,x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
%%capture capt
xg_l1_oof_train_l1f1,xg_l1_oof_cv_l1f1,xg_l1_oof_test_l1f1 = get_oof(xg_l1,x_train_scaled, y_train, x_cv_scaled, x_test_scaled)
# performance of layer 0 inputs via various models on cv data
print("KNC-CV f1: {}".format(f1s_ec(y_cv, knc_oof_cv_l1f1)))
print("SVC-CV f1: {}".format(f1s_ec(y_cv, svc_oof_cv_l1f1)))
print("ET-CV f1: {}".format(f1s_ec(y_cv, et_oof_cv_l1f1)))
print("MLP-CV f1: {}".format(f1s_ec(y_cv, mlp_oof_cv_l1f1)))
print("ADA-CV f1: {}".format(f1s_ec(y_cv, ada_oof_cv_l1f1)))
print("GPC-CV f1: {}".format(f1s_ec(y_cv, gpc_oof_cv_l1f1)))
print("RF-CV f1: {}".format(f1s_ec(y_cv, rf_oof_cv_l1f1)))
print("LR-CV f1: {}".format(f1s_ec(y_cv, lr_oof_cv_l1f1)))
print("XGB-CV f1: {}".format(f1s_ec(y_cv,xg_l1_oof_cv_l1f1)))
# preparing layer 2 inputs:

x_train_l2f_i = np.concatenate((lr_oof_train_l1f1.astype(int),ada_oof_train_l1f1.astype(int),
                                gpc_oof_train_l1f1.astype(int)), axis=1)
x_cv_l2f_i = np.concatenate((lr_oof_cv_l1f1.astype(int),ada_oof_cv_l1f1.astype(int), 
                             gpc_oof_cv_l1f1.astype(int)), axis=1)
x_test_l2f_i = np.concatenate((lr_oof_test_l1f1.astype(int),ada_oof_test_l1f1.astype(int),
                               gpc_oof_test_l1f1.astype(int)), axis=1)
print("%s, %s, %s"%(x_train_l2f_i.shape, x_cv_l2f_i.shape, x_test_l2f_i.shape))
# fitting second layer input via LGBM

params_init_l = {}
params_init_l['learning_rate'] = 0.02
params_init_l['boosting_type'] = 'gbdt'
params_init_l['objective'] = 'binary'
params_init_l['metric'] = 'binary_logloss'
params_init_l['sub_feature'] = 0.5
params_init_l['num_leaves'] = 255
params_init_l['num_trees '] = 500
params_init_l['min_data'] = 50
params_init_l['max_depth'] = 10
params_init_l['num_threads'] = 16
params_init_l['min_sum_hessian_in_leaf '] = 100


lg_layer = LgbWrapper(params=params_init_l)
lg_layer.train(x_train_l2f_i, y_train)
print(f1s_e(y_cv,lg_layer.predict(x_cv_l2f_i)))
# fit and test on LGBM directly on layer 0

params_init = {}
params_init['learning_rate'] = 0.1
params_init['boosting_type'] = 'gbdt'
params_init['objective'] = 'binary'
params_init['metric'] = 'binary_logloss'
params_init['sub_feature'] = 0.5
params_init['num_leaves'] = 255
params_init['min_data'] = 50
params_init['max_depth'] = 10
params_init['num_threads'] = 16
params_init['min_sum_hessian_in_leaf '] = 16


lg = LgbWrapper(params=params_init)
lg.train(x_train_scaled, y_train)
print(f1s_e(y_cv,lg.predict(x_cv_scaled)))
# Generate Submission File for top prictions for cv
predictions_lg_direct = lg.predict(x_test_scaled).astype(int)
predictions_lg_layer = lg_layer.predict(x_test_l2f_i).astype(int)
predictions_ada = ada_oof_test_l1f1.astype(int)
StackingSubmission_lg_direct = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': predictions_lg_direct })
StackingSubmission_lg_layer = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': predictions_lg_layer })
StackingSubmission_ada = pd.DataFrame({ 'PassengerId': test.index,
                            'Survived': predictions_ada.reshape((418,)) })
StackingSubmission_lg_direct.to_csv("Submission_lg_direct_1.csv", index=False)
StackingSubmission_lg_layer.to_csv("StackingSubmission_lg_layer_1.csv", index=False)
StackingSubmission_ada.to_csv("StackingSubmission_ada_1.csv", index=False)