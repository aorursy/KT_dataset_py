%load_ext autoreload

%autoreload 2



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

%matplotlib inline



import pandas as pd



import numpy as np



import sklearn

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

from sklearn import metrics



import xgboost as xgb

from xgboost.sklearn import XGBClassifier



from IPython.display import display





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return [('gini', gini_score)]
PATH = "../input/train.csv"

data_raw= pd.read_csv(f'{PATH}', low_memory=False)
def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df)

display_all(data_raw.head(5))
# Describe the data set

display_all(data_raw.describe(include='all'))
# Distribution of target variable

import matplotlib.pyplot as plt

plt.hist(data_raw['target'])

plt.show()



print('Percentage of claims filed :' , str(np.sum(data_raw['target'])/data_raw.shape[0]*100), '%')
nas = np.sum(data_raw == -1)/len(data_raw) *100

print("The percentage of missing values is")

print (nas[nas>0].sort_values(ascending = False))
# make a copy of the initial dataset

data_clean = data_raw.copy()

#data_clean.columns

cat_cols = [c for c in data_clean.columns if c.endswith('cat')]

for column in cat_cols:

    temp=pd.get_dummies(data_clean[column], prefix=column, prefix_sep='_')

    data_clean=pd.concat([data_clean,temp],axis=1)

    data_clean=data_clean.drop([column],axis=1)



print('data_clean shape is:',data_clean.shape)
# Impute missing values with medians



num_cols = ['ps_reg_03','ps_car_14', 'ps_car_11', 'ps_car_12' ]



for n in num_cols:

    dummy_name = str(n) + 'NA'

    data_clean[dummy_name] = (data_clean[n]==-1).astype(int)

    med = data_clean[data_clean[n]!=-1][n].median()

    data_clean.loc[data_clean[n]==-1,n] = med

    



    
#Make transformation to ps_car_13, as suggested here: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41489

data_clean['ps_car_13_trans'] = round(data_clean['ps_car_13']* data_clean['ps_car_13']* 90000,2)
sub_df_0= data_clean[(data_clean['target']==0)]

sub_df_1= data_clean[(data_clean['target']==1)]

sub_df_1.shape
sub_df = sub_df_0.sample(frac = 0.25, random_state = 42)

data_sub = pd.concat([sub_df_1,sub_df])
# First split the data into training and validation (test) sets

training_features, test_features, \

training_target, test_target, = train_test_split(data_sub.drop(['id','target'], axis=1),

                                               data_sub['target'],

                                               test_size = .2,

                                               random_state=12)



# Now further split the training test into training and validation to 

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,

                                                  test_size = .2,

                                                  random_state=12)
xgb_params = {'eta': 0.02, 

              'max_depth': 6, 

              'subsample': 1.0, 

              'colsample_bytree': 0.3,

              'min_child_weight': 1,

              'objective': 'binary:logistic', 

              'eval_metric': 'auc', 

              'seed': 99, 

              'silent': True}

d_train = xgb.DMatrix(x_train, y_train)

d_valid = xgb.DMatrix(x_val,y_val)

d_test = xgb.DMatrix(test_features)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#model = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)

#print(model.best_score, model.best_iteration, model.best_ntree_limit)
#results = {'best_score':[],'best_iter':[],'best_ntree_limit':[]}
results = {'eta':[],'best_score':[],'best_ntree_limit':[]}

for e in [0.01, 0.02, 0.03,0.05,0.1,0.2]:

    xgb_params = {'eta': e, 

                  'max_depth': 6, 

                  'subsample': 1.0, 

                  'colsample_bytree': 0.3,

                  'min_child_weight': 1,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}



    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

   # m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)

    #results['best_score'].append(m.best_score)

    #results['best_ntree_limit'].append(m.best_ntree_limit)

    #results['eta'].append(e)

    

#print('eta:',results['eta'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])
results = {'max_depth':[],'best_score':[],'best_ntree_limit':[]}

for md in range(3,9,1):

    xgb_params = {'eta': 0.03, 

                  'max_depth': md, 

                  'subsample': 1.0, 

                  'colsample_bytree': 0.3,

                  'min_child_weight': 1,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}



    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=200)

    #results['best_score'].append(m.best_score)

    #results['best_ntree_limit'].append(m.best_ntree_limit)

    #results['max_depth'].append(md)

    

#print('max_depth:',results['max_depth'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])
results = {'min_child_w':[],'best_score':[],'best_ntree_limit':[]}

for mcw in range(1,10,1):

    xgb_params = {'eta': 0.03, 

                  'max_depth': 6, 

                  'subsample': 1.0, 

                  'colsample_bytree': 0.3,

                  'min_child_weight': mcw,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}



    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)

    #results['best_score'].append(m.best_score)

    #results['best_ntree_limit'].append(m.best_ntree_limit)

    #results['min_child_w'].append(mcw)

    

#print('min_child_w:',results['min_child_w'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])
results = {'colsample_bytree':[],'best_score':[],'best_ntree_limit':[]}

for cst in [0.3,0.4,0.5]:

    xgb_params = {'eta': 0.03, 

                  'max_depth': 6, 

                  'subsample': 1.0, 

                  'colsample_bytree': cst,

                  'min_child_weight': 2,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}



    #watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)

    #results['best_score'].append(m.best_score)

    #results['best_ntree_limit'].append(m.best_ntree_limit)

    #results['colsample_bytree'].append(cst)

    

#print('colsample_bytree:',results['colsample_bytree'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])
training_features, test_features, \

training_target, test_target, = train_test_split(data_clean.drop(['id','target'], axis = 1),

                                               data_clean['target'],

                                               test_size = .2,

                                               random_state=12)



# Now further split the training test into training and validation to 

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,

                                                  test_size = .2,

                                                  random_state=12)
#Final model

xgb_params = {'eta': 0.03, 

                  'max_depth': 6, 

                  'subsample': 1.0, 

                  'colsample_bytree': 0.3,

                  'min_child_weight': 2,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}

d_train = xgb.DMatrix(x_train, y_train)

d_valid = xgb.DMatrix(x_val,y_val)



#watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#model = xgb.train(xgb_params, d_train, 392,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)
#Feature importance

#feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)

#feat_imp.plot(kind='bar', title='Feature Importances')

#feat_imp[:60]
#to_keep = feat_imp[feat_imp>=100].index

#df = data_clean[to_keep]

#x_train = df

#y_train = data_clean['target']
xgb_params = {'eta': 0.03, 

                  'max_depth': 6, 

                  'subsample': 1.0, 

                  'colsample_bytree': 0.3,

                  'min_child_weight': 2,

                  'objective': 'binary:logistic', 

                  'seed': 99, 

                  'silent': True}

#xgb.DMatrix(x_train[predictors].values, label=y_train.values)

#d_train = xgb.DMatrix(x_train, y_train)

#d_valid = xgb.DMatrix(x_val,y_val)



#watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#model = xgb.train(xgb_params, d_train, 392, feval=gini_xgb, maximize=True, verbose_eval=False)
#Download and transform test set

test = pd.read_csv('../input/test.csv', low_memory=False)
test.head(5)
nas = np.sum(test == -1)/len(test) *100

print("The percentage of missing values is")

print (nas[nas>0].sort_values(ascending = False))
#Transformations

test_clean = test.copy()



cat_cols = [c for c in test_clean.columns if c.endswith('cat')]



# Creating dummies for missing values in categorical features

for column in cat_cols:

    temp=pd.get_dummies(test_clean[column], prefix=column, prefix_sep='_')

    test_clean=pd.concat([test_clean,temp],axis=1)

    test_clean=test_clean.drop([column],axis=1)



print('test_clean shape is:',test_clean.shape)



    

# Impute missing values with medians



num_cols = ['ps_reg_03','ps_car_14', 'ps_car_11']



for n in num_cols:

    dummy_name = str(n) + 'NA'

    test_clean[dummy_name] = (test_clean[n]==-1).astype(int)

    med = test_clean[test_clean[n]!=-1][n].median()

    test_clean.loc[test_clean[n]==-1,n] = med

    print(n,np.sum(data_clean[n] == -1)/len(data_clean) *100)

#x_test = test_clean[to_keep]

#dtest = xgb.DMatrix(x_test)

#xgb_pred = model.predict(dtest)



#id_test = test_clean['id'].values

#output = pd.DataFrame({'id': id_test, 'target': xgb_pred})
