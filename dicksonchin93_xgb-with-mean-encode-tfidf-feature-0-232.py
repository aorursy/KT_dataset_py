# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
print('train data shape is :', tr.shape)
print('test data shape is :', te.shape)
data = pd.concat([tr, te], axis=0)
tr.head()
data.shape
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
data.activation_date = pd.to_datetime(data.activation_date)
tr.activation_date = pd.to_datetime(tr.activation_date)

data['day_of_month'] = data.activation_date.apply(lambda x: x.day)
data['day_of_week'] = data.activation_date.apply(lambda x: x.weekday())

tr['day_of_month'] = tr.activation_date.apply(lambda x: x.day)
tr['day_of_week'] = tr.activation_date.apply(lambda x: x.weekday())
data['char_len_title'] = data.title.apply(lambda x: len(str(x)))
data['char_len_desc'] = data.description.apply(lambda x: len(str(x)))
agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_month','day_of_week'];
for c in tqdm(agg_cols):
    gp = tr.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
data.head()
cate_cols = ['city',  'category_name', 'user_type',]
for c in cate_cols:
    data[c] = LabelEncoder().fit_transform(data[c].values)
from nltk.corpus import stopwords
stopWords = stopwords.words('russian')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
data['description'] = data['description'].fillna(' ')
tfidf = TfidfVectorizer(max_features=100, stop_words = stopWords)
tfidf_train = np.array(tfidf.fit_transform(data['description']).todense(), dtype=np.float16)
for i in range(100):
    data['tfidf_' + str(i)] = tfidf_train[:, i]
new_data = data.drop(['user_id','description','image','parent_category_name','region',
                      'item_id','param_1','param_2','param_3','title'], axis=1)
import gc
del data
del tr
del te
gc.collect()
from sklearn.model_selection import train_test_split
X = new_data.loc[new_data.activation_date<=pd.to_datetime('2017-04-07')]
X_te = new_data.loc[new_data.activation_date>=pd.to_datetime('2017-04-08')]

y = X['deal_probability']
X = X.drop(['deal_probability','activation_date'],axis=1)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=2018)
X_te = X_te.drop(['deal_probability','activation_date'],axis=1)

print(X_tr.shape, X_va.shape, X_te.shape)


#del X
#del y
#gc.collect()
'''
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBRegressor(
        n_jobs = 1,
        objective = 'regression',
        eval_metric = 'rmse',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
    
'''
import xgboost as xgb

params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'reg:logistic', 
          'eval_metric': 'rmse', 
          'random_state': 99, 
          'silent': True}

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)
del X_tr
del X_va
del y_tr
del y_va
gc.collect()

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(params, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=5)
X_te = xgb.DMatrix(X_te)
y_pred = model.predict(X_te, ntree_limit=model.best_ntree_limit)
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('xgb_with_mean_encode_and_nlp.csv', index=False)
sub.head()
from xgboost import plot_importance
import matplotlib.pyplot as plt
plot_importance(model)
plt.gcf().savefig('feature_importance_xgb.png')


