import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

import lightgbm as lgb

import xgboost as xgb

from sklearn.decomposition import PCA,KernelPCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm

import tensorflow as tf

import keras

from lightgbm import LGBMClassifier





# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold



from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

from sklearn import preprocessing
train = pd.read_csv("../input/india-ml-hiring-hackathon-2019/train.csv")

test = pd.read_csv("../input/india-ml-hiring-hackathon-2019/test.csv")

sub = pd.read_csv("../input/india-ml-hiring-hackathon-2019/sample_submission.csv")
# I have used pandans profiling for EDA , based on that we removed number_of_borrowers 

train = train.drop(['loan_id','number_of_borrowers'],axis=1)

test = test.drop(['loan_id','number_of_borrowers'],axis=1)
target = train['m13']

train=train.drop('m13',axis=1)
# Lets try something



train.head()
#train['LTV_by_DTI'] = train['loan_to_value'] / train['debt_to_income_ratio']

#test['LTV_by_DTI'] = test['loan_to_value'] / test['debt_to_income_ratio']
train = pd.get_dummies(train)

test = pd.get_dummies(test)
# Align train and test



train_labels = target



# Align the training and testing data, keep only columns present in both dataframes

train_df, test_df = train.align(test, join = 'inner', axis = 1)



# Add the target back in

train_df['m13'] = train_labels



print('Training Features shape: ', train_df.shape)

print('Testing Features shape: ', test_df.shape)
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
train_df = train_df.reindex(

    np.random.permutation(train_df.index))
y = train_df['m13']

X = train_df.drop('m13',axis=1)
from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_res, y_res = tl.fit_resample(X_res, y_res)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=10,stratify=y_res)
from sklearn.metrics import f1_score



def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_hat), True
def run_lgb(X_train, X_test, y_train, y_test, test_df):

    params = {

        "objective" : "binary",

       "n_estimators":10000,

       "reg_alpha" : 2.0,

       "reg_lambda":2.1,

       "n_jobs":-1,

       "colsample_bytree":.8,

       "min_child_weight":0.8,

       "subsample":0.8715623,

       "min_data_in_leaf":20,

       "nthread":4,

       "metric" : "f1",

       "num_leaves" : 100,

       "learning_rate" : 0.01,

       "verbosity" : -1,

       "seed": 120,

       "max_bin":60,

       'max_depth':15,

       'min_gain_to_split':.0222415,

       'scale_pos_weight':1

    }

    

    lgtrain = lgb.Dataset(X_train, label=y_train)

    lgval = lgb.Dataset(X_test, label=y_test)

    evals_result = {}

    model = lgb.train(params, lgtrain, 10000, 

                      valid_sets=[lgtrain, lgval], 

                      early_stopping_rounds=100, 

                      verbose_eval=100, 

                      evals_result=evals_result,feval=lgb_f1_score)

    

    pred_test_y = model.predict(test_df, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result
pred_test, model, evals_result = run_lgb(X_train, X_test, y_train, y_test, test_df)

print("LightGBM Training Completed...")
sub['m13'] = pred_test
sub['m13'] = sub['m13'].apply(lambda x : 1 if (x>=0.40) else 0)
sub['m13'].sum()
sub.to_csv('subm.csv',index=False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np 

import base64 
def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()) 

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>' 

    html = html.format(payload=payload,title=title,filename=filename) 

    return HTML(html)
create_download_link(sub)
!pip install catboost
import catboost



class ModelOptimizer:

    best_score = None

    opt = None

    

    def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

        self.model = model

        self.X_train = X_train

        self.y_train = y_train

        self.categorical_columns_indices = categorical_columns_indices

        self.n_fold = n_fold

        self.seed = seed

        self.early_stopping_rounds = early_stopping_rounds

        self.is_stratified = is_stratified

        self.is_shuffle = is_shuffle

        

        

    def update_model(self, **kwargs):

        for k, v in kwargs.items():

            setattr(self.model, k, v)

            

    def evaluate_model(self):

        pass

    

    def optimize(self, param_space, max_evals=10, n_random_starts=2):

        start_time = time.time()

        

        @use_named_args(param_space)

        def _minimize(**params):

            self.model.set_params(**params)

            return self.evaluate_model()

        

        opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

        best_values = opt.x

        optimal_values = dict(zip([param.name for param in param_space], best_values))

        best_score = opt.fun

        self.best_score = best_score

        self.opt = opt

        

        print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

        print('updating model with optimal values')

        self.update_model(**optimal_values)

        plot_convergence(opt)

        return optimal_values

    

class CatboostOptimizer(ModelOptimizer):

    def evaluate_model(self):

        validation_scores = catboost.cv(

        catboost.Pool(self.X_train, 

                      self.y_train, 

                      cat_features=self.categorical_columns_indices),

        self.model.get_params(), 

        nfold=self.n_fold,

        stratified=self.is_stratified,

        seed=self.seed,

        early_stopping_rounds=self.early_stopping_rounds,

        shuffle=self.is_shuffle,

        verbose=100,

        plot=False)

        self.scores = validation_scores

        test_scores = validation_scores.iloc[:, 2]

        best_metric = test_scores.max()

        return 1 - best_metric
!pip install scikit-optimize
from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.utils import use_named_args

from skopt.plots import plot_convergence

import time
import matplotlib.pyplot as plt

%matplotlib inline

from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

m=CatBoostClassifier(n_estimators=3000,random_state=1994,eval_metric='AUC',max_depth=12,learning_rate=0.029,od_wait=50

                     ,l2_leaf_reg=5,bagging_temperature=0.85,random_strength=100,

                     use_best_model=True)

m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=100,verbose=100)
test_pred = m.predict(test_df)
test_pred.sum()