import numpy as np

import pandas as pd



import random

random.seed(28)

np.random.seed(28)



import matplotlib.pyplot as plt

from matplotlib_venn import venn2



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)

import os

import copy



pd.options.display.precision = 15



from collections import defaultdict

import lightgbm as lgb

import xgboost as xgb

import time

from collections import Counter

import datetime

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

#import eli5

import shap

from IPython.display import HTML

import json



import matplotlib.pyplot as plt

%matplotlib inline

import os



pd.set_option('max_rows', 500)

import re



pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', '{:20,.2f}'.format)

pd.set_option('display.max_colwidth', -1)



np.random.seed(2206)
train = pd.read_csv("../input/widsdatathon2020/training_v2.csv")

samplesubmission = pd.read_csv("../input/widsdatathon2020/samplesubmission.csv")

test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")

dictionary = pd.read_csv("../input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")

solution_template = pd.read_csv("../input/widsdatathon2020/solution_template.csv")



print('train ' , train.shape)

print('test ' , test.shape)

print('samplesubmission ' , samplesubmission.shape)

print('solution_template ' , solution_template.shape)

print('dictionary ' , dictionary.shape)
dico = pd.DataFrame(dictionary.T.head(6))

dico.columns=list(dico.loc[dico.index == 'Variable Name'].unstack())

dico = dico.loc[dico.index != 'Variable Name']

dico.columns

train_stat = pd.DataFrame(train.describe())

train_stat2 = pd.concat([dico,train_stat],axis=0)

train_stat2.head(20)
train_stat2.T.head(200)
# Missing Values

train.isna().sum()
# function to evaluate the score of our model

def eval_auc(pred,real):

    false_positive_rate, recall, thresholds = roc_curve(real, pred)

    roc_auc = auc(false_positive_rate, recall)

    return roc_auc    
# a wrapper class  that we can have the same ouput whatever the model we choose

class Base_Model(object):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True,ps={}):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'hospital_death'

        self.cv = self.get_cv()

        self.verbose = verbose

#         self.params = self.get_params()

        self.params = self.set_params(ps)

        self.y_pred, self.score, self.model , self.oof_pred = self.fit()

        

    def train_model(self, train_set, val_set):

        raise NotImplementedError

        

    def get_cv(self):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])

    

    def get_params(self):

        raise NotImplementedError

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError

        

    def convert_x(self, x):

        return x

        

    def fit(self):

        oof_pred = np.zeros((len(self.train_df), ))

        y_pred = np.zeros((len(self.test_df), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits



            print('Partial score of fold {} is: {}'.format(fold,eval_auc(oof_pred[val_idx],y_val) ))

        #print(oof_pred, self.train_df[self.target].values)

        loss_score = eval_auc(oof_pred,self.train_df[self.target].values) 

        if self.verbose:

            print('Our oof AUC score is: ', loss_score)

        return y_pred, loss_score, model , oof_pred
#we choose to try a LightGbM using the Base_Model class

class Lgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)

        val_set   = lgb.Dataset(x_val,    y_val,  categorical_feature=self.categoricals)

        return train_set, val_set

        

    def get_params(self):

        params = {'n_estimators':5000,

                  'boosting_type': 'gbdt',

                  'objective': 'binary',

                  'metric': 'auc',

                  'subsample': 0.75,

                  'subsample_freq': 1,

                  'learning_rate': 0.1,

                  'feature_fraction': 0.9,

                  'max_depth': 15,

                  'lambda_l1': 1,  

                  'lambda_l2': 1,

                  'early_stopping_rounds': 100,

                  #'is_unbalance' : True ,

                  'scale_pos_weight' : 3,

                  'device': 'gpu',

                  'gpu_platform_id': 0,

                  'gpu_device_id': 0,

                  'num_leaves': 31

                    }

        return params

    def set_params(self,ps={}):

        params = self.get_params()

        if 'subsample_freq' in ps:

            params['subsample_freq']=int(ps['subsample_freq'])

            params['learning_rate']=ps['learning_rate']

            params['feature_fraction']=ps['feature_fraction']

            params['lambda_l1']=ps['lambda_l1']

            params['lambda_l2']=ps['lambda_l2']

            params['scale_pos_weight']=ps['scale_pos_weight']

            params['max_depth']=int(ps['max_depth'])

            params['subsample']=ps['subsample']

            params['num_leaves']=int(ps['num_leaves'])

            params['min_split_gain']=ps['min_split_gain']

#             params['min_child_weight']=ps['min_child_weight']

        

        return params  
def plot_importances(importances_, plot_name):

    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()

    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])

    plt.figure(figsize=(18, 44))

    data_imp = importances_.sort_values('mean_gain', ascending=False)

    sns.barplot(x='gain', y='feature', data=data_imp[:300])

    plt.tight_layout()

    plt.savefig('{}.png'.format(plot_name))

    plt.show()
# Replace values



print('Replacing: {}'.format('hospital_admit_source'))



replace_hospital_admit_source =  {'Other ICU': 'ICU',

                                  'ICU to SDU':'SDU', 

                                  'Step-Down Unit (SDU)': 'SDU', 

                                  'Other Hospital':'Other',

                                  'Observation': 'Recovery Room',

                                  'Acute Care/Floor': 'Acute Care'}

train['hospital_admit_source'].replace(replace_hospital_admit_source, inplace=True)

test['hospital_admit_source'].replace(replace_hospital_admit_source, inplace=True)



#combined_dataset['icu_type'] = combined_dataset['icu_type'].replace({'CCU-CTICU': 'Grpd_CICU', 'CTICU':'Grpd_CICU', 'Cardiac ICU':'Grpd_CICU'})



print('Replacing: {}'.format('apache_2_bodysystem'))



replace_apache_2_bodysystem =  {'Undefined diagnoses': 'Undefined Diagnoses'}

train['apache_2_bodysystem'].replace(replace_apache_2_bodysystem, inplace=True)

test['apache_2_bodysystem'].replace(replace_apache_2_bodysystem, inplace=True)
#we are going to drop these columns because we dont want our ML model to be bias toward these consideration

#(we also remove the target and the ids.)

to_drop = ['gender','ethnicity' ,'encounter_id', 'patient_id',  'hospital_death']



# this is a list of features that look like to be categorical

categoricals_features = ['hospital_id','ethnicity','gender','hospital_admit_source','icu_admit_source',

                         'icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem']

categoricals_features = [col for col in categoricals_features if col not in to_drop]



# this is the list of all input feature we would like our model to use 

features = [col for col in train.columns if col not in to_drop ]

print('numerber of features ' , len(features))

print('shape of train / test ', train.shape , test.shape)
# categorical feature need to be transform to numeric for mathematical purpose.

# different technics of categorical encoding exists here we will rely on our model API to deal with categorical

# still we need to encode each categorical value to an id , for this purpose we use LabelEncoder



print('Transform all String features to category.\n')

for usecol in categoricals_features:

    train[usecol] = train[usecol].astype('str')

    test[usecol] = test[usecol].astype('str')

    

    #Fit LabelEncoder

    le = LabelEncoder().fit(

            np.unique(train[usecol].unique().tolist()+

                      test[usecol].unique().tolist()))



    #At the end 0 will be used for dropped values

    train[usecol] = le.transform(train[usecol])+1

    test[usecol]  = le.transform(test[usecol])+1

    

    train[usecol] = train[usecol].replace(np.nan, 0).astype('int').astype('category')

    test[usecol]  = test[usecol].replace(np.nan, 0).astype('int').astype('category')
def adversarial_validation(train, test, features):

    tr_data   = train.copy()

    tst_data = test.copy()

    tr_data['target']  = 0 

    tst_data['target'] = 1

    av_data = pd.concat([tr_data, tst_data], axis = 0)

    av_data.reset_index(drop = True)        

    params = {

            'learning_rate': 0.1, 

            'seed': 50,

            'objective':'binary',

            'boosting_type':'gbdt',

            'metric': 'auc',

        }    

    # define a KFold strategy

    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)

    target = 'target'

    oof_pred = np.zeros(len(av_data))

    important_features = pd.DataFrame()

    fold_auc = []    

    

    for fold, (tr_ind, val_ind) in enumerate(kf.split(av_data, av_data[target])) :

        print('Fold {}'.format(fold + 1))

        x_train, x_val = av_data[features].iloc[tr_ind], av_data[features].iloc[val_ind]

        y_train, y_val = av_data[target].iloc[tr_ind], av_data[target].iloc[val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set   = lgb.Dataset(x_val, y_val)

        

        model = lgb.train(params, train_set, num_boost_round = 1000, early_stopping_rounds = 20, valid_sets = [train_set, val_set], verbose_eval = 100)

        

        fold_importance = pd.DataFrame()

        fold_importance['feature'] = features

        fold_importance['gain'] = model.feature_importance()

        important_features = pd.concat([important_features, fold_importance], axis = 0)

        

        oof_pred[val_ind] = model.predict(x_val)

        fold_auc.append(metrics.roc_auc_score(y_train, model.predict(x_train)))

        

    print('Our mean train roc auc score is :', np.mean(fold_auc))

    print('Our oof roc auc score is :', metrics.roc_auc_score(av_data[target], oof_pred))

    return important_features
# run the adversatial model with all the feature we used :

    

adversarial_features = adversarial_validation(train, test, features)
# AUC is almost perfect so we can expect that some feature are perfectly different between train / test



adversarial_features = adversarial_features[['gain', 'feature']].groupby('feature').mean().reset_index()

adversarial_features= adversarial_features.sort_values('gain', ascending=False)



plot_importances(adversarial_features, 'importances-lgb-v6')
# So icu_id columns seems to be the feature that dominate the feature importance for the adversarial 

# validation model, so it is likely to be totally different between train and test, 

# lets check the distribution of the top features :



def plot_differente_between_train_test(adversarial_features):

    import warnings

    warnings.filterwarnings("ignore")

    warnings.simplefilter(action='ignore', category=UserWarning)

    i=0

    for index, row in adversarial_features.sort_values(by=['gain'],ascending=False).iterrows():  

        column=row['feature']

        if i< 10:

                print(column,i,"gain :",row['gain'])

                df1      = train.copy()

                df2      = test.copy()



                fig = plt.figure(figsize=(20,4))

                sns.distplot(df1[column].dropna(),  color='yellow', label='train', kde=True); 

                sns.distplot(df2[column].dropna(),  color='violet', label='test', kde=True); 

                fig=plt.legend(loc='best')

                plt.xlabel(column, fontsize=12);

                plt.show()

                i=i+1



plot_differente_between_train_test(adversarial_features)
# it is .... Let's remove icu_id and see the results ..

adversarial_features2 = adversarial_validation(train, test, [ f for f in features if f not in ['icu_id'] ])
# Let`s check again the difference between train / test



adversarial_features2 = adversarial_features2[['gain', 'feature']].groupby('feature').mean().reset_index()

adversarial_features2= adversarial_features2.sort_values('gain', ascending=False)



plot_importances(adversarial_features2, 'importances-lgb-v6')
plot_differente_between_train_test(adversarial_features2)
# hospital_id seems to be also from a different distribution. 

# We can check it directly, obviously only few hospital are common to both dataset ..



common_id  = list([id for id in train['hospital_id'].unique() if id in test['hospital_id'].unique() ])

id_only_in_train  = [id for id in train['hospital_id'].unique() if id not in test['hospital_id'].unique() ]

id_only_in_test   = [id for id in test['hospital_id'].unique()  if id not in train['hospital_id'].unique() ]

count_common_train = train.loc[train['hospital_id'].isin(common_id)].shape[0]

count_common_test  = test.loc[test['hospital_id'].isin(common_id)].shape[0]



count_train = train.loc[train['hospital_id'].isin(id_only_in_train)].shape[0]

count_test  = test.loc[test['hospital_id'].isin(id_only_in_test)].shape[0]



 

fig = plt.figure(figsize=(20,6))

venn2(subsets = (count_train,  count_test, count_common_train+count_common_test), set_labels = ('Hospital only in train', 'Hospital only in test'),set_colors=('purple', 'yellow'), alpha = 0.7);

plt.show()
# Let's do an ultimate try without 'icu_id','hospitaadversarial_features3 = adversarial_validation(train, test, [ f for f in features if f not in ['icu_id','hospital_id'] ])l_id'

adversarial_features3 = adversarial_validation(train, test, [ f for f in features if f not in ['icu_id','hospital_id'] ])
# I leave it to you to see what you can do with other features..

adversarial_features3 = adversarial_features3[['gain', 'feature']].groupby('feature').mean().reset_index()

adversarial_features3= adversarial_features3.sort_values('gain', ascending=False)



plot_importances(adversarial_features3, 'importances-lgb-v6')
plot_differente_between_train_test(adversarial_features3)
# Lets remove hospital_id and icu_id



print('Difference between train and teste> -- hospital_id: ')

print(len(list(set(train['hospital_id']) - set(test['hospital_id']))))



print('\nDifference between train and teste> -- icu_id: ')

print(len(list(set(train['icu_id']) - set(test['icu_id']))))





# Drop features with zero importance

print('\nLength train features: {}'.format(len(features)))

for feat_to_remove in ['icu_id', 'hospital_id']:

    if feat_to_remove in categoricals_features:

        print('Removing from categoricals_features....{}'.format(feat_to_remove))

        categoricals_features.remove(feat_to_remove)

    if feat_to_remove in features:

        print('Removing from features....{}'.format(feat_to_remove))

        features.remove(feat_to_remove)

    

print('\nNew length train features: {}'.format(len(features)))
# percentage of death , hopefully it s a bit unbalanced

train['hospital_death'].sum()/train['hospital_death'].count()
# You want Bayesian Optimization?



boll_BayesianOptimization = False

# boll_BayesianOptimization = True
%time



def LGB_Beyes(subsample_freq,

                    learning_rate,

                    feature_fraction,

                    max_depth,

                    lambda_l1,

                    lambda_l2,

                    scale_pos_weight,

                    subsample,

                    num_leaves,

                    min_split_gain):

#                     min_child_weight):

    params={}

    params['subsample_freq']=subsample_freq

    params['learning_rate']=learning_rate

    params['feature_fraction']=feature_fraction

    params['lambda_l1']=lambda_l1

    params['lambda_l2']=lambda_l2

    params['max_depth']=max_depth

    params['scale_pos_weight']=scale_pos_weight

    params['subsample']=subsample

    params['num_leaves']=num_leaves

    params['min_split_gain']=min_split_gain

   # params['min_child_weight']=min_child_weight

    

    

    lgb_model= Lgb_Model(train, test, features, categoricals=categoricals_features,ps=params)

    print('auc: ',lgb_model.score)

    return lgb_model.score



bounds_LGB = {

    'max_depth': (5, 17),

    'subsample': (0.5, 1),

    'num_leaves': (10, 45),

    'feature_fraction': (0.1, 1),

    'min_split_gain': (0.0, 0.1),

#     'min_child_weight': (1e-3, 50),

    'subsample_freq': (1, 10),

    'learning_rate': (0.005, 0.02),

    'lambda_l1': (0, 5),

    'lambda_l2': (0, 5),

    'scale_pos_weight': (1, 10)

}



# ACTIVATE it if you want to search for better parameter

if boll_BayesianOptimization: 

    LGB_BO = BayesianOptimization(LGB_Beyes, bounds_LGB, random_state=1029)

    import warnings

    init_points = 16

    n_iter = 16

    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')    

        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
if boll_BayesianOptimization and LGB_BO:

    print(LGB_BO.max['params'])
# params = {'feature_fraction': 0.9,

#  'lambda_l1': 1,

#  'lambda_l2': 1,

#  'learning_rate': 0.1,

#  'max_depth': 13,

#  'subsample_freq': 1,

#  'scale_pos_weight':1}



# Best Hyperparams from Bayesian Optimization in notebook lgb-v2

# params = {'feature_fraction': 0.524207414205945,

#  'lambda_l1': 4.171808735757517,

#  'lambda_l2': 4.6435328298317256,

#  'learning_rate': 0.007897539397989824,

#  'max_depth': 16.62053004755999,

#  'scale_pos_weight': 1.2199266532301127,

#  'subsample_freq': 1.0276518730971627}





# # Best Hyperparams from Bayesian Optimization in notebook lgb-v3

# params = {'feature_fraction': 0.524207414205945,

#  'lambda_l1': 4.171808735757517,

#  'lambda_l2': 4.6435328298317256,

#  'learning_rate': 0.007897539397989824,

#  'max_depth': 16.62053004755999,

#  'scale_pos_weight': 1.2199266532301127,

#  'subsample_freq': 1.0276518730971627}



# # Best Hyperparams from Bayesian Optimization in notebook lgb-v4

# params = {'feature_fraction': 0.5348508368206359,

#  'lambda_l1': 0.0009370993396629057,

#  'lambda_l2': 4.743745312344983,

#  'learning_rate': 0.012891827059322746,

#  'max_depth': 15.784155449197529,

#  'scale_pos_weight': 1.0325760631926175,

#  'subsample_freq': 1.0744384574974872}





# Best Hyperparams from Bayesian Optimization in notebook lgb-v5

# params = {'feature_fraction': 0.3245039721724266,

#  'lambda_l1': 1.416727346446085,

#  'lambda_l2': 2.779776916582821,

#  'learning_rate': 0.006854369969433722,

#  'max_depth': 16.673905691676964,

#  'min_split_gain': 0.05643417986130283,

#  'num_leaves': 44.8672896759208,

#  'scale_pos_weight': 1.1577974342088542,

#  'subsample': 0.630352165410007,

#  'subsample_freq': 1.2158674819047501}



# Best Hyperparams from Bayesian Optimization in notebook lgb-v6 -- BEST MODEL

params = {

   "feature_fraction":0.1743912077888097,

   "lambda_l1":2.838660318794291,

   "lambda_l2":0.292397357257721,

   "learning_rate":0.012602188092427687,

   "max_depth":16.575351761228106,

   "min_split_gain":0.04631934372471113,

   "num_leaves":44.81666226482246,

   "scale_pos_weight":1.0897617979884857,

   "subsample":0.8260779721854892,

   "subsample_freq":1.2473380372944387

}
%time



if boll_BayesianOptimization: # ACTIVATE it if you want to search/use for better parameter

    lgb_model = Lgb_Model(train,test, features, categoricals=categoricals_features, ps= LGB_BO.max['params'])

else :

    lgb_model = Lgb_Model(train,test, features, categoricals=categoricals_features, ps=params)
imp_df = pd.DataFrame()

imp_df['feature'] = features

imp_df['gain']  = lgb_model.model.feature_importance(importance_type='gain')

imp_df['split'] = lgb_model.model.feature_importance(importance_type='split')
plot_importances(imp_df, 'importances-lgb-v6-lgb_model')
import shap

explainer   =  shap.TreeExplainer(lgb_model.model)

shap_values = explainer.shap_values(train[features].iloc[:1000,:])

shap.summary_plot(shap_values, train[features].iloc[:1000,:])
print('AUC Version 1: ', lgb_model.score)

#print('AUC:Version 2: ', lgb_model_v2.score)
test["hospital_death"] = lgb_model.y_pred

#test[["encounter_id","hospital_death"]].to_csv("submission6-lgb-v6.csv",index=False)



test[["encounter_id","hospital_death"]].head()