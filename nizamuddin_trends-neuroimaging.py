%%time

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import matplotlib.pyplot as plt

import seaborn as sns
# to_drop_67auc=['IC_20',

#          'IC_02',

#          'IC_05',

#          'IC_16',

#          'IC_10',

#          'IC_08',

#          'CBN(4)_vs_CON(37)',

#          'CBN(4)_vs_CON(38)',

#          'SCN(99)_vs_SCN(98)',

#          'DMN(23)_vs_CON(37)',

#          'DMN(40)_vs_CON(48)',

#          'DMN(17)_vs_DMN(40)',

#          'DMN(17)_vs_CON(88)',

#          'DMN(17)_vs_CON(33)',

#          'CON(79)_vs_SMN(54)',

#          'CON(55)_vs_SCN(45)',

#          'CON(88)_vs_SMN(54)',

#          'CON(83)_vs_CON(48)',

#          'CON(83)_vs_CON(67)',

#          'CON(83)_vs_CON(37)',

#          'CON(83)_vs_CON(33)',

#         ]

# to_drop=['IC_20',

#          'IC_02',

#          'IC_05',

#          'IC_16',

#          'IC_10',

#          'IC_08',

#          'CBN(4)_vs_CON(37)',

#          'CBN(4)_vs_CON(38)',

#          'SCN(99)_vs_SCN(98)',

#         ]



# len(to_drop)
# !pip install fastai2>/dev/null

# !pip install fast_tabnet>/dev/null

# from fastai2.basics import *

# from fastai2.tabular.all import *

# from fast_tabnet.core import *
# import numpy as np

# import pandas as pd

# import matplotlib.pyplot as plt



# from tqdm.notebook import tqdm

# import gc



# import lightgbm as lgb





# fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

# loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

# labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")



# fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

# df = fnc_df.merge(loading_df, on="Id")

# labels_df["is_train"] = True



# df = df.merge(labels_df, on="Id", how="left")



# targets = ['age','domain1_var1','domain1_var2','domain2_var1', 'domain2_var2']



# #imputing missing values in targets

# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors = 5, weights="distance")

# df[targets] = pd.DataFrame(imputer.fit_transform(df[targets]), columns = targets)



# test_df = df[df["is_train"] != True].copy()

# train_df = df[df["is_train"] == True].copy()



# train_df = train_df.drop(['is_train'], axis=1)

# test_df = test_df.drop(targets+['is_train'], axis=1)





# features=list(set(train_df.columns)-set(targets)-set(['Id']))





# #train_df[loading_features]=train_df[loading_features].pow(2)

# train_df[fnc_features]=train_df[fnc_features].mul(1/600)

# # train_df[fnc_features]=train_df[fnc_features].pow(2)



# #test_df[loading_features]=test_df[loading_features].pow(2)

# test_df[fnc_features]=test_df[fnc_features].mul(1/600)

# # test_df[fnc_features]=test_df[fnc_features].pow(2)









# #-------Normalizing------------------------

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])

# test_df[features] = scaler.transform(test_df[features])

# #----------------------------------------------------

# to_drop=['IC_20',

#          'IC_02',

#          'IC_05',

#          'IC_16',

#          'IC_10',

#          'IC_18'

#         ]

# train_df = train_df.drop(to_drop, axis=1)

# test_df = test_df.drop(to_drop, axis=1)

# print(train_df.shape,test_df.shape)

# print("Train and test dataframes contain Id column!!")

# def trends_scorer_multitask_scoring(y_true,y_preds):

#     '''

#     custom scoring function used for evaluation in this competition

#     '''



#     y_true=torch.tensor(y_true,requires_grad=True)

#     y_preds=torch.tensor(y_preds,requires_grad=True)

#     inp,targ = flatten_check(y_true,y_preds)

#     w = torch.tensor([.3, .175, .175, .175, .175],requires_grad=True)

#     op = torch.mean(torch.matmul(torch.abs(y_true-y_preds),w/torch.mean(y_true,axis=0)),axis=0)

#     return op



# def trends_scorer_multitask_scoring_gpu(y_true,y_preds):

#     '''

#     custom scoring function used for evaluation in this competition

#     '''

#     import numpy as np

#     y_true = y_true.cpu().detach().numpy()

#     y_preds= y_preds.cpu().detach().numpy()

#     w = np.array([.3, .175, .175, .175, .175])

#     op = np.mean(np.matmul(np.abs(y_true-y_preds),w/np.mean(y_true,axis=0)),axis=0)

#     return op
# from fastai2.layers import L1LossFlat,MSELossFlat 

# from torch.nn import SmoothL1Loss

# class SmoothMAELoss(torch.nn.Module):

#     '''

#     For use with GPU only

#     '''

#     def __init__(self,l1):

#         super().__init__()

#         self.l1=l1

        

#     def forward(self,y, y_hat):

#         loss = (1-self.l1)*SmoothL1Loss()(y, y_hat) + self.l1*L1LossFlat()(y, y_hat)

#         return loss
# def get_tabnet_data(df,train_val_idx):

#     targets=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']

#     features=list(set(df.columns)-set(targets))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     to = TabularPandas(

#         df=df,

#         procs=[Normalize],

#         cat_names=None,

#         cont_names=features,

#         y_names=targets,

#         y_block=TransformBlock(),

#         splits=train_val_idx,

#         do_setup=True,

#         device=device,

#         inplace=False,

#         reduce_memory=True,

#     )

#     return to,len(features),len(targets)

# def get_model(emb_szs,dls,n_features,n_labels):

#     model=TabNetModel(

#         emb_szs,

#         n_cont=n_features,

#         out_sz=n_labels,

#         embed_p=0.0,

#         y_range=None,

#         n_d=32,

#         n_a=32,

#         n_steps=2,#DO NOT CHANGE

#         gamma=2.194,

#         n_independent=0,

#         n_shared=2,

#         epsilon=1e-15,

#         virtual_batch_size=128,

#         momentum=0.25,#keep it small

#     )

#     return model
# !rm -rf /kaggle/working/models

# from sklearn.model_selection import KFold

# from torch.nn import SmoothL1Loss

# NUM_FOLDS = 7

# kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2019)

# all_preds = []

# for i,(train_index, val_index) in enumerate(kf.split(train_df,train_df)):

#     print('fold-',i+1)

#     #get data

#     to,n_features,n_labels = get_tabnet_data(train_df,(list(train_index), list(val_index)))

#     dls = to.dataloaders(bs=512, path='/kaggle/working/')

#     emb_szs = get_emb_sz(to)

#     #get model

#     model = get_model(emb_szs,dls,n_features,n_labels)

#     opt_func = partial(Adam,lr=0.01,mom=0.9,sqr_mom=0.99,wd=0.01,eps=1e-5,decouple_wd=True)

#     learn = Learner(dls, model, loss_func=SmoothMAELoss(l1=0.0), opt_func=opt_func, metrics=trends_scorer_multitask_scoring_gpu)



#     learn.fit_one_cycle(

#         100,

#         lr_max=0.09,

#         div=25.0,

#         div_final=1000000,

#         pct_start=0.25,

#         cbs=[EarlyStoppingCallback(min_delta=0.01,patience=50),

#                              SaveModelCallback(fname="model_{}".format(i+1),min_delta=0.01)]

#     )

# #     learn.load("model_{}".format(i+1))

# #     learn.fit_one_cycle(

# #         50,

# #         lr_max=0.001,

# #         div=5.0,

# #         div_final=1000000,

# #         pct_start=0.5,

# #         cbs=[EarlyStoppingCallback(min_delta=0.01,patience=50),

# #                              SaveModelCallback(fname="model_{}".format(i+1),min_delta=0.01)]

# #     )

#     #predicting

#     learn.load("model_{}".format(i+1))

#     print("Best model:",learn.loss)

#     to_tst = to.new(test_df)

#     to_tst.process()

#     tst_dl = dls.valid.new(to_tst)

#     tst_preds,_ = learn.get_preds(dl=tst_dl)

#     cb = None

#     all_preds.append(tst_preds)



# # PREDICTING......

# p=sum(all_preds)/NUM_FOLDS

# targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']

# res = pd.DataFrame(np.array(p),columns=[targets])

# ids=pd.DataFrame(test_df.Id.values,columns=['Id'])

# a=pd.concat([ids,res],axis=1)

# b=a.iloc[:,0:6]

# b.columns=['Id','age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']

# b.head()
# learn.fine_tune(

#     100,

#     base_lr=0.002,

#     freeze_epochs=1,

#     pct_start=0.3,

#     div=5.0,

#     div_final=100000.0,

# )
# from hyperopt import tpe

# from hyperopt import STATUS_OK

# from hyperopt import Trials

# from hyperopt import hp

# from hyperopt import fmin

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import cross_val_score

# from sklearn.datasets import load_breast_cancer

# import sys

# if not sys.warnoptions:

#     import warnings

#     warnings.simplefilter("ignore")

    

# N_FOLDS = 10

# MAX_EVALS = 200

# def objective(params, n_folds = N_FOLDS):

    

#     model=TabNetModel(

#     ems[0],

#     n_cont=1399,

#     out_sz=5,

#     embed_p=0.0,

#     y_range=None,

#     epsilon=1e-15,

#     virtual_batch_size=128,

#     **params

#     )

#     opt_func = partial(Adam, wd=0.01, eps=1e-5)

#     learn = Learner(data[0], model, loss_func=SmoothMAELoss(l1=0.0), opt_func=opt_func, metrics=[trends_scorer_multitask_scoring_gpu])

#     return {'loss':learn.loss,'params': params, 'status': STATUS_OK}





# space = {

#     'n_d' : hp.choice('n_d', range(2,64,1)),

#     'n_a' : hp.choice('n_a', range(2,64,1)),

#     'n_steps':hp.choice('n_steps', range(1,10,1)),

#     'gamma': hp.uniform('gamma', 1, 5),

#     'n_independent':hp.choice('n_independent', range(1,10,1)),

#     'n_shared': hp.choice('n_shared', range(1,10,1)),

#     'momentum' : hp.uniform('momentum', 0, 1)

# }

# # Algorithm

# tpe_algorithm = tpe.suggest



# # Trials object to track progress

# bayes_trials = Trials()



# # Optimize

# best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

# best
# import optuna.integration.lightgbm as lgb



# import numpy as np

# import pandas as pd



# from sklearn.model_selection import KFold, train_test_split



# from tqdm.notebook import tqdm

# import gc







# fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

# loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

# labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")



# fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

# df = fnc_df.merge(loading_df, on="Id")

# labels_df["is_train"] = True



# df = df.merge(labels_df, on="Id", how="left")



# target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']



# #imputing missing values in targets

# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors = 5, weights="distance")

# df[target_cols] = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns = target_cols)



# test_df = df[df["is_train"] != True].copy()

# train_df = df[df["is_train"] == True].copy()



# #y_train_df = train_df[target_cols]

# train_df = train_df.drop(['is_train'], axis=1)

# #train_df = train_df.drop(target_cols + ['is_train'], axis=1)

# test_df = test_df.drop(target_cols+['is_train'], axis=1)





# targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']

# features=list(set(train_df.columns)-set(targets)-set(['Id']))

              

# train_df[features]=train_df[features].pow(2)

# train_df[fnc_features]=train_df[fnc_features].mul(1/100)

# train_df[fnc_features]=train_df[fnc_features].pow(2)



# test_df[features]=test_df[features].pow(2)

# test_df[fnc_features]=test_df[fnc_features].mul(1/100)

# test_df[fnc_features]=test_df[fnc_features].pow(2)



# print(train_df.shape,test_df.shape)

# print("Train and test dataframes contain Id columns,too!!")









# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])

# test_df[features] = scaler.transform(test_df[features])





# X_train = train_df[features]

# X_test = test_df[features]

# y_train = train_df[targets]

# print(X_train.shape,X_test.shape)



# def my_metric(y_pred,train_data):

#     y_true = train_data.get_label()

#     print(len(y_true),len(y_pred))

#     return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))



# X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=20)

# train_data = lgb.Dataset(X_tr, label=y_tr['age'])

# val_data = lgb.Dataset(X_val, label=y_val['age'])

# params = {

#         'objective':'fair',

#         'metric':'l1',

#         'boosting_type':'gbdt',

#         'learning_rate':0.001,

#         'tree_learner':'feature_parallel',

#         'num_threads':4,

#         'seed':0

#         }



# best_params, tuning_history = dict(), list()



# model = lgb.train(params, 

#                   train_data, 

#                   num_boost_round=1000, 

#                   early_stopping_rounds=20, 

#                   valid_sets=[train_data,val_data], 

#                   verbose_eval=100,

#                   learning_rates=lambda it: 0.01 * (0.8 ** it),

#                   best_params=best_params,

#                  tuning_history=tuning_history)

 

# print("Best Params", best_params)

# params={

#         'age':{'lambda_l1': 0.029688407904725312,

#      'lambda_l2': 4.927181117399353e-03,

#      'num_leaves': 101,

#      'feature_fraction': 0.90,

#      'bagging_fraction': 1.0,

#      'bagging_freq': 0,

#      'min_child_samples': 20,

#      'objective': 'fair',

#      'metric': 'l1',

#      'boosting_type': 'gbdt',

#      'learning_rate': 0.001,

#      'tree_learner': 'feature_parallel',

#      'num_threads': 4,

#      'seed': 0}

#     ,

    

#     'domain1_var1':{'lambda_l1': 0.0,

#  'lambda_l2': 0.0,

#  'num_leaves': 200,

#  'feature_fraction': 0.95,

#  'bagging_fraction': 0.9765733975192812,

#  'bagging_freq': 1,

#  'min_child_samples': 10,

#  'objective': 'fair',

#  'metric': 'l1',

#  'boosting_type': 'gbdt',

#  'learning_rate': 0.001,

#  'tree_learner': 'feature_parallel',

#  'num_threads': 4,

#  'seed': 0}  

#     ,

#     'domain1_var2':{'lambda_l1': 7.733581684659643e-05,

#  'lambda_l2': 1.1878841440097718,

#  'num_leaves': 31,

#  'feature_fraction': 1.0,

#  'bagging_fraction': 1.0,

#  'bagging_freq': 0,

#  'min_child_samples': 25,

#  'objective': 'huber',

#  'metric': 'l1',

#  'boosting_type': 'gbdt',

#  'learning_rate': 0.01,

#  'tree_learner': 'feature_parallel',

#  'num_threads': 4,

#  'seed': 0}

#     ,

#     'domain2_var1':{'lambda_l1': 0.041395115988296434,

#  'lambda_l2': 0.00011959715500563623,

#  'num_leaves': 105,

#  'feature_fraction': 0.6,

#  'bagging_fraction': 0.5439884362351342,

#  'bagging_freq': 4,

#  'min_child_samples': 10,

#  'objective': 'huber',

#  'metric': 'l1',

#  'boosting_type': 'gbdt',

#  'learning_rate': 0.01,

#  'max_depth': -1,

#  'tree_learner': 'feature_parallel',

#  'num_threads': 8,

#  'seed': 0}

#     ,

#     'domain2_var2':{'lambda_l1': 9.606755708273219e-05,

#  'lambda_l2': 0.17107930638380894,

#  'num_leaves': 59,

#  'feature_fraction': 1.0,

#  'bagging_fraction': 1.0,

#  'bagging_freq': 0,

#  'min_child_samples': 20,

#  'objective': 'huber',

#  'metric': 'l1',

#  'boosting_type': 'gbdt',

#  'learning_rate': 0.01,

#  'max_depth': -1,

#  'tree_learner': 'feature_parallel',

#  'num_threads': 8,

#  'seed': 0}

# }
# # LightGBM--Cross Validation implementation



# from sklearn.model_selection import KFold



# import numpy as np

# import pandas as pd

# import matplotlib.pyplot as plt



# from tqdm.notebook import tqdm

# import gc



# import lightgbm as lgb





# fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

# loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

# labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")



# fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

# df = fnc_df.merge(loading_df, on="Id")

# labels_df["is_train"] = True



# df = df.merge(labels_df, on="Id", how="left")



# target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']



# #imputing missing values in targets

# from sklearn.impute import KNNImputer

# imputer = KNNImputer(n_neighbors = 5, weights="distance")

# df[target_cols] = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns = target_cols)



# test_df = df[df["is_train"] != True].copy()

# train_df = df[df["is_train"] == True].copy()



# train_df = train_df.drop(['is_train'], axis=1)

# test_df = test_df.drop(target_cols+['is_train'], axis=1)





# targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']

# features=list(set(train_df.columns)-set(targets)-set(['Id']))





# train_df[features]=train_df[features].pow(2)

# train_df[fnc_features]=train_df[fnc_features].mul(1/500)

# train_df[fnc_features]=train_df[fnc_features].pow(2)



# test_df[features]=test_df[features].pow(2)

# test_df[fnc_features]=test_df[fnc_features].mul(1/500)

# test_df[fnc_features]=test_df[fnc_features].pow(2)









# #-------Normalizing------------------------

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])

# test_df[features] = scaler.transform(test_df[features])

# #----------------------------------------------------



# print(train_df.shape,test_df.shape)

# print("Train and test dataframes contain Id column!!")





# def my_metric(y_true, y_pred):

#     return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))





# NFOLDS = 5

# from sklearn.model_selection import KFold

# kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)



# targets=['age','domain2_var1','domain2_var2', 'domain1_var1','domain1_var2']

# features=list(set(train_df.columns)-set(targets)-set(['Id']))

# overal_score = 0.0

# for target,w in tqdm([('age',0.3),('domain1_var1',0.175),('domain1_var2',0.175),('domain2_var1',0.175),('domain2_var2',0.175)]):

#     y_oof = np.zeros(train_df.shape[0])

#     y_test = np.zeros((test_df.shape[0], NFOLDS))

#     print('*'*20,target,'*'*20)

#     for i,(train_index, valid_index) in enumerate(kf.split(train_df, train_df)):

#         print('>'*20,'Fold-',i+1)

#         train,val = train_df.iloc[train_index],train_df.iloc[valid_index]

#         X_train = train[features]

#         y_train = train[target]

#         X_val = val[features]

#         y_val = val[target]

#         train_data = lgb.Dataset(X_train, label=y_train)

#         val_data = lgb.Dataset(X_val, label=y_val)

#         #create model

#         model = lgb.train(params[target], 

#                           train_data, 

#                           num_boost_round=10000, 

#                           early_stopping_rounds=20, 

#                           valid_sets=[train_data,val_data], 

#                           learning_rates=lambda it: 0.01 * (0.8 ** it),

#                           verbose_eval=100)

    

#         val_pred = model.predict(X_val)

#         test_pred = model.predict(test_df[features])

#         y_oof[valid_index] = val_pred

#         y_test[:, i] = test_pred



#     train_df["pred_{}".format(target)] = y_oof

#     test_df[target] = y_test.mean(axis=1)



#     score = my_metric(train_df[train_df[target].notnull()][target].values, train_df[train_df[target].notnull()]["pred_{}".format(target)].values)

#     print("="*20,target, np.round(score, 5))

#     print("-"*100)

#     overal_score += w*score

        

# print("Overal score:", np.round(overal_score, 5))
# X_train = df.drop(columns=['Id','age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2', 'is_train'],axis=1)

# X_test = test_df.drop(columns=['Id','age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2', 'is_train'],axis=1)

# y_train = df[['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']]
# %%time

# from sklearn.linear_model import MultiTaskElasticNetCV

# cv_model = MultiTaskElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],cv=5,verbose=1,n_jobs=10)

# cv_model.fit(train_df[3634:],y_train_df[3634:][['domain2_var1','domain2_var2']])

# #fitting multitask net with hyperparameters obtained from cross validation above

# from sklearn.linear_model import MultiTaskElasticNet

# model_d2 = MultiTaskElasticNet(alpha=cv_model.alpha_,l1_ratio=cv_model.l1_ratio_,random_state=0)

# model_d2.fit(train_df[:3634],y_train_df[:3634][['domain2_var1','domain2_var2']])
# NN_model = Sequential()



# # The Input Layer :

# NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))



# # The Hidden Layers :

# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))



# # The Output Layer :

# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))



# # Compile the network :

# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# NN_model.summary()
# def trends_scorer_multitask_scoring(estimator,X,y_true):

#     '''

#     custom scoring function used for evaluation in this competition

#     '''

#     import numpy as np

#     y_true = np.array(y_true)

#     y_preds=estimator.predict(X)

#     y_preds = np.array(y_preds)

#     w = np.array([.3, .175, .175, .175, .175])

#     op = np.mean(np.matmul(np.abs(y_true-y_preds),w/np.mean(y_true,axis=0)),axis=0)

#     print(op)

#     return op   
# %%time

# from sklearn.datasets import make_regression

# from sklearn.multioutput import MultiOutputRegressor

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import GradientBoostingRegressor

# m = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,random_state=0))

# m.fit(X_train,y_train)

# preds = m.predict(X_test)

# test_df[['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']] = preds

# test_df.drop(columns=["is_train"],inplace=True)

# test_df.head()
# from statsmodels.tsa.stattools import grangercausalitytests

# grangercausalitytests(X_train, maxlag=20, verbose=False) 
# #### Matrix factorisation using SVD

# X_train = np.asarray(X_train)#.to_gpu_matrix(),dtype=np.float32)

# mean = np.mean(X_train, axis = 1)

# sd = np.std(X_train, axis = 1)

# X_train_norm = (X_train - mean.reshape(-1, 1))/sd.reshape(-1, 1)

# from scipy.sparse.linalg import svds

# latent_factors = 50

# U, sigma, V_T = svds(X_train_norm, k = latent_factors)

# sigma = np.diag(sigma)

# U.shape,sigma.shape,V_T.shape

#--------------------------------------------------------------------------------------------------------

# from sklearn.decomposition import FactorAnalysis

# transformer = FactorAnalysis(n_components=50, random_state=0)

# X_transformed = transformer.fit_transform(X_train)

# X_transformed.shape
import os

import pandas as pd

import cudf

import gc

def get_train_test(fnc_file,loadings_file,lablels_file):

    '''

    function to get training and test data sets

    Works with Rapids.ai ONLY

    

    '''

    path = "../input/trends-assessment-prediction/"

    fnc_df = pd.read_csv(os.path.join(path,fnc_file))

    loading_df = pd.read_csv(os.path.join(path,loadings_file))

    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

    df = fnc_df.merge(loading_df, on="Id")

    labels_df = pd.read_csv(os.path.join(path,lablels_file))

    labels_df["is_train"] = True

    df = df.merge(labels_df, on="Id", how="left")

    test_df = df[df["is_train"] != True].copy()

    train_df = df[df["is_train"] == True].copy()

    train_df = train_df.drop(['is_train'], axis=1)

    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

    test_df = test_df.drop(target_cols + ['is_train'], axis=1)

    features = loading_features + fnc_features 

    #-----------------Normalizing------------------------

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    train_df[features] = scaler.fit_transform(train_df[features],train_df[target_cols])

    test_df[features] = scaler.transform(test_df[features])

    #----------------------------------------------------

    # Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

    train_df[fnc_features] = train_df[fnc_features].mul(1/800)

    test_df[fnc_features]  = test_df[fnc_features].mul(1/800) 

    #imputing missing values in targets

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors = 5, weights="distance")

    train_df = cudf.from_pandas(pd.DataFrame(imputer.fit_transform(train_df), columns = list(train_df.columns)))

    test_df = cudf.from_pandas(test_df)#necessary for casting to gpu matrix

    del df

    gc.collect()

    return train_df,test_df,features,target_cols

import numpy as np

from cuml import SVR

from cuml import RandomForestRegressor

from cuml import NearestNeighbors,KMeans,UMAP,Ridge,ElasticNet

import cupy as cp

from sklearn.model_selection import KFold





def my_metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))



def cv_train_predict(df,test_df,features):

    '''

    training with k-fold cross-validation

    '''

    weights={}#Weights & other hyperparameters

    #[target,score_weight,SVR_penalty(C),ElasticNet_l1ratio,Blend_ElNet_weight,Blend_RandForest_weight]

    weights['age']         =["age",         0.3,   40, 0.8, 0.5, 0.3]

    weights['domain1_var1']=["domain1_var1",0.175,  8, 0.5, 0.6, 0.4]

    weights['domain1_var2']=["domain1_var2",0.175,  8, 0.5, 0.6, 0.4]

    weights['domain2_var1']=["domain2_var1",0.175, 10, 0.8, 0.8, 0.5]

    weights['domain2_var2']=["domain2_var2",0.175, 10, 0.5, 0.6, 0.5]

    NUM_FOLDS = 7

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

    overal_score = 0

    for target,w,c,l1_ratio,el,rf in [weights['age'], weights['domain1_var1'], weights['domain1_var2'],weights['domain2_var1'],weights['domain2_var2']]:    

        y_oof = np.zeros(df.shape[0])

        y_test = np.zeros((test_df.shape[0], NUM_FOLDS))



        for f, (train_ind, val_ind) in enumerate(kf.split(df,df)):

            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

          

            #-------training,val,test data preparation for RandomForestRegressor since it operates on float32

            X_train = np.array(train_df[features].to_gpu_matrix(),dtype=np.float32)

            y_train = np.array(train_df[[target]].to_gpu_matrix(),dtype=np.float32)

            X_val = np.array(val_df[features].to_gpu_matrix(),dtype=np.float32)

            y_val = np.array(val_df[[target]].to_gpu_matrix(),dtype=np.float32)

            X_test = np.array(test_df[features].to_gpu_matrix(),dtype=np.float32)

            #---------------------------------------------------------------------------------------

            

            

            model = RandomForestRegressor(n_estimators=200,split_criterion=2,accuracy_metric=my_metric,bootstrap=True,seed=0)

            model.fit(X_train,y_train)

            model_1 = SVR(C=c, cache_size=3000.0)

            model_1.fit(train_df[features].values, train_df[target].values)

        

            model_2 = ElasticNet(alpha = 1,l1_ratio=l1_ratio)

            model_2.fit(train_df[features].values, train_df[target].values)

            

            val_pred_rf=model.predict(X_val)

            val_pred_1 = model_1.predict(val_df[features])

            val_pred_2 = model_2.predict(val_df[features])



            test_pred_rf=model.predict(X_test)

            test_pred_1 = model_1.predict(test_df[features])

            test_pred_2 = model_2.predict(test_df[features])

            #pred    = Blended prediction(RandomForest + Blended prediction(ElasticNet & SVR))

            

            

            val_pred = rf*val_pred_rf + cp.asnumpy((1-rf)*((1-el)*val_pred_1+el*val_pred_2))

            #val_pred = cp.asnumpy(val_pred.values.flatten())

            

            test_pred = rf*test_pred_rf + cp.asnumpy((1-rf)*((1-el)*test_pred_1+el*test_pred_2))

            #test_pred = cp.asnumpy(test_pred.values.flatten())



            y_oof[val_ind] = val_pred

            y_test[:, f] = test_pred



        df["pred_{}".format(target)] = y_oof

        test_df[target] = y_test.mean(axis=1)



        score = my_metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)

        print(target, np.round(score, 5))

        print()

        overal_score += w*score

        

    print("Overal score:", np.round(overal_score, 5))

%%time

df,test_df,features, targets = get_train_test("fnc.csv","loading.csv","train_scores.csv")

print("training shape={0} | testing shape={1}".format(df.shape, test_df.shape))

print(type(df),type(test_df),'Id' in features,'Id' in df.columns,'Id' in test_df.columns)

targets = ['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']

df[targets].isna().sum()
%%time

to_drop=['IC_20','IC_02','IC_05','IC_16','IC_10','IC_18']

features = list(set(features)-set(to_drop))#Id is not present in features

print("After excluding features and Id, training shape={0} | testing shape={1}".format(df[features].shape, test_df[features].shape))

cv_train_predict(df,test_df,features)
# NUM_FOLDS = 7

# kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

# for f,(train_ind, val_ind) in enumerate(kf.split(df,df)):

#     #model = SVR(C=12, cache_size=3000.0,verbose=True)

#     train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

    

#     X_train = np.array(train_df[features].to_gpu_matrix(),dtype=np.float32)

#     y_train = np.array(train_df[['domain2_var2']].to_gpu_matrix(),dtype=np.float32)

#     X_val = np.array(val_df[features].to_gpu_matrix(),dtype=np.float32)

#     y_val = np.array(val_df[['domain2_var2']].to_gpu_matrix(),dtype=np.float32)

#     model = RandomForestRegressor(n_estimators=100,split_criterion=2,accuracy_metric=my_metric,bootstrap=True,seed=0)

#     model.fit(X_train,y_train)

#     print(my_metric(y_val,model.predict(X_val)))
# 510 Ids of site2 are known in test data, there are more...510 are not all of site2 Ids. 

# training data does not have site2 Ids

# site2_df = cudf.read_csv("../input/trends-assessment-prediction/reveal_ID_site2.csv")

# testdf_site2 = test_df[test_df['Id'].isin(list(site2_df['Id']))]

# testdf_site2.shape

# np.array(train_df[features].fillna(0).to_gpu_matrix(),dtype=np.float32)
# #SCRATCH-PAD

# def metric(y_true, y_pred):#-----------------------------C-----H----A---N---G----E----S------

#     import numpy as np

#     return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))#CHANGED y_true--->y_pred



# features = loading_features + fnc_features

# X = np.array(df[features].to_gpu_matrix(),dtype=np.float32)[:5000]

# y = np.array(df[['domain2_var1']].to_gpu_matrix(),dtype=np.float32)[:5000]

# #model = RandomForestRegressor(n_estimators=100,split_criterion=3,accuracy_metric=metric,seed=0,bootstrap=True)

# model=MBSGDRegressor(loss='squared_loss',penalty='elasticnet',learning_rate='adaptive',n_iter_no_change=5,verbose=True)

# model.fit(X,y)

# X_test = np.array(df[features].to_gpu_matrix(),dtype=np.float32)[5000:]

# print(X_test.shape)
output = test_df #

sub_df = cudf.melt(output[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.head(10)
sub_df.to_csv("submission.csv", index=False)