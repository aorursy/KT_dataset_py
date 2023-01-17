# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# traindf=pd.read_csv('../input/club-mahindra-hackathon-analytics-vidhya/train.csv')

# testdf=pd.read_csv('../input/club-mahindra-hackathon-analytics-vidhya/test.csv')

# total_data=traindf.append(testdf)

# total_data.shape



# total_data['checkout_date']=pd.to_datetime(total_data['checkout_date']).dt.strftime('%d-%m-%Y')

# total_data['checkin_date']=pd.to_datetime(total_data['checkin_date']).dt.strftime('%d-%m-%Y')

# total_data['booking_date']=pd.to_datetime(total_data['booking_date']).dt.strftime('%d-%m-%Y')



# #### ignore room night and create new feild





# total_data['month_checkout']=pd.to_datetime(total_data.checkout_date).dt.month

# total_data['month_checkin']=pd.to_datetime(total_data.checkin_date).dt.month

# total_data['month_booking']=pd.to_datetime(total_data.booking_date).dt.month



# total_data['dayofweek_checkout']=pd.to_datetime(total_data.checkout_date).dt.dayofweek

# total_data['dayofweek_checkin']=pd.to_datetime(total_data.checkin_date).dt.dayofweek

# total_data['dayofweek_booking']=pd.to_datetime(total_data.booking_date).dt.dayofweek



# total_data['day_checkout']=pd.to_datetime(total_data.checkout_date).dt.day

# total_data['day_checkin']=pd.to_datetime(total_data.checkin_date).dt.day

# total_data['day_booking']=pd.to_datetime(total_data.booking_date).dt.day



# total_data['year_checkout']=pd.to_datetime(total_data.checkout_date).dt.year

# total_data['year_checkin']=pd.to_datetime(total_data.checkin_date).dt.year

# total_data['year_booking']=pd.to_datetime(total_data.booking_date).dt.year



# total_data['days_spent']=(pd.to_datetime(total_data.checkout_date)-pd.to_datetime(total_data.checkin_date)).dt.days



# ####

# def create_dict_resort():

#     _=dict(enumerate(list(total_data.resort_id.unique())))

#     return dict((v,k) for k,v in _.items())

# dict_resort=create_dict_resort()    



# total_data['resort_id']=total_data['resort_id'].astype('str')

# total_data['resort_id']=total_data.resort_id.map(dict_resort)



# total_data.to_csv('new_total.csv',index=False)
totaldata=pd.read_csv('../input/mhakss/new_total.csv')
# totaldata.head()

totaldata['checkout_date']=pd.to_datetime(totaldata['checkout_date']).dt.strftime('%d-%m-%Y')

totaldata['checkin_date']=pd.to_datetime(totaldata['checkin_date']).dt.strftime('%d-%m-%Y')

# total_data['booking_date']=pd.to_datetime(total_data['booking_date']).dt.strftime('%d-%m-%Y')
# pd.to_datetime(totaldata.checkout_date.head()).dt.month
totaldata['week_checkout']=pd.to_datetime(totaldata.checkout_date).dt.week

totaldata['week_checkin']=pd.to_datetime(totaldata.checkin_date).dt.week



# traindf=pd.read_csv('../input/club-mahindra-hackathon-analytics-vidhya/train.csv')

# testdf=pd.read_csv('../input/club-mahindra-hackathon-analytics-vidhya/test.csv')
#### create feature - persons * nights spent



#### check if adullt childere n eq total pax



# anomly=totaldata[totaldata['numberofadults']+totaldata['numberofchildren'] != totaldata['total_pax']]



def anomaly_persons_flag(x):

    if x['numberofadults']+x['numberofchildren'] <  x['total_pax']:

        return 1

    elif x['numberofadults']+x['numberofchildren'] >  x['total_pax']:

        return 2

    else:

        return 0

    

def anomaly_persons_difference(x):

    return (x['numberofadults']+x['numberofchildren'])- x['total_pax']





def anomaly_nights_flag(x):

    if x['roomnights'] < x['days_spent']:

        return 1

    elif x['roomnights'] > x['days_spent']:

        return 2 

    else:

        return 0

    

def anomaly_nights_difference(x):

    return x['roomnights']- x['days_spent']



def room_persons_night(x):

    return (x['numberofadults']+x['numberofchildren'])*x['roomnights']

def room_pax_night(x):

    return x['total_pax']*x['roomnights']

def room_persons_days(x):

    return (x['numberofadults']+x['numberofchildren'])*x['days_spent']

def room_pax_days(x):

    return x['total_pax']*x['days_spent']



def tot_guests(x):

    return x['numberofadults']+x['numberofchildren']

totaldata.loc[36008,'roomnights']=7

totaldata.loc[36008,'nights_difference']=0

totaldata.loc[36008,'nights_flag']=0
totaldata['nights_difference']=totaldata.apply(anomaly_nights_difference,axis=1)

totaldata['nights_flag']=totaldata.apply(anomaly_nights_flag,axis=1)

totaldata['persons_difference']=totaldata.apply(anomaly_persons_difference,axis=1)

totaldata['persons_flag']=totaldata.apply(anomaly_persons_flag,axis=1)

totaldata['roompnight']=totaldata.apply(room_persons_night,axis=1)

totaldata['roompaxnight']=totaldata.apply(room_pax_night,axis=1)

totaldata['roompday']=totaldata.apply(room_persons_days,axis=1)

totaldata['roompaxday']=totaldata.apply(room_pax_days,axis=1)
totaldata['total_guests']=totaldata.numberofadults+totaldata.numberofchildren

totaldata['adult_nights']=totaldata.numberofadults*totaldata.days_spent
# totaldata.booking_diff.describe()
#### index where checkin date was before booking date 

# book_check_anaomaly=totaldata[totaldata.booking_diff<0].index

# book_check_anaomaly
# totaldata.loc[book_check_anaomaly,'booking_diff']=None
col_order=['amount_spent_per_room_night_scaled', 'booking_date','month_booking','day_booking','year_booking','dayofweek_booking',

        'checkin_date', 'checkout_date', 'month_checkout', 'month_checkin'

       , 'dayofweek_checkout', 'dayofweek_checkin','day_checkout','day_checkin',

       'year_checkout', 'year_checkin',

 'days_spent','roomnights','nights_difference', 'nights_flag',

  'numberofadults', 'numberofchildren','total_pax', 'total_guests','persons_difference',

       'persons_flag','roompnight','roompaxnight','roompday','roompaxday','adult_nights',

       'cluster_code', 'main_product_code', 'member_age_buckets', 'memberid'

       , 'persontravellingid',

       'reservation_id', 'reservationstatusid_code', 'resort_id',

       'resort_region_code', 'resort_type_code', 'room_type_booked_code',

       'season_holidayed_code', 'state_code_residence',

       'state_code_resort','booking_type_code', 'channel_code','week_checkout'

,'week_checkin'

          ]

totaldata=totaldata[col_order]
# pd.read_excel('../input/club-mahindra-hackathon-analytics-vidhya/Data_Dictionary.xlsx')
totaldata.head()
cols_to_del=['memberid','reservation_id','booking_date','checkin_date','checkout_date']

totaldata2=totaldata.drop(columns=cols_to_del,axis=1)
#### create christmas new year 
totaldata2.columns
cols_cat=['month_booking', 'day_booking',

       'year_booking', 'dayofweek_booking', 'month_checkout', 'month_checkin',

       'dayofweek_checkout', 'dayofweek_checkin', 'day_checkout',

       'day_checkin', 'year_checkout', 'year_checkin', 'nights_flag','persons_flag',

            'cluster_code', 'main_product_code', 'member_age_buckets',

       'persontravellingid', 'reservationstatusid_code', 'resort_id',

       'resort_region_code', 'resort_type_code', 'room_type_booked_code',

       'season_holidayed_code', 'state_code_residence', 'state_code_resort',

       'booking_type_code', 'channel_code','week_checkout'

,'week_checkin'

           ]

for c in cols_cat:

    totaldata2[c]=totaldata2[c].astype(object)
totaldata2['season_holidayed_code'].fillna(np.nan,inplace=True)

totaldata2['state_code_residence'].fillna(np.nan,inplace=True)
totaldata2.amount_spent_per_room_night_scaled.describe()
def create_hdays(x):

    if (x['month_checkin']==2) & (x['day_checkin']==26):

        return 'republic'

    elif (x['month_checkin']==8) & (x['day_checkin']==15):

        return 'ind'

    elif (x['month_checkin']==12) & (x['day_checkin']==25):

        return 'christmas'

    elif (x['month_checkin']==12) & (x['day_checkin']==31):

        return 'newyr'

    elif (x['month_checkin']==1) & (x['day_checkin']==1):

        return 'newyr'

    else:

        return 'other'

        
# totaldata2['holiday']=totaldata2.apply(create_hdays,axis=1)
# totaldata2[(totaldata2.month_checkin==1) & (totaldata2.day_checkin==1)]['amount_spent_per_room_night_scaled'].describe()
dftrain=totaldata2[totaldata2['amount_spent_per_room_night_scaled'].isnull()!=True]

dftest=totaldata2[totaldata2['amount_spent_per_room_night_scaled'].isnull()==True]

# dftrain.head()
# import catboost



# class ModelOptimizer:

#     best_score = None

#     opt = None

    

#     def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

#         self.model = model

#         self.X_train = X_train

#         self.y_train = y_train

#         self.categorical_columns_indices = categorical_columns_indices

#         self.n_fold = n_fold

#         self.seed = seed

#         self.early_stopping_rounds = early_stopping_rounds

#         self.is_stratified = is_stratified

#         self.is_shuffle = is_shuffle

        

        

#     def update_model(self, **kwargs):

#         for k, v in kwargs.items():

#             setattr(self.model, k, v)

            

#     def evaluate_model(self):

#         pass

    

#     def optimize(self, param_space, max_evals=10, n_random_starts=2):

#         start_time = time.time()

        

#         @use_named_args(param_space)

#         def _minimize(**params):

#             self.model.set_params(**params)

#             return self.evaluate_model()

        

#         opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

#         best_values = opt.x

#         optimal_values = dict(zip([param.name for param in param_space], best_values))

#         best_score = opt.fun

#         self.best_score = best_score

#         self.opt = opt

        

#         print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

#         print('updating model with optimal values')

#         self.update_model(**optimal_values)

#         plot_convergence(opt)

#         return optimal_values

    

# class CatboostOptimizer(ModelOptimizer):

#     def evaluate_model(self):

#         validation_scores = catboost.cv(

#         catboost.Pool(self.X_train, 

#                       self.y_train, 

#                       cat_features=self.categorical_columns_indices),

#         self.model.get_params(), 

#         nfold=self.n_fold,

#         stratified=self.is_stratified,

#         seed=self.seed,

#         early_stopping_rounds=self.early_stopping_rounds,

#         shuffle=self.is_shuffle,

#         verbose=100,

#         plot=False)

#         self.scores = validation_scores

#         test_scores = validation_scores.iloc[:, 2]

#         best_metric = test_scores.max()

#         return 1 - best_metric
import catboost
x,y=dftrain.drop('amount_spent_per_room_night_scaled',axis=1),dftrain['amount_spent_per_room_night_scaled']

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.3,random_state = 42)
# for i in x_train.columns:

#     if x[i].dtype=='O':

#         print(x[i].value_counts())

# try keeping imp vars

# imp_vars=[]
# x_train.month_booking.dtype=='O'
# x_train.dtypes

# x_train.columns
categorical_features_indices = np.where(x_train.dtypes =='object')[0]

categorical_features_indices
from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.utils import use_named_args

from skopt.plots import plot_convergence

import time
# cb = catboost.CatBoostRegressor(n_estimators=6000, # use large n_estimators deliberately to make use of the early stopping

#                          boosting_type='Ordered', # use permutations

#                          random_seed=1,

#                          task_type='GPU',

#                          use_best_model=True)
# cb_optimizer = CatboostOptimizer(cb, x_train, y_train,categorical_columns_indices=categorical_features_indices)

# params_space = [Real(0.01, 0.8, name='learning_rate'),]

# cb_optimal_values = cb_optimizer.optimize(params_space)
# cb_optimal_values.get('learning_rate')
# optimal_parameters: {'learning_rate': 0.14543173170768622}

# 0.39994351552579455
# x_train.columns
import matplotlib.pyplot as plt

%matplotlib inline

# from catboost import CatBoostClassifier

from sklearn.metrics import mean_squared_error

# m=catboost.CatBoostRegressor(n_estimators=6000,random_state=1,max_depth=6,learning_rate=0.03

#                      ,cat_features=categorical_features_indices,task_type='GPU',

#                      use_best_model=True)

# m.fit(x_train,y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=100,verbose=100)

# p2=m.predict_proba(X_val)[:,-1]
# x_train.columns
# totaldata2.season_holidayed_code.value_counts()
# totaldata2[(totaldata2.month_checkin==8) & (totaldata2.week_checkin==33)]['amount_spent_per_room_night_scaled'].describe()
# sorted(zip(m.feature_importances_,x_train),reverse=True)
# dftest.columns==dftrain.columns



errcb=[]

y_pred_totcb=[]

from sklearn.model_selection import KFold,RepeatedKFold

fold=RepeatedKFold(n_splits=5,random_state=1)

i=1

for train_index, test_index in fold.split(x,y):

    X_train, X_test = x.iloc[train_index], x.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    m=catboost.CatBoostRegressor(n_estimators=6000,random_state=1,max_depth=6,

                                 learning_rate=0.084

                     ,cat_features=categorical_features_indices,task_type='GPU',

                     use_best_model=True)

    m.fit(X_train,y_train,eval_set=[(X_test, y_test)], early_stopping_rounds=100,verbose=100)    

    preds=m.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    print("MSE: %.4f" % mse)

    

    errcb.append(mse)

    

    p = m.predict(dftest.drop('amount_spent_per_room_night_scaled',axis=1))

    

    y_pred_totcb.append(p)
mean_df=pd.DataFrame({'mse':errcb})

mean_df.to_csv('meanerr.csv')
print(np.mean(errcb))
s=pd.DataFrame({'amount_spent_per_room_night_scaled':np.mean(y_pred_totcb,0)})

s.head()

s.to_csv('cb10foldscatMostfeat.csv',index=False)
# y_sub=m.predict(dftest.drop('amount_spent_per_room_night_scaled',axis=1))

# reserv_ids=totaldata[totaldata['amount_spent_per_room_night_scaled'].isnull()==True]['reservation_id']
# submss=pd.DataFrame({'reservation_id':reserv_ids,'amount_spent_per_room_night_scaled':y_sub})

# submss=pd.DataFrame({'amount_spent_per_room_night_scaled':y_sub})
# submss.to_csv('submss14.csv',index=False)
# from IPython.display import FileLink

# FileLink('submss14.csv')
# from sklearn.model_selection import GridSearchCV
# model = catboost.CatBoostRegressor(cat_features=categorical_features_indices,

#                              task_type='GPU',

#                      use_best_model=True)



# grid = GridSearchCV(estimator=model, param_grid = params, cv = 4)

# grid.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=100,verbose=100)    



# # Results from Grid Search

# print("\n========================================================")

# print(" Results from Grid Search " )

# print("========================================================")    

    

# print("\n The best estimator across ALL searched params:\n",

#           grid.best_estimator_)

    

# print("\n The best score across ALL searched params:\n",

#           grid.best_score_)

    

# print("\n The best parameters across ALL searched params:\n",

#           grid.best_params_)
# # this function does 3-fold crossvalidation with catboostclassifier          

# def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):

#     kf = KFold(n_splits=n_splits,shuffle=True) 

#     res = []

#     for train_index, test_index in kf.split(train_set):

#         train = train_set.iloc[train_index,:]

#         test = train_set.iloc[test_index,:]



#         labels = train_label.ix[train_index]

#         test_labels = train_label.ix[test_index]



#         clf = cb.CatBoostClassifier(**params)

#         clf.fit(train, np.ravel(labels), cat_features=cat_dims)



#         res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))

#     return np.mean(res)



# # this function runs grid search on several parameters

# def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):

#     ps = paramsearch(params)

#     # search 'border_count', 'l2_leaf_reg' etc. individually 

#     #   but 'iterations','learning_rate' together

#     for prms in chain(ps.grid_search(['border_count']),

#                       ps.grid_search(['ctr_border_count']),

#                       ps.grid_search(['l2_leaf_reg']),

#                       ps.grid_search(['iterations','learning_rate']),

#                       ps.grid_search(['depth'])):

#         res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)

#         # save the crossvalidation result so that future iterations can reuse the best parameters

#         ps.register_result(res,prms)

#         print(res,prms,s'best:',ps.bestscore(),ps.bestparam())

#     return ps.bestparam()
# folds = 4

# param_comb = 5

# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



# grid_search = GridSearchCV(model_xgb, param_grid=params,  scoring='roc_auc', n_jobs=4,

#                                    cv=skf.split(x_train_after_resample,y_train_after_resample), verbose=3)