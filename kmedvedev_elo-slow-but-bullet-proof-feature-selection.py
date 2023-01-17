import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import lightgbm as lgb

import warnings

import time

import sys

import datetime

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error



warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 500)



import os

print(os.listdir("../input"))

print(os.listdir("../input/elo-merchant-category-recommendation"))



def save_dict_to_file(dic, fname):

    f = open(fname,'w')

    f.write(str(dic))

    f.close()



def load_dict_from_file(fname):

    f = open(fname,'r')

    data=f.read()

    f.close()

    return eval(data)





print("Loading")

t0 = time.time()





train=pd.read_csv("../input/elo-merchant-category-recommendation/train.csv")



#test=pd.read_csv("test.csv")

print("Number of records loaded ", len(train.index))



# dummy features for demo purposes

train['demo1']=train['feature_1']*train['feature_2']

train['demo2']=train['feature_2']*train['feature_3']

train['demo3']=train['feature_1']*train['feature_3']

train['demo4']=train['feature_1']*train['feature_3']*train['feature_2']

train['demo5']=train['feature_1']+train['feature_3']+train['feature_2']

train.head(10)
split=80000 # part of the data for training. There will be no folds for speedup



train=train.sample(frac=0.50, random_state=2019).reset_index(drop=True)

target=train["target"]

del(train["target"])





#import cartegorical features prepared in advance

#with open("categorical_features"+ver+".l", 'r') as fp:

#	#categorical_features=pickle.load(fp)

#	categorical_feats=[line.rstrip('\n') for line in fp]

#	



features = [c for c in train.columns]# take all features from the file



categorical_feats=['city_id','authorized_flag','category_1','merchant_category_id','state_id','subsector_id','merchant_group_id']

categorical_feats = categorical_feats + [c for c in features if 'feature_' in c]# "calculated_merchant_category", "calculated_merchant_group","calculated_city",





# remove categorical features if they do not match column names (to preven exceptions)

categorical_feats=[i for i in categorical_feats if i in train.columns] # for a case if we lost a column to prevent exception later



if "target" in features:

    features.remove("target") # do not add target to the train or test



if "card_id" in features:

    features.remove("card_id") # do not add target to the train or test

    

if "first_active_month" in features:

    features.remove("first_active_month") # do not add target to the train or test

    

    

print("Number of features", len(features))



param = {'num_leaves': 120,

         'min_data_in_leaf': 90, 

         'objective':'regression',

         'max_depth': 9,

         'learning_rate': 0.005,

         "boosting": "gbdt",

         "feature_fraction": 0.4,

         "bagging_freq": 1,

         "bagging_fraction": 0.92 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 14.5,

         "random_state": 133,

         "verbosity": -1}



print(f'{time.time() - t0:.1f} seconds')

print("Start looping")

t0 = time.time()



added_feats=dict()

added_feats_cv=dict()

step_n=0



features_history = pd.DataFrame(columns=['feature','cv','improvement','step'])



# Prepare feature list to start

# features from previous run

feat = {512: 'new_purchase_amount_max', 1023: 'month_lag_std', 1533: 'hist_coocrnc_merchant_group_id_merchant_group_id_4', 2042: 'new_month_lag_mean', 2550: 'trade_count', 3057: 'installments_1000_count', 3563: 'elapsed_time', 4068: 'auth_month_lag_max', 4572: 'latest_merchant_id_1', 5075: 'new_merchant_id_nunique', 5577: 'hist_coocrnc_city_id_city_id_5', 6078: 'hist_coocrnc_city_id_city_id_1', 6578: 'most_frequent_merchant_category_id_count_pct', 7077: 'new_purchase_date_ptp', 7575: 'hist_purchase_amount_sum', 8072: 'auth_numerical_2_sum', 8568: 'purchase_amount_max', 9063: 'auth_purchase_amount_min', 9557: 'new_purchase_weekofyear_mean'}

features_out=list()

for f in feat.keys():

    features_out.append(feat[f])



# features to start with added manually

features_out=features_out+['installments_sum_mean','auth_category_1_sum','hist_coocrnc_city_id__1','hist_month_lag_std', 'feature_3','feature_2','feature_1','new_purchase_date_max','latest_purchase_date_1','auth_purchase_date_max','hist_purchase_date_max','new_installments_mean','hist_installments_mean','auth_installments_mean']



features_out=list(set(features_out)) # remove duplicates



print("Configuration for the  features list has ",len(features_out), "records")



# cleanup if made a mistake

features_out = [c for c in features_out if c in features] 



print("Cleaned features list has ",len(features_out), "records")



# save the list for later use (if process fails or to run concurent process)

#with open("features_selection_starting_with_"+ver+".l", 'w') as fp:

#    for s in features_out:

#        fp.write(s + '\n')



print("Starting features list",len(features_out))

print("Total  features list",len(features))



categorical_feats_out=categorical_feats.copy()



pass_no=0



while len(features_out)<300:

    cv_dic=dict()

    pass_no=pass_no+1

    is_excluded=0

    print("Pass ---- " , pass_no)

    for fet in features:

        

        if fet not in features_out:

            is_excluded=1

            print("################ ",step_n," #####################")

            step_n=step_n+1

            print("feature added >>>> ", fet)

            oof = np.zeros(len(train)-split)

            

            features_in=features_out.copy() + [fet]



            categorical_in=[i for i in categorical_feats if i in features_in]





            trn_data = lgb.Dataset(train.iloc[:split][features_in],

                                   label=target.iloc[:split],

                                   categorical_feature=categorical_in

                                  )

            val_data = lgb.Dataset(train.iloc[split:][features_in],

                                   label=target.iloc[split:],

                                   categorical_feature=categorical_in

                                  )

        

            start = time.time()

            feature_importance_df = pd.DataFrame()

        

        

            num_round = 10000

            clf = lgb.train(param,

                            trn_data,

                            num_round,

                            valid_sets = [trn_data, val_data],

                            verbose_eval=100,

                            early_stopping_rounds = 200)

            

            oof = clf.predict(train.iloc[split:][features_in], num_iteration=clf.best_iteration)

            cv=mean_squared_error(oof, target[split:])**0.5

            cv_dic[fet]=cv

            print("CV score: {:<8.5f}".format(cv))

            print(f'{time.time() - t0:.1f} seconds')

            t0 = time.time()

    if is_excluded==0:

        print("No more features to add")

        break

        #fold_importance_df = pd.DataFrame()

        #fold_importance_df["feature"] = features_in

        #fold_importance_df["importance-split"] = clf.feature_importance(importance_type='split') # split levels

        #fold_importance_df["importance-gain"] = clf.feature_importance(importance_type='gain') # split levels

        #fold_importance_df.to_csv("importance-"+fet+".csv")

        

    cv_df=pd.DataFrame.from_dict(cv_dic, orient='index', columns=[ 'cv'])

    mean_cv=cv_df['cv'].mean()

    cv_df['improvement']=mean_cv-cv_df['cv']

    

    cv_df.reset_index(inplace=True)

    cv_df=cv_df[["index",'cv','improvement']]

    cv_df.columns=['feature','cv','improvement']

    cv_df['step']=step_n

    features_history=features_history.append(cv_df)

    cv_df=cv_df.sort_values(by="improvement", ascending=False).reset_index()

    column_to_add=cv_df.head(1)['feature'].values[0]

    

    #features_history.to_csv("importance_direct.csv")



    features_out.append(column_to_add)

    #with open("full_features_list"+ver+".l", 'w') as fp:

    #    fp.write("'"+"','".join(features_out)+"'" + '\n')



    cv=cv_df.head(1)['cv'].values[0]

    

    features_tmp=list(cv_df['feature']) # reset order of features. Most improving go first

    

    features=features_tmp + [c for c in features if c not in features_tmp] # adding non rated features for more longer run



        

    print('>>>>>>>>>>',column_to_add, 'removed with cv ', cv)

    added_feats[step_n]=column_to_add

    added_feats_cv[step_n]=cv

    #save_dict_to_file(added_feats, 'added_feats.txt')

    #save_dict_to_file(added_feats_cv, 'added_feats_cv.txt')

    #save_dict_to_file(added_feats, 'added_feats_full_list.txt')

    print(f'{time.time() - t0:.1f} seconds')

    print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))

    print("Cycle ", step_n)

    t0 = time.time()

    

    

print(added_feats)

print(added_feats_cv)



    





print(f'{time.time() - t0:.1f} seconds')

print("Start looping")

t0 = time.time()



df=features_history.copy()

dic=dict()

ncount=0

for st in list(df[['step']].drop_duplicates()['step']):

    dic[st]=ncount

    ncount=ncount+1

    

df['step']=df['step'].apply(lambda x: dic[x])



from matplotlib.pyplot import cm

import numpy as np

import matplotlib.pyplot as plt



#variable n should be number of curves to plot (I skipped this earlier thinking that it is obvious when looking at picture - sorry my bad mistake xD): n=len(array_of_curves_to_plot)

#version 1:



features=list(df[['feature']].drop_duplicates()['feature'])

n=len(features)





color=iter(cm.rainbow(np.linspace(0,1,n)))

fig, ax1 = plt.subplots(figsize=(14,14))

for i in range(n):

    x=df[df['feature']==features[i]]['step'].values



    y=df[df['feature']==features[i]]['cv'].values

    c=next(color)

    ax1.plot(x, y,c=c)

#ax1.show()