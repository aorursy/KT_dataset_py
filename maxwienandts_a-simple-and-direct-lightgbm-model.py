import time

import pandas as pd

import numpy as np

import re



import lightgbm as lgb
#Load the data. The path is the same for everyone.

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
#Some adjustments so we can rum the lightGBM



#Even though 'sig_id' is not in the feature vecture used to model, lightgbm will presente a error if we do not delete it. 

train_x = train_features.drop(['sig_id'], axis = 1)

test_x = test_features.drop(['sig_id'], axis = 1)

train_y = train_targets_scored.drop(['sig_id'], axis = 1)





#lightgbm accepts only numerical and categorical varibles. 'cp_type' and 'cp_dose' are categorical variables but were not defined as such. 

#Considering that there are only 2 categories in each variable, I transformed them in numerical variables.

#If we had more categories, the model would ran, but probably would not be a good one.

train_x['cp_type'] = train_features.apply(lambda row: 0 if row['cp_type'] == 'trt_cp' else 1, axis = 1)

test_x['cp_type'] = test_features.apply(lambda row: 0 if row['cp_type'] == 'trt_cp' else 1, axis = 1)



train_x['cp_dose'] = train_features['cp_dose'].str.extract(r"([1-2])", expand = True).astype(np.int8)

test_x['cp_dose'] = test_features['cp_dose'].str.extract(r"([1-2])", expand = True).astype(np.int8)
#Modeling

'''

This model took 3 hours and 30 minutes. 

Therefore, there are plenty of space to develop new variables and increase some hyperparameters as "bagging_freq" and 'num_iterations'.

'''



progress = 0  #Usefull to see the progress of the code. 

progress_20 = 1

start = time.time()



predicted_MoA_test = pd.DataFrame(test_features['sig_id'])

for col in train_y:   ##########   Tirar os [['']]

    if col == 'sig_id':

        continue

        

    lgb_params = {

        "objective": "binary",

        "metric": "binary_logloss",

        "boosting_type": "gbdt",

        "bagging_freq": 20,

        "bagging_fraction": 0.3,

        "feature_fraction": 0.6,

        "learning_rate": 0.01,

        "lambda_l2": 0.1,

        'verbosity': 1,

        'num_iterations': 2000,

        #'early_stopping_round': 200,

        'num_leaves': 400,

        "min_data_in_leaf": 200,

        'seed': 1

    }

    

    features = train_x.columns.tolist()

    lgb_train = lgb.Dataset(data = train_x, label = train_y[col], feature_name = features)

    model = lgb.train(train_set = lgb_train, params = lgb_params)

    

    #Prediction

    predicted_MoA = []

    predicted_MoA = model.predict(test_x)

    predicted_MoA_test[col] = predicted_MoA

    

    

    #Running time

    progress += 1

    if progress == progress_20 * 20:



        progress_per = round(progress / len(train_y.columns), 4)

        print(progress_per)

        progress_20 +=1

        

        end = time.time()

        elapsed = int(round(end - start, 0))

        total_run_time =  int(round(elapsed / (progress_per), 0))

        time_to_finish = int(round(elapsed / (progress_per), 0)) - elapsed

        print('Elapsed: {:02d}:{:02d}:{:02d}'.format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        print('Total run time: {:02d}:{:02d}:{:02d}'.format(total_run_time // 3600, (total_run_time % 3600 // 60), total_run_time % 60))

        print('Time to finish: {:02d}:{:02d}:{:02d}'.format(time_to_finish // 3600, (time_to_finish % 3600 // 60), time_to_finish % 60))

        print()
#submit

#P.S.: You must select "Save & Run All (Commit)" or your submission file will not appear as an option to submit. 

predicted_MoA_test.to_csv('submission.csv', index=False)
predicted_MoA_test