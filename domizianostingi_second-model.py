'''

!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl

import datatable as dt

'''
import riiideducation

import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

env = riiideducation.make_env()
train_csv = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", 

                        usecols=[1, 2, 3, 4, 7, 8, 9],

                        dtype={'timestamp': 'int64',

                          'user_id': 'int32',

                          'content_id': 'int16',

                          'content_type_id': 'int8',

                          'answered_correctly':'int8',

                          'prior_question_elapsed_time': 'float32',

                          'prior_question_had_explanation': 'boolean'}, nrows=65000000  

                   )

questions_csv = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv")
train_csv = train_csv[train_csv.content_type_id == False]



#arrange by timestamp

train_csv = train_csv.sort_values(['timestamp'], ascending=True).reset_index(drop = True)

train_csv = train_csv.drop(columns=['content_type_id'])

train_csv.head(10)
# how, on average, the users correclty answered at the first question, second question and so on 

content_mean_final = train_csv[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])

content_mean_final.columns = ["answered_correctly_content_mean"]  

# in average how much a student correclty replay and the total number of questions for student 

user_mean_final = train_csv[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'count'])

user_mean_final.columns = ["answered_correctly_user_mean", 'count']
#saving value to fillna

elapsed_mean = train_csv.prior_question_elapsed_time.mean()
'''

from datetime import datetime

train_csv['timestamp'] = pd.to_datetime(train_csv['timestamp'], unit='ms',origin='2017-1-1')

train_csv['month']=(train_csv.timestamp.dt.month)

aveg = train_csv[['user_id','month','prior_question_elapsed_time']].groupby(['user_id','month']).mean()

aveg.columns=['mean']

'''
import gc

gc.collect()
# for each user, the last 3 intereaction will be the validation set 

validation = pd.DataFrame()

for i in range(4):

    last_records = train_csv.drop_duplicates('user_id', keep = 'last')

    train_csv = train_csv[~train_csv.index.isin(last_records.index)]

    validation = validation.append(last_records)
train_csv = pd.merge(train_csv, user_mean_final , on=['user_id'], how="left")

train_csv= pd.merge(train_csv, content_mean_final, on=['content_id'], how="left")



validation = pd.merge(validation, user_mean_final, on=['user_id'], how="left")

validation = pd.merge(validation,content_mean_final , on=['content_id'], how="left")
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()



train_csv.prior_question_had_explanation.fillna(False, inplace = True)

validation.prior_question_had_explanation.fillna(False, inplace = True)



validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])

train_csv["prior_question_had_explanation_enc"] = lb_make.fit_transform(train_csv["prior_question_had_explanation"])
questions_csv = questions_csv.drop(columns = ['bundle_id'])

questions_csv = questions_csv.drop(columns = ['correct_answer'])

questions_csv = questions_csv.drop(columns = ['tags'])
train_csv= pd.merge(train_csv, questions_csv, left_on = 'content_id', right_on = 'question_id', how = 'left')

validation = pd.merge(validation, questions_csv, left_on = 'content_id', right_on = 'question_id', how = 'left')
y_train = train_csv['answered_correctly']

train_csv = train_csv.drop(['answered_correctly'], axis=1)



y_val = validation['answered_correctly']

validation = validation.drop(['answered_correctly'], axis=1)
train_csv = train_csv[['answered_correctly_user_mean', 'answered_correctly_content_mean', 'count',

       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']]

validation = validation[['answered_correctly_user_mean', 'answered_correctly_content_mean', 'count',

       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']]
train_csv['prior_question_elapsed_time'].fillna(train_csv['prior_question_elapsed_time'].mean(), inplace = True)

validation['prior_question_elapsed_time'].fillna(validation['prior_question_elapsed_time'].mean(), inplace = True)
import lightgbm as lgb



params = {

    'objective': 'binary',

    'max_bin': 700,

    'learning_rate': 0.0175,

    'num_leaves': 80

}



lgb_train = lgb.Dataset(train_csv, y_train, categorical_feature = ['part', 'prior_question_had_explanation_enc'])

lgb_eval = lgb.Dataset(validation, y_val, categorical_feature = ['part', 'prior_question_had_explanation_enc'], reference=lgb_train)

model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=50, # ogni quanti cicli mostra il valore ottenuto 

    num_boost_round=1000,

    early_stopping_rounds=10

)
y_pred = model.predict(validation)

y_true = np.array(y_val)

roc_auc_score(y_true, y_pred)
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = pd.merge(test_df, questions_csv, left_on = 'content_id', right_on = 'question_id', how = 'left')

    test_df = pd.merge(test_df, user_mean_final, on=['user_id'],  how="left")

    test_df = pd.merge(test_df, content_mean_final, on=['content_id'],  how="left")

    test_df['answered_correctly_user_mean'].fillna(0.5,  inplace=True)

    test_df['answered_correctly_content_mean'].fillna(0.5,  inplace=True)

    #test_df['part'] = test_df.part - 1



    test_df['part'].fillna(int(test_df['part'].mean()), inplace = True)

    test_df['count'].fillna(0, inplace=True)

    test_df['prior_question_elapsed_time'].fillna(test_df['prior_question_elapsed_time'].mean(), inplace = True)

    test_df['prior_question_had_explanation'].fillna(False, inplace=True)

    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])

    test_df['answered_correctly'] =  model.predict(test_df[['answered_correctly_user_mean', 'answered_correctly_content_mean','count',

                                                                  'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
print('finish :)')