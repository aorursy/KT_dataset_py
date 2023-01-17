# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report



from lightgbm import LGBMClassifier

from lightgbm import plot_importance, plot_metric, early_stopping
questions = pd.read_csv('../input/feature-engineering-datasets/questions_fe.csv')

question_stats = pd.read_csv('../input/feature-engineering-datasets/questions_stats_fe.csv')
#given the dataset, I retrieve the cumulative history of students

def get_last_performance( data,last_record=pd.DataFrame()):

    last_record.index= -1 - last_record.index

    df = data[data['content_type_id']==0].merge(questions[['question_id','part','kmean_cluster']], left_on='content_id', right_on = 'question_id', how = 'left')

    df = df[['timestamp','user_id','answered_correctly','part','row_id','kmean_cluster']]

    df['part_answer'] = 'part_'+ df.part.astype(str)  + '_' + df.answered_correctly.astype(str)

    df['cluster_answer'] = 'cluster_'+ df.kmean_cluster.astype(str)  + '_' + df.answered_correctly.astype(str)

    

    user_performance = df[['user_id','row_id']].merge(

        pd.get_dummies(df['part_answer']),

        left_index = True, 

        right_index = True).merge(

        pd.get_dummies(df['cluster_answer']),

        left_index = True, 

        right_index = True).set_index('row_id')

    try:

        user_performance = user_performance.append(last_record.drop(columns=['timestamp'],inplace=False))

    except:

            None

    user_performance = user_performance.groupby('user_id').cumsum().merge(

        df[['user_id','row_id','timestamp']].set_index('row_id'), left_index = True, right_index = True)

    user_performance_part_last = last_record.append(user_performance).groupby('user_id',as_index=False).max().fillna(0)

    

    return user_performance, user_performance_part_last
target_col = ['answered_correctly']

feature_col = [

 'timestamp',

 'prior_question_elapsed_time',

 'part_1_avg',

 'part_1_count',

 'part_2_avg',

 'part_2_count',

 'part_3_avg',

 'part_3_count',

 'part_4_avg',

 'part_4_count',

 'part_5_avg',

 'part_5_count',

 'part_6_avg',

 'part_6_count',

 'part_7_avg',

 'part_7_count',

 'cluster_0_avg',

 #'cluster_0_count',

 'cluster_1_avg',

 #'cluster_1_count',

 'cluster_2_avg',

 #'cluster_2_count',

 'cluster_3_avg',

 #'cluster_3_count',

 'cluster_4_avg',

 #'cluster_4_count',

    'cluster_5_avg',

 #'cluster_5_count',

 'cluster_6_avg',

 #'cluster_6_count',

 'bundle_mean_answered_correctly',

 'bundle_std_answered_correctly',

 'bundle_perc_students',

 'bundle_times',

 'part_mean_answered_correctly',

 'part_std_answered_correctly',

 'part_perc_students',

 'part_times',

 'cluster_mean_answered_correctly',

 'cluster_std_answered_correctly',

 'cluster_perc_students',

 'cluster_times']
lgbm = LGBMClassifier(

    num_leaves = 80,

    max_depth = 7,

    n_estimators = 100,

    min_child_samples = 1000, 

    subsample=0.7, 

    subsample_freq=5,

    n_jobs= -1,

    min_data_in_leaf = 100,

    is_higher_better = True,

    first_metric_only = True,

    feature_fraction = 0.8,

    reg_alpha = 0.3,

    learning_rate = 0.1

)
import gc



iter_df =  pd.read_csv('../input/riiid-test-answer-prediction/train.csv',chunksize = 1000000)



user_last_performance = pd.DataFrame()

debug = []

score_old = 0



for i,df in enumerate(iter_df):

    #print('Running iteration: {}'.format(i))

    gc.collect()

    user_stats = df.loc[df['content_type_id']==0,['row_id','user_id','content_id','timestamp','answered_correctly','prior_question_elapsed_time']]

    user_performance, user_last_performance = get_last_performance(df, user_last_performance)

    data = user_stats.merge(

        user_performance.reset_index().rename(columns={'index':'row_id'}),

        on =['row_id','user_id','timestamp'], how = 'inner'

    ).merge(

        question_stats, left_on = 'content_id', right_on = 'question_id'

    ).set_index('row_id')

    

    #Rescaling some features

    

    data.timestamp/=(365*24*60*60*100)

    data.prior_question_elapsed_time/=(60*60*24)

    data = data.fillna(-1)

    

    # Calculate stats performance on parts

    for part in questions.part.unique():

        data['part_'+str(part)+'_avg'] =  data['part_'+str(part)+'_1'].divide(data['part_'+str(part)+'_0'] + data['part_'+str(part)+'_1'])

        data['part_'+str(part)+'_count'] = data['part_'+str(part)+'_0'] + data['part_'+str(part)+'_1']

        data.drop(columns = ['part_'+str(part)+'_0', 'part_'+str(part)+'_1'],inplace = True)

    data = data.fillna(0.5)

    

    # Calculate stats performance on clusters

    

    for part in questions.kmean_cluster.unique():

        data['cluster_'+str(part)+'_avg'] =  data['cluster_'+str(part)+'_1'].divide(data['cluster_'+str(part)+'_0'] + data['cluster_'+str(part)+'_1'])

        data['cluster_'+str(part)+'_count'] = data['cluster_'+str(part)+'_0'] + data['cluster_'+str(part)+'_1']

        data.drop(columns = ['cluster_'+str(part)+'_0', 'cluster_'+str(part)+'_1'],inplace = True)

    data = data.fillna(0.5)

    

    # Model Training & Validation in Loop

    

    X_train,X_test,Y_train, Y_test = train_test_split(data[feature_col], data[target_col], test_size = 0.33, shuffle =True)

    

    

    

    if i == 0:

        lgbm.fit(X_train, Y_train,callbacks = [early_stopping])

    score_train = roc_auc_score(Y_train.values, lgbm.predict_proba(X_train)[:,1])

    if ((score_old - score_train > 0.15) | (score_train < 0.745)) & (i%25==0):

        lgbm.fit(X_train, Y_train,callbacks = [early_stopping])

        score_train = roc_auc_score(Y_train.values, lgbm.predict_proba(X_train)[:,1])

    score_test = roc_auc_score(Y_test.values, lgbm.predict_proba(X_test)[:,1])

    print('\nRun: {}'.format(i))

    print('Score Train: {}'.format(score_train))

    print('Score Test: {}'.format(score_test))



    debug.append([i, len(df),len(user_stats),len(user_performance),len(user_last_performance),score_train, score_test])

    score_old = score_train

    

    del user_performance, score_train, score_test, data, X_train, X_test, Y_train, Y_test

    #print('\nClosed iteration: {}'.format(i))

    #if i == 10:

    #    break



debug_df = pd.DataFrame(debug, columns = [

    'run', 'len_chunk', 'len_user_stats', 'len_performance','num_users','score_train','score_test'

])



debug_df[['score_train','score_test']].plot.line()    



plot_importance(lgbm, figsize  = (18,10))
# salvataggio del modello lgbm

lgbm.booster_.save_model('./model.txt')

# salvataggio user_last_performance

user_last_performance.to_csv('./user_stats.csv', index = False)
import riiideducation



env = riiideducation.make_env()

iter_test = env.iter_test()
def enrich_dataset(dataset, last_record,answers = 0):

    data = dataset[['row_id','user_id','content_id','timestamp','prior_question_elapsed_time']]

    data = data.merge(

        question_stats, left_on = 'content_id', right_on = 'question_id', how = 'left'

    ).merge(

        last_record.drop(columns = ['timestamp'],inplace=False),

        on =['user_id'], how = 'left'

    ).fillna(0).set_index('row_id')

    

    data.timestamp/=(365*24*60*60*100)

    data.prior_question_elapsed_time/=(60*60*24)

    

    # Calculate stats performance on parts

    for part in questions.part.unique():

        data['part_'+str(part)+'_avg'] =  data['part_'+str(part)+'_1'].divide(data['part_'+str(part)+'_0'] + data['part_'+str(part)+'_1'])

        data['part_'+str(part)+'_count'] = data['part_'+str(part)+'_0'] + data['part_'+str(part)+'_1']

        data.drop(columns = ['part_'+str(part)+'_0', 'part_'+str(part)+'_1'],inplace = True)

    data = data.fillna(0.5)

    

    # Calculate stats performance on clusters

    

    for part in questions.kmean_cluster.unique():

        data['cluster_'+str(part)+'_avg'] =  data['cluster_'+str(part)+'_1'].divide(data['cluster_'+str(part)+'_0'] + data['cluster_'+str(part)+'_1'])

        data['cluster_'+str(part)+'_count'] = data['cluster_'+str(part)+'_0'] + data['cluster_'+str(part)+'_1']

        data.drop(columns = ['cluster_'+str(part)+'_0', 'cluster_'+str(part)+'_1'],inplace = True)

    data = data.fillna(0.5)

    return data
for (test_df,sample_prediction_df) in iter_test:

    data = enrich_dataset(test_df, user_last_performance)

    test_df[ 'answered_correctly'] = lgbm.predict_proba(data[feature_col])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0,['row_id', 'answered_correctly']])

    test_df[ 'answered_correctly'] = lgbm.predict_proba(data[feature_col])

    print(len(test_df)==len(data))

    print(test_df['answered_correctly'])

    wait = input()

    _, user_last_performance = get_last_performance(test_df.loc[test_df['answered_correctly']==0, : ], user_last_performance)