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
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl > /dev/null



import lightgbm as lgb

import datatable as dt
lgbm = lgb.Booster(model_file = '../input/model-building-with-cumulative-features/model.txt')
user_last_performance = dt.fread('../input/model-building-with-cumulative-features/user_stats.csv')

question_stats = dt.fread('../input/feature-engineering-datasets/questions_stats_fe.csv')

questions = dt.fread('../input/feature-engineering-datasets/questions_fe.csv')

questions.names={'question_id':'content_id'}

questions.key='content_id'

question_stats.names={'question_id':'content_id'}

question_stats.key='content_id'

user_last_performance.key='user_id'
def enrich_dataset(dataset, last_record,answers = 0):

    data = dt.Frame(

        dataset[['row_id','user_id','content_id','timestamp','prior_question_elapsed_time']].fillna(-60*60*24)

    )

    row_id, user_id, content_id, timestamp, prior_question_elapsed_time = data.export_names()

    data = data[:,{'row_id':row_id, 

                   'user_id':user_id, 

                   'content_id':content_id, 

                   'timestamp':timestamp/(365*24*60*60*100), 

                   'prior_question_elapsed_time':prior_question_elapsed_time/(60*60*24)}]

    data = data[:,:,dt.join(question_stats)]

    X = user_last_performance[:, dt.f[:].remove(dt.f['timestamp'])]

    X.key = 'user_id'

    data = data[:,:,dt.join(X)]

    

    del X



    for part in dt.unique(questions[:,'part']).to_list()[0]:

        col1 = 'part_'+str(part)+'_avg'

        col2 = 'part_'+str(part)+'_count'

        part_1 = 'part_'+str(part)+'_1'

        part_0 = 'part_'+str(part)+'_0'

        data[:,col1] = data[:,dt.f[part_1]/(dt.f[part_0]+dt.f[part_1])]

        data[:,col2] = data[:,dt.f[part_0] + dt.f[part_1]]

        del data[:, part_1]

        del data[:, part_0]

    for part in dt.unique(questions[:,'kmean_cluster']).to_list()[0]:

        col1 = 'cluster_'+str(part)+'_avg'

        col2 = 'cluster_'+str(part)+'_count'

        part_1 = 'cluster_'+str(part)+'_1'

        part_0 = 'cluster_'+str(part)+'_0'

        data[:,col1] = data[:,dt.f[part_1]/(dt.f[part_0]+dt.f[part_1])]

        data[:,col2] = data[:,dt.f[part_0] + dt.f[part_1]]

        del data[:, part_1]

        del data[:, part_0]

    data = data.to_pandas().fillna(0.5)

    return data
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
import riiideducation



env = riiideducation.make_env()

iter_test = env.iter_test()
def get_last_performance( data,last_record=pd.DataFrame()):

    data = dt.Frame(test_df.loc[test_df['content_type_id']==0,['timestamp','user_id','answered_correctly','row_id','content_id']])

    data.key = 'row_id'

    data = data[:,:,dt.join(questions)]

    del data[:,['content_id','bundle_id','tags']]

    data.key = 'row_id'

    data = data.to_pandas()

    data['part_answer'] = 'part_'+ data.part.astype(str)  + '_' + data.answered_correctly.astype(str)

    data['cluster_answer'] = 'cluster_'+ data.kmean_cluster.astype(str)  + '_' + data.answered_correctly.astype(str)

    

    user_performance = data[['user_id','row_id']].join(

        pd.get_dummies(data['part_answer']), how = 'left').join(

        pd.get_dummies(data['cluster_answer']),how = 'left').set_index('row_id')

    last_record = user_last_performance.to_pandas()

    last_record.index=  - last_record.index

    try:

        user_performance = user_performance.append(last_record.drop(columns=['timestamp'],inplace=False))

    except:

        None

    user_performance = user_performance.groupby('user_id').cumsum().join(

        data[['user_id','row_id','timestamp']].set_index('row_id'),how = 'left')

    user_performance_part_last = last_record.append(user_performance).groupby('user_id',as_index=False).max().fillna(0).astype({'user_id':'int64','timestamp':'int64'})

    user_performance_part_last = dt.Frame(user_performance_part_last)

    user_performance_part_last.key='user_id'

    return user_performance_part_last
import gc

for (test_df,sample_submission) in iter_test:

    gc.collect()

    #print(len(test_df.loc[test_df['content_type_id']==1,:]))

    data = enrich_dataset(test_df, user_last_performance)

    #print('Len data: ',len(data))

    #print('Len test: ',len(test_df))

    test_df[ 'answered_correctly'] = lgbm.predict(data[feature_col].fillna(-1))

    test_df[ 'answered_correctly'] = test_df[ 'answered_correctly'].astype('float64')

    #print('Len test questions: ',len(test_df.loc[test_df['content_type_id']==0,:]))

    #print(test_df.info())

    #print(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    #print(sample_submission.info())

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    test_df[ 'answered_correctly'] = test_df[ 'answered_correctly'].apply(lambda x: 0 if (x < 0.5)&(x>=0) else 1)

    #print(test_df.loc[test_df['content_type_id']==0,:])

    user_last_performance = get_last_performance(test_df.loc[test_df['content_type_id']==0,:], user_last_performance)