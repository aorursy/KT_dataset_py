import pandas as pd

import numpy as np

import random



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import  tqdm
%%time



data_type_dict = {'row_id': 'int64',

                  'timestamp': 'int64',

                  'user_id': 'int32',

                  'content_id': 'int16',

                  'content_type_id': 'int8',

                  'task_container_id': 'int16',

                  'user_answer': 'int8',

                  'answered_correctly': 'int8',

                  'prior_question_elapsed_time': 'float32', 

                  'prior_question_had_explanation': 'boolean'}



train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=True,

                       dtype=data_type_dict,

                       nrows = 10**7)
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',index_col=0)

questions_df = questions_df.fillna(value={'tags':'-1'})

questions_df['correct_answer']=questions_df['correct_answer'].astype(np.int8)

questions_df['part']=questions_df['part'].astype(np.int8)
test_split = True

training_set_ratio = 0.75
if test_split==True:

    train_df = train_df.merge(pd.DataFrame(train_df.groupby('user_id')['task_container_id'].agg('max')).rename(columns={'task_container_id':'task_container_id_max'}),

                          left_on='user_id',right_index=True)





    train_df['for_training'] = train_df['task_container_id'].values<=training_set_ratio*train_df['task_container_id_max'].values



    train_df = train_df.drop('task_container_id_max',axis=1)





    test_df = train_df.loc[~train_df['for_training']]

    train_df = train_df.loc[train_df['for_training']]



    train_df = train_df.drop('for_training',axis=1)

    test_df = test_df.drop('for_training',axis=1)
test_df['group_no']=0



out_col_dict = {name:ii for ii,name in enumerate(test_df.columns)}



for ii in tqdm(range(1,test_df.shape[0])):

    if (test_df.iat[ii,out_col_dict['user_id']]==test_df.iat[ii-1,out_col_dict['user_id']]):

        if (test_df.iat[ii,out_col_dict['task_container_id']]!=test_df.iat[ii-1,out_col_dict['task_container_id']]):

           test_df.iat[ii,out_col_dict['group_no']] = 1 + test_df.iat[ii-1,out_col_dict['group_no']]

        else:

            test_df.iat[ii,out_col_dict['group_no']] = test_df.iat[ii-1,out_col_dict['group_no']]



    else:

        test_df.iat[ii,out_col_dict['group_no']] = (ii//50_000)*100
columns_index_expected = ['row_id','group_no','timestamp', 'user_id', 'content_id', 'content_type_id',

                           'task_container_id', 'prior_question_elapsed_time',

                           'prior_question_had_explanation', 'prior_group_responses',

                           'prior_group_answers_correct']
groups = test_df.groupby('group_no')



group_lengths = []

test_groups_list = []

for ii,frame in tqdm(groups):

    group_lengths.append([ii,len(frame['user_id'].unique())])

    test_groups_list.append(frame)

    

for ii in tqdm(range(len(test_groups_list))):

    test_groups_list[ii]['prior_group_answers_correct'] = np.nan

    test_groups_list[ii]['prior_group_responses'] = np.nan 

    if ii>0:

        test_groups_list[ii].loc[test_groups_list[ii].index[0],'prior_group_responses'] = str(list(test_groups_list[ii-1]['user_answer'].values.astype(np.int8)))

        test_groups_list[ii].loc[test_groups_list[ii].index[0],'prior_group_answers_correct'] = str(list(test_groups_list[ii-1]['answered_correctly'].values.astype(np.int8)))

        test_groups_list[ii-1].drop(['user_answer','answered_correctly'],axis=1,inplace=True)

        test_groups_list[ii-1] = test_groups_list[ii-1][columns_index_expected]

    else:

        test_groups_list[ii].loc[test_groups_list[ii].index[0],'prior_group_responses'] = str('[]')

        test_groups_list[ii].loc[test_groups_list[ii].index[0],'prior_group_answers_correct'] = str('[]')

    

    

test_groups_list[-1] = test_groups_list[-1][columns_index_expected]
test_group_counts_df = pd.DataFrame(group_lengths,columns=['group_no','counts'])



fig,ax=plt.subplots(1,1,figsize=(16,8))

sns.distplot(test_group_counts_df['counts'],ax=ax,kde_kws={"cut":3});
# Intial groups have lots of bundles



temp = random.choice(test_groups_list[:10])

print(f'This Group has {len(temp["user_id"].unique())} bundle/bundles')

display(temp)
# Last groups have only a few bundles



temp = random.choice(test_groups_list[-10:])

print(f'This Group has {len(temp["user_id"].unique())} bundle/bundles')

display(temp)
for test_group_single in tqdm(test_groups_list):

        test_group_single = test_group_single.merge(questions_df,how='left',left_on='content_id',right_index=True)

        test_group_single['timestamp'] = (test_group_single['timestamp']//1000)



        test_group_single['part'] =  test_group_single['part'].fillna(0).astype(dtype=np.int8)

        test_group_single['bundle_id'] = test_group_single['bundle_id'].fillna(0).astype(dtype=np.uint16)

        

        for ii in range(test_group_single.shape[0]):

            # make predictions here for each user_id or row_id

            pass