import riiideducation

import pandas as pd



# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', usecols=[1, 2, 3,7],

                       dtype={'timestamp': 'int64', 'user_id': 'int64' ,'content_id': 'int16','answered_correctly':'int8'}

                      )
content_acc = train_df.query('answered_correctly != -1').groupby('content_id')['answered_correctly'].mean().to_dict()

user_acc = train_df.query('answered_correctly != -1').groupby('user_id')['answered_correctly'].mean().to_dict()

# explanation_acc = train_df.query('answered_correctly != -1').groupby('prior_question_had_explanation')['answered_correctly'].mean().to_dict()

# task_acc = train_df.query('answered_correctly != -1').groupby('task_container_id')['answered_correctly'].mean().to_dict()

iter_test = env.iter_test()
def add_user_acc(x):

    if x in user_acc.keys():

        return user_acc[x]

    else:

        return 0.5

    

def add_content_acc(x):

    if x in content_acc.keys():

        return content_acc[x]

    else:

        return 0.5



def add_task_acc(x):

    if x in task_acc.keys():

        return task_acc[x]

    else:

        return 0.5



def add_explanation_acc(x):

    if x in explanation_acc.keys():

        return explanation_acc[x]

    else:

        return 0.5 





for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly1'] = test_df['user_id'].apply(add_user_acc).values

    test_df['answered_correctly2'] = test_df['content_id'].apply(add_content_acc).values

#     test_df['answered_correctly3'] = test_df['task_container_id'].apply(add_task_acc).values

#     test_df['answered_correctly4'] = test_df['prior_question_had_explanation'].apply(add_explanation_acc).values

    test_df['answered_correctly'] = 0.5*test_df['answered_correctly1']+0.5*test_df['answered_correctly2']#+0.15*test_df['answered_correctly3']+0.15*test_df['answered_correctly4']

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])