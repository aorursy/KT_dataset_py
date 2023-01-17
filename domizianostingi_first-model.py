import riiideducation

import numpy as np 

import pandas as pd 



from sklearn.metrics import roc_auc_score

from  sklearn.tree import DecisionTreeClassifier

from  sklearn.model_selection import train_test_split



env = riiideducation.make_env()
train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', 

                      usecols=[1,2,3,4,5,7,8,9],nrows=50000000)

lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

example_sample_submission = pd.read_csv('../input/riiid-test-answer-prediction/example_sample_submission.csv')
train_df = train_df[train_df['content_type_id'] == 0]

#keeping just the questions 



train_df= train_df.drop(columns=['content_type_id'])

#dropping the column content_type_id and the answer of the users column
results_c = train_df[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])

results_c.columns = ["answered_correctly_content"]



results_u = train_df[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum', 'count'])

results_u.columns = ['answered_correctly_user', 'sum', 'count']
train_df =train_df.sort_values(['timestamp'], ascending=True)

train_df = train_df.iloc[10000000:,:]

# sort the dataset by timestamp and than take only the last N observation. In this way all the values with timestamp = 0 are removed, and the db is 

# more easy to treat    
train_df = pd.merge(train_df, results_u, on=['user_id'], how="left")

train_df = pd.merge(train_df, results_c, on=['content_id'], how="left")
questions = questions.rename(columns={'question_id':'content_id'})

train_df = train_df.merge(questions)

train_df= train_df.drop(columns=['correct_answer'])

train_df= train_df.drop(columns=['bundle_id'])

#merging together the 2 db and dropping the column correct_answer and bundle_id
train_df['task_container_id'] = (

    train_df

    .groupby('user_id')['task_container_id']

    .transform(lambda x : pd.factorize(x)[0])

    .astype('int16')

)

#this is a function that assure the monotonicity of task container id
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

train_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(train_df["prior_question_had_explanation"])

train_df = train_df.drop(columns=['prior_question_had_explanation'])

#this is just for encoding 0-1 the variable prior question had explanation
train_df = train_df.sort_values(by=['user_id'])

# sorting by user_id
train_df=train_df.drop(columns=['part'])

train_df=train_df.drop(columns=['tags'])



#I remove part and tag, I put these command here because in a second moment I would like to integrete these 2 variables, it could be useful.
from datetime import datetime

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='ms',origin='2017-1-1')

train_df['month']=(train_df.timestamp.dt.month)

train_df['day']=(train_df.timestamp.dt.day)

#i trasform timestamp in date format, then I extrapolte month and day to generate 2 columns



aveg = train_df[['user_id','month','day','prior_question_elapsed_time']].groupby(['user_id','month','day']).mean()/1000

aveg.columns=['mean']

#with the 2 columns generated before it is now possible 

#to calculate the average elapsed time for each user for each month for each day. 



train_df = pd.merge(train_df, aveg, on=['user_id','month','day'], how='left')

# merge the 2 db
y = train_df[["answered_correctly"]]

# extrapolate the dependent variable 
train_df.isnull().sum(axis = 0)

#checking for any missing value
keep = ['prior_question_had_explanation_enc',

        'mean', 

        'answered_correctly_user',

        'sum', 

        'count',

        'answered_correctly_content']

x=train_df[keep]
x.head(20)
Xt, Xv, Yt, Yv = train_test_split(x, y, test_size =0.2, shuffle=False)

# split train in train and validation 



import lightgbm as lgb



'''

https://lightgbm.readthedocs.io/en/latest/Parameters.html

'''



params = {

    'objective': 'binary', #specify how is the dependet variable, binary can be used for logistic regression or log loss classification

    'max_bin': 600, #max number of bins that features values will be bucketed in. Small number may reduce training accuracy but may increase general power

    'learning_rate': 0.02, #learning_rate refers to the step size at each interation while moving toward an optimal point

    'num_leaves': 80 # maximum number of leaves in a tree, where a leave is a final termination of a tree

}





lgb_train = lgb.Dataset(Xt, Yt)

lgb_eval = lgb.Dataset(Xv, Yv, reference=lgb_train)

#lightgbm need to take as argument lightgbm dataset, it is required to make this trasformation



model = lgb.train(

    params, lgb_train, #it is required to insert the parameters, then the train set

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000, # number of boosting iterations 

    early_stopping_rounds=10 # will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds, so if 

    #  for ten 'epochs' the model will stop, in this way the num_boost_round is a maximum value.  

)  
y_pred = model.predict(Xv)

y_true = np.array(Yv)

roc_auc_score(y_true, y_pred)
example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

example_test["prior_question_had_explanation_enc"] = lb_make.fit_transform(example_test["prior_question_had_explanation"])

example_test.head(10)
example_test['timestamp'] = pd.to_datetime(example_test['timestamp'], unit='ms',origin='2017-1-1')

example_test['month']=(example_test.timestamp.dt.month)

example_test['day']=(example_test.timestamp.dt.day)

aveg = example_test[['user_id','month','day','prior_question_elapsed_time']].groupby(['user_id','month','day']).mean()/1000

aveg.columns=['mean']



example_test = pd.merge(example_test, results_u, on=['user_id'], how="left")

example_test = pd.merge(example_test, results_c, on=['content_id'], how="left")

example_test = pd.merge(example_test, aveg, on=['user_id','month','day'], how='left')

example_test = example_test[keep]

example_test.head(10)

'''

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)

imp.fit(example_test)

example_test = pd.DataFrame(imp.transform(example_test), columns = example_test.columns)

'''
example_test['answered_correctly_user'].fillna(example_test['answered_correctly_user'].mean(), inplace=True)

example_test['sum'].fillna(example_test['sum'].mean(), inplace=True)

example_test['count'].fillna(example_test['count'].mean(), inplace=True)
example_test.head(10)
y_pred = model.predict(example_test[keep])

example_test['answered_correctly'] = y_pred
'''

    imp = IterativeImputer(max_iter=10, random_state=0)

    imp.fit(test_df)

    test_df = pd.DataFrame(imp.transform(test_df), columns = test_df.columns)

'''



iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

    test_df['prior_question_had_explanation'].fillna(False, inplace=True)

    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])

    

    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='ms',origin='2017-1-1')

    test_df['month']=(test_df.timestamp.dt.month)

    test_df['day']=(test_df.timestamp.dt.day)

    avegm = test_df[['user_id','month','day','prior_question_elapsed_time']].groupby(['user_id','month','day']).mean()/1000

    avegm.columns=['mean']

    

    test_df = pd.merge(test_df, results_u, on=['user_id'],  how="left")

    test_df = pd.merge(test_df, results_c, on=['content_id'],  how="left")

    test_df = pd.merge(test_df, avegm, on=['user_id','month','day'], how='left')

    

    test_df['answered_correctly_user'].fillna(test_df['answered_correctly_user'].mean(), inplace=True)

    test_df['sum'].fillna(test_df['sum'].mean(), inplace=True)

    test_df['count'].fillna(test_df['count'].mean(), inplace=True)

    

    test_df['answered_correctly'] =  model.predict(test_df[keep])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
print('finish')