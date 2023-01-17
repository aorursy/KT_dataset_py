# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (b|y clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold,KFold



import lightgbm as lgb

from sklearn.metrics import roc_auc_score
# %%time



# # reading the dataset from raw csv file

# import datatable as dt



# dt.fread("train.csv").to_jay("train.jay")



# train_df = dt.fread("train.jay").to_pandas()



# print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)

# train_df

train_df = pd.read_csv(

    '../input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    nrows=10**6, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8', 

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)

expTest = pd.read_csv("../input/riiid-test-answer-prediction/example_test.csv")

questions=pd.read_csv("../input/riiid-test-answer-prediction/questions.csv")

lectures=pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv")
train_df
train_df['user_id'].nunique()
train_df.nunique()
lectures
questions
train_lect = train_df[train_df['content_type_id']==1]

train_ques = train_df[train_df['content_type_id']==0]
train_lect.shape
train_ques.shape
# train_lect = pd.merge(train_lect,lectures,left_on='content_id',right_on='lecture_id',how='left')

train_ques = pd.merge(train_ques,questions,left_on='content_id',right_on='question_id',how='left')
#Creating folds

kf = KFold()

train_ques.loc[:,'fold'] = -1

for fold,(trn_idx,val_idx) in enumerate(kf.split(train_ques,train_ques['answered_correctly'].values)):

    train_ques.loc[val_idx,'fold'] = fold
train_ques.nunique()
train_ques["answered_correctly"].value_counts()
train_ques['task_container_id'].nunique()
elapsed_mean = train_ques['prior_question_elapsed_time'].mean()
train_ques['prior_question_had_explanation'].fillna(0,inplace=True)

train_ques['prior_question_had_explanation']=train_ques['prior_question_had_explanation'].astype(int)

train_ques['prior_question_elapsed_time'].fillna(elapsed_mean,inplace=True)
train_ques.isna().sum()
train_ques.columns
train_ques.sample(10)
questions

def arrangeGroup(group,prefix):

    group=group.reset_index()

    cols = [prefix.join(x).strip() for x in group.columns.values]

    group =pd.DataFrame(group.to_records()).drop(columns=['index'])

    group.columns=cols

    group.reset_index(inplace=True)

    group=group.drop(columns='index')

    return group
def targetEncodingUser():

    # User aggs

    aggs= {

        "prior_question_elapsed_time":["mean","max","std"],

        "user_id":"size",

#         "timestamp":"var",

        "answered_correctly":["mean","sum","skew"]

    }

    group_user = train_ques.groupby(by="user_id").agg(aggs)

    group_user = arrangeGroup(group_user,'_user_')

#     group_user=group_user.reset_index()

#     cols = [" ".join(x).strip() for x in group_user.columns.values]

#     group_user =pd.DataFrame(group_user.to_records()).drop(columns=['index'])

#     group_user.columns=cols

    return group_user
def targetEncodingQuest():

    quest_aggs = {

    'answered_correctly' :["count","mean"],

#     'user_answer':lambda x:x.value_counts().index[0]

    "prior_question_elapsed_time":["mean","skew","std"]

    }

    bundle_aggs = {

        'answered_correctly' :["count","mean"],

        "prior_question_elapsed_time":["mean","skew"]

    }

    group_quest = train_ques.groupby(by="question_id").agg(quest_aggs)

    group_bundle = train_ques.groupby(by="bundle_id").agg(bundle_aggs)

    group_quest  = arrangeGroup(group_quest,'_quest_')

    group_bundle = arrangeGroup(group_bundle,'_bundle_')

#     group_quest=group_quest.reset_index()

#     cols = [' '.join(x).strip() for x in group_quest.columns.values]

#     group_quest =pd.DataFrame(group_quest.to_records()).drop(columns=['index'])

#     group_quest.columns=cols

#     group_quest.reset_index(inplace=True)

#   group_quest.columns=['question_id','nbAnswered_correctlyQuest','meanAnswered_correctlyQuest','user_answerFreq']

    

    

    return group_quest,group_bundle
def targetEncodingContent():

    content_aggs = {

    'answered_correctly' :["count","mean"],

    }



    group_content = train_ques.groupby(by="content_id").agg(content_aggs)

    group_content  = arrangeGroup(group_content,'_content_')

    

    return group_content
def targetEncodingContentUser():

    contentUser_aggs = {

    'answered_correctly' :["count","mean"],

    }



    group_contentUser = train_ques.groupby(by=["user_id","content_id"]).agg(contentUser_aggs)

    group_contentUser  = arrangeGroup(group_contentUser,'_contentUser_')

    

    return group_contentUser
%%time

group_quest,group_bundle= targetEncodingQuest()

group_user= targetEncodingUser()

group_content = targetEncodingContent()

group_contentUser = targetEncodingContentUser()



group_user['user_id_user_'] = group_user['user_id_user_'].astype(np.int32)
group_contentUser
group_quest.dtypes
train_ques.dtypes
del(train_df)
dd = train_ques.copy()

dd=pd.merge(dd,group_quest,left_on='question_id',right_on="question_id_quest_",how='left')

dd=pd.merge(dd,group_bundle,left_on='bundle_id',right_on="bundle_id_bundle_",how='left')



dd=pd.merge(dd,group_user,left_on='user_id',right_on='user_id_user_',how='left')

dd=pd.merge(dd,group_content,left_on='content_id',right_on='content_id_content_',how='left')



# dd=pd.merge(dd,group_contentUser,left_on=['user_id','content_id'],right_on=['user_id_contentUser_',"content_id_contentUser_"],how='left')



dd.head()
dd.columns

todel = ['row_id','user_id','content_type_id','content_id','task_container_id', 'user_answer', 'answered_correctly','question_id', 'bundle_id', 'correct_answer', 'part', 'tags', 'fold',

       'index']
cols = list(set(list(dd.columns)) - set(list(todel)))
cols
len(cols)
del dd
# cols =['timestamp',

#        'prior_question_elapsed_time', 'prior_question_had_explanation',

#         'nbAnswered_correctlyQuest', 'meanAnswered_correctlyQuest',

#        'user_answerFreq', 'prior_question_elapsed_time mean',

#        'prior_question_elapsed_time max', 'prior_question_elapsed_time median',

#        'prior_question_elapsed_time std', 'user_id size', 'timestamp var',

#        'answered_correctly size', 'answered_correctly mean',

#        'answered_correctly sum', 'answered_correctly skew',

#        'answered_correctly std', 'answered_correctly median']
params = {'num_leaves': 32,

          'max_bin': 300,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "metric": 'auc',

         }





# params = {

#     'objective': 'binary',

#     'max_bin': 700,

#     'learning_rate': 0.0175,

#     'num_leaves': 80

# }

#Training

feature_importance=pd.DataFrame()

def run(fold,feature_importance):

    clf = lgb.LGBMClassifier(**params,n_estimators=700)

    df_train = train_ques[train_ques['fold']==k]

    df_val = train_ques[train_ques['fold']!=k]

    

    

    df_train=pd.merge(df_train,group_quest,left_on='question_id',right_on="question_id_quest_",how='left')

    df_train=pd.merge(df_train,group_bundle,left_on='bundle_id',right_on="bundle_id_bundle_",how='left')

    df_train=pd.merge(df_train,group_user,left_on='user_id',right_on='user_id_user_',how='left')

    df_train=pd.merge(df_train,group_content,left_on='content_id',right_on='content_id_content_',how='left')

#     df_train=pd.merge(df_train,group_contentUser,left_on=['user_id','content_id'],right_on=['user_id_contentUser_',"content_id_contentUser_"],how='left')



    df_val=pd.merge(df_val,group_quest,left_on='question_id',right_on="question_id_quest_",how='left')

    df_val=pd.merge(df_val,group_bundle,left_on='bundle_id',right_on="bundle_id_bundle_",how='left')

    df_val=pd.merge(df_val,group_user,left_on='user_id',right_on='user_id_user_',how='left')    

    df_val=pd.merge(df_val,group_content,left_on='content_id',right_on='content_id_content_',how='left')

#     df_val=pd.merge(df_val,group_contentUser,left_on=['user_id','content_id'],right_on=['user_id_contentUser_',"content_id_contentUser_"],how='left')



    X_train = df_train.drop(columns="answered_correctly")[cols]

    y_train = df_train['answered_correctly'].values

    print(X_train.shape)

    X_val = df_val.drop(columns="answered_correctly")[cols]

    y_val = df_val['answered_correctly'].values

    

    

    clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],eval_metric='auc',verbose=1000, early_stopping_rounds=20)

    preds = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val,preds)

    print(score)

    

    fold_importance = pd.DataFrame()

    fold_importance["feature"] = cols

    fold_importance["importance"] = clf.feature_importances_

    fold_importance["fold"] = fold + 1

    feature_importance= pd.concat([feature_importance,fold_importance],axis=0)

    return clf,feature_importance
models = []

for k in range(5):

    clf,feature_importance=run(k,feature_importance)

    models.append(clf)
feature_importance.groupby(by=['feature']).agg({'importance':'sum'}).reset_index()
train_ques
import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()
group_user.dtypes
for (test_df, sample_prediction_df) in iter_test:



    

    y_preds=[]

    test_df =test_df[test_df['content_type_id'] == 0]

    test_df['prior_question_had_explanation'].fillna(0,inplace=True)

    test_df['prior_question_had_explanation']=test_df['prior_question_had_explanation'].astype(int)





    

    test_df['prior_question_elapsed_time'].fillna(elapsed_mean,inplace=True)

    test_df = pd.merge(test_df,questions,left_on='content_id',right_on='question_id',how='left')



    test_df=pd.merge(test_df,group_user,left_on='user_id',right_on='user_id_user_',how='left')   

    test_df=pd.merge(test_df,group_content,left_on='content_id',right_on='content_id_content_',how='left')

    test_df=pd.merge(test_df,group_quest,left_on='question_id',right_on="question_id_quest_",how='left')

    test_df=pd.merge(test_df,group_bundle,left_on='bundle_id',right_on="bundle_id_bundle_",how='left')

    test_df=pd.merge(test_df,group_contentUser,left_on=['user_id','content_id'],right_on=['user_id_contentUser_',"content_id_contentUser_"],how='left')



    print(test_df.dtypes)

    print(test_df.columns)

    print(test_df.shape)

   

    for model in models:

        y_pred = model.predict_proba(test_df[cols], num_iteration=model.best_iteration_)[:, 1]

        y_preds.append(y_pred)

    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
test_df
group_quest