# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import cohen_kappa_score

from scipy.stats import mode

from sklearn.model_selection import train_test_split

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.options.display.max_seq_items = 2000

# Any results you write to the current directory are saved as output.
keep_cols = ['game_session', 'installation_id','timestamp', 'event_data', 'event_code', 'title', 'game_time', 'type']



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', usecols=keep_cols)

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', usecols=keep_cols)

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

#specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

#train = train.head(1000000)
# Removing installation ids not found in train_labels as these will not be useful in training

start_mem = train.memory_usage().sum() / 1024**2

#print("Memory usage at start is {:.2f} MB".format(start_mem))

not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))

train_new=~train['installation_id'].isin(not_req)

train.where(train_new,inplace=True)

train.dropna(inplace=True)

train['event_code']=train.event_code.astype(int)

end_mem = train.memory_usage().sum() / 1024**2

#print("Memory usage at end is {:.2f} MB".format(end_mem))
activityTypes= train[['title','type']].copy().drop_duplicates()

list_of_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])

list_of_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

activity_correct_event_code = dict(zip(list_of_activities, (4100*np.ones(len(list_of_activities))).astype('int')))

activity_correct_event_code['Bird Measurer (Assessment)'] = 4110
def getFeatures(userData, isTest):

    counter = 0

    #test_set=False

    session_List=[]



    for i, (ins_id, user_sample) in enumerate(userData.groupby('installation_id', sort=False)):

        session_type_counter = dict({'Clip':0, 'Activity': 0, 'Game': 0, 'Assessment': 0})

        time_spent = 0

        clipTime=0

        activityTime=0

        gameTime=0

        assessmentTime=0



        activityLog = dict()

        for i, row in activityTypes.iterrows():

            activityLog[row['title']]=0

            activityLog[row["title"]+"Time"]=0

            if ("Assessment" in row['title']):

                activityLog[row["title"]+"Solved"]=0





        for i, session in user_sample.groupby('game_session', sort=False):

            session_type = session['type'].iloc[0]

            session_title = session['title'].iloc[0]

            session_id = session['game_session'].iloc[0]

            processAssessment=False

            if (isTest) & (len(session)==1) & (session_type == 'Assessment'):

                processAssessment = True

                #print('checkpoint1 ', session_type, '  ', len(session), ' ', session_id)

            elif (~isTest) & (len(session) > 1):

                processAssessment = True

            else:

                processAssessment= False

            



            if (session_type == 'Assessment'):

                all_attempts = session.query(f'event_code == {activity_correct_event_code[session_title]}')

                true_attempts = all_attempts['event_data'].str.contains('true').sum()

                false_attempts = all_attempts['event_data'].str.contains('false').sum()



            if (session_type == 'Assessment') & (processAssessment):

                Dict1 =dict({'installation_id': session['installation_id'].iloc[0]})

                Dict1['game_session']=session['game_session'].iloc[0]



                # Dict1['type']=session['type'].iloc[0]

                Dict1['title']=session['title'].iloc[0]

                Dict1['priorClips']=session_type_counter['Clip']

                Dict1['priorActivity']=session_type_counter['Activity']

                Dict1['priorGames']=session_type_counter['Game']

                Dict1['priorAssessments']=session_type_counter['Assessment']

                #Dict1['session_time']= session['game_time'].max()

                #Dict1['total_prior_time']=time_spent

    #            Dict1['avgclipTime']=clipTime/session_type_counter['Clip']

    #            Dict1['avgactivityTime']=activityTime/session_type_counter['Activity']

    #            Dict1['avggameTime']= gameTime/session_type_counter['Game']

    #            Dict1['avgassessmentTime']=assessmentTime/session_type_counter['Assessment']

                Dict1.update(activityLog)

                if (isTest):

                    session_List.append(Dict1)

                else:

                    if true_attempts+false_attempts > 0:

                        session_List.append(Dict1)

                              

            if len(session)>1:

                    session_type_counter[session_type]+=1

                    time_spent+=session['game_time'].max()

                    activityLog[session_title]+=1

                    activityLog[session_title+"Time"]+=session['game_time'].max()



                    if(session_type=="Clip"):

                        clipTime+=session['game_time'].max()

                    elif(session_type=="Game"):

                        gameTime+=session['game_time'].max()

                    elif(session_type=="Activity"):

                        activityTime+=session['game_time'].max()

                    elif(session_type=="Assessment"):

                        assessmentTime+=session['game_time'].max()

                        if true_attempts > 0:

                            activityLog[session_title+"Solved"]+=1

    processedDf=pd.DataFrame(session_List)

    return processedDf

# featuresDf.head(20)

# print(len(session_List))

# featuresDf=pd.DataFrame(session_List)

# featuresDf.head(20)
featuresDf_xgb = getFeatures(train, False)
del train
testDf_xgb = getFeatures(test, True)
del test
# merge with accuracy group data

featuresDf_xgb=pd.merge(train_labels, featuresDf_xgb, on=['game_session', 'installation_id', 'title'], how='left')
#columns

id_columns=['game_session', 'installation_id']

drop_columns=['num_correct', 'num_incorrect', 'accuracy']

cat_columns=['title']

#numeric_columns= ['priorClips',

       #'priorActivity', 'priorGames', 'priorAssessments', 'total_prior_time', 'clipTime', 'activityTime', 'gameTime', 'assessmentTime']
featuresDf_xgb.drop(id_columns, axis=1, inplace=True)

featuresDf_xgb.drop(drop_columns, axis=1, inplace=True)
testDf_xgb.drop(id_columns, axis=1, inplace=True)
all_features = [x for x in featuresDf_xgb.columns if x not in ['accuracy_group']]

X, y = featuresDf_xgb[all_features], featuresDf_xgb['accuracy_group']

#del train
def make_classifier():

    clf = CatBoostClassifier(

                               loss_function='MultiClass',

    #                            eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=2000,

                               od_type="Iter",

#                                depth=8,

                               early_stopping_rounds=500,

    #                            l2_leaf_reg=1,

    #                            border_count=96,

                               random_seed=2020

                              )

        

    return clf
clf = make_classifier()

clf.fit(X, y, verbose=500, cat_features=cat_columns)
preds = clf.predict(testDf_xgb)
submission['accuracy_group'] = np.round(preds).astype('int')

submission.to_csv('submission.csv', index=None)

# lb = LabelEncoder()

# lb.fit(featuresDf_xgb['title'])
# featuresDf_xgb['title']=lb.transform(featuresDf_xgb['title'])

# testDf_xgb['title']=lb.transform(testDf_xgb['title'])
# X_train=featuresDf_xgb.drop('accuracy_group',axis=1)

# y_train=featuresDf_xgb['accuracy_group']
# pars = {

#     'colsample_bytree': 0.8,                 

#     'learning_rate': 0.08,

#     'max_depth': 10,

#     'subsample': 1,

#     'objective':'multi:softprob',

#     'num_class':4,

#     'eval_metric':'mlogloss',

#     'min_child_weight':3,

#     'gamma':0.25,

#     'n_estimators':500

# }



# y_pre=np.zeros(len(testDf_xgb),dtype=float)

#testDf_xgb=xgb.DMatrix(testDf_xgb)



#kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# y_pre=np.zeros((l,dtype=float)

# testDf_xgb=xgb.DMatrix(testDf_xgb)



# for train_index, val_index in kf.split(X_train, y_train):

#     print('learning')

#     train_X = X_train.iloc[train_index]

#     val_X = X_train.iloc[val_index]

#     train_y = y_train[train_index]

#     val_y = y_train[val_index]

#     xgb_train = xgb.DMatrix(train_X, train_y)

#     xgb_eval = xgb.DMatrix(val_X, val_y)



#     xgb_model = xgb.train(pars,

#                   xgb_train,

#                   num_boost_round=1000,

#                   evals=[(xgb_train, 'train'), (xgb_eval, 'val')],

#                   verbose_eval=False,

#                   early_stopping_rounds=20

#                  )



#     pred=xgb_model.predict(testDf_xgb)

#     y_pre+=pred



#     #val_X=xgb.DMatrix(val_X)

#     #pred_val=[np.argmax(x) for x in xgb_model.predict(val_X)]

# 3

# # fit model no training data

# model = XGBClassifier()

# model.fit(X_train.values, y_train.values)



#y_pre=model.predict(testDf_xgb.values)



# pred = np.asarray(y_pre)
#y_pre=xgb_model.predict(testDf_xgb)
#xgb_model, pred=model(X_train,y_train,testDf_xgb, 5)
#fig, ax = plt.subplots(figsize=(10,10))

#xgb.plot_importance(xgb_model, max_num_features=50, height=0.5, ax=ax,importance_type='gain')

#plt.show()
# submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

# submission['accuracy_group'] = np.round(pred).astype('int')

# submission.to_csv('submission.csv', index=None)