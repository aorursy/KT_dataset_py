# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the data

pd.set_option('display.max_colwidth', -1)

path = '/kaggle/input/data-science-bowl-2019/'

#test = pd.read_csv(path+'test.csv',parse_dates=["timestamp"],dtype = {'event_count':np.int16,'event_code':np.int16})

train = pd.read_csv(path+'train.csv',parse_dates=["timestamp"],dtype = {'event_count':np.int16,'event_code':np.int16})

train_labels = pd.read_csv(path+'train_labels.csv')
# Drop records in train where no assessement is taken

train = train[train.installation_id.isin(train_labels.installation_id.unique())]

del train_labels

gc.collect()
def AssessmentScore(df):

    s1 = df[(df.title=='Bird Measurer (Assessment)') & (df.event_code==4110)].event_data.apply(lambda x: x.find('"correct":true')>-1)

    s2 = df[(df.type=='Assessment') & (df.title!='Bird Measurer (Assessment)') & (df.event_code==4100)].event_data.apply(lambda x: x.find('"correct":true')>-1)

    s1.name = 'correct'

    s2.name = 'correct'

    s1 = s1.append(s2).sort_index()

    return df.join(s1)



def TimeDifference(df):

    gb = df.loc[:,['installation_id','timestamp']].groupby('installation_id',as_index=False)

    store = gb.diff()

    store.index = df.index

    store = store.rename(columns = {'timestamp':'dt'})

    store.dt = store.dt-pd.to_datetime(0,utc=True)

    store.dt = store.dt.dt.total_seconds()

    # store.dt = store.dt.astype(np.float32)

    return df.join(store)



def GameSessionStats(df):

    gb = df.loc[:,['game_session','dt']].groupby('game_session')

    store = gb.sum()

    store.columns = ['game_session_time']

    return df.reset_index().merge(store, how='left', on='game_session').set_index('index')



def InstallationIdStats(df):

    subset = df.loc[:,['installation_id','game_session','title']]

    titles = subset.title.unique()

    subset = subset.drop_duplicates()

    for title in titles:

        subset[title] = (subset.title==title).apply(lambda x: np.uint8(x))

    subset = subset.drop(columns = 'title')

    

    gb = subset.groupby(['installation_id'])

    store = gb.cumsum().join(subset.installation_id)

    store = store.groupby('installation_id').shift(periods=1,fill_value=0)

    store = store.drop(columns='installation_id').join(subset.loc[:,['installation_id','game_session']])

    return df.reset_index().merge(store, how='left', on=['installation_id','game_session']).set_index('index')



def LabelData(df):

    store = df[(train.correct==True) | (df.correct==False)]

    out = pd.DataFrame(store.game_session.unique(),columns = ['game_session'])

    

    # calculate num_correct and num_incorrect

    gb = store.loc[:,['game_session','installation_id','title','correct','event_id']].groupby(['game_session','installation_id','title','correct'])

    store = gb.count().unstack().fillna(value=0)

    

    # cleanup column names

    store.columns = store.columns.to_flat_index()

    new_cols=[]

    for t in store.columns:

        if True in t:

            new_cols.append('num_correct')

        else:

            new_cols.append('num_incorrect')

    store.columns = new_cols

    store = store.reset_index()

    

    # merge into out

    out = out.merge(store, how='left',on='game_session')

    out['accuracy'] = out.num_correct/(out.num_correct+out.num_incorrect)

    

    def CalcAccuracyGroup(x):

        if x == 1:

            return 3

        elif x==0.5:

            return 2

        elif x==0:

            return 0

        else:

            return 1

    

    out['accuracy_group'] = out.accuracy.apply(CalcAccuracyGroup)

    return out



def checkanswer(df1,df2):

    store = df1.merge(df2.loc[:,['game_session','accuracy_group']],how='left',on='game_session')

    return (store.accuracy_group_x==store.accuracy_group_y).unique()
train = AssessmentScore(train)

train = TimeDifference(train)

train = GameSessionStats(train)

train = InstallationIdStats(train)

#labels = LabelData(train)

#checkanswer(train_labels,labels)
train.describe()
def ExtendTrainData(df):

    # find the game_sessions where Assessments are taken

    subset = df[(df.correct==True) | (df.correct==False)]

    game_sessions = subset.game_session.unique()

    out = pd.DataFrame(game_sessions,columns = ['game_session'])

    

    # grab title cumcounts

    titles = df.title.unique()

    gb = subset.loc[:,['game_session']+list(titles)].groupby('game_session')

    store = gb.first()

    for title in titles:

        store[title] = store[title].apply(np.uint8)

    out = out.merge(store,how='left',on='game_session')

    

    # grab timestamp and dt

    store = subset.loc[:,['game_session','timestamp','dt']].groupby('game_session').first()

    out = out.merge(store,how='left',on='game_session')

    

    # calculate num_correct and num_incorrect

    subset = df[(df.correct==True) | (df.correct==False)]

    gb = subset.loc[:,['game_session','installation_id','title','correct','event_id']].groupby(['game_session','installation_id','title','correct'])

    store = gb.count().unstack().fillna(value=0)

    # cleanup column names

    store.columns = store.columns.to_flat_index()

    new_cols=[]

    for t in store.columns:

        if True in t:

            new_cols.append('num_correct')

        else:

            new_cols.append('num_incorrect')

    store.columns = new_cols

    store = store.reset_index()

    # merge into out

    out = out.merge(store, how='left',on='game_session')

    

    # calc accuracy and accuracy group

    out['accuracy'] = out.num_correct/(out.num_correct+out.num_incorrect)

    def CalcAccuracyGroup(x):

        if x == 1:

            return 3

        elif x==0.5:

            return 2

        elif x==0:

            return 0

        else:

            return 1

    out['accuracy_group'] = out.accuracy.apply(CalcAccuracyGroup)

    

    # accuracy group statistics

    store = out.loc[:,['installation_id','accuracy']].groupby('installation_id').shift(1)

    store = pd.DataFrame({'installation_id': out.installation_id,'accuracy': store.accuracy})

    gb = store.loc[:,['installation_id','accuracy']].groupby('installation_id')

    out['acc_sum'] = gb.cumsum()

    out['acc_min'] = gb.cummin()

    out['acc_max'] = gb.cummax()

    out['acc_cnt'] = gb.cumcount()

    out['acc_avg'] = out['acc_sum']/out['acc_cnt']

    

    store = out.loc[:,['installation_id','accuracy_group']].groupby('installation_id').shift(1)

    store = pd.DataFrame({'installation_id': out.installation_id,'accuracy_group': store.accuracy_group})

    gb = store.loc[:,['installation_id','accuracy_group']].groupby('installation_id')

    out['acc_gr_sum'] = gb.cumsum()

    out['acc_gr_min'] = gb.cummin()

    out['acc_gr_max'] = gb.cummax()

    out['acc_gr_cnt'] = gb.cumcount()

    out['acc_gr_avg'] = out['acc_gr_sum']/out['acc_gr_cnt']



    out['acc_min'] = out.acc_min.fillna(value=-1)

    out['acc_max'] = out.acc_max.fillna(value=-1)

    out['acc_avg'] = out.acc_avg.fillna(value=-1)

    out['acc_gr_min'] = out.acc_gr_min.fillna(value=-1)

    out['acc_gr_max'] = out.acc_gr_max.fillna(value=-1)

    out['acc_gr_avg'] = out.acc_gr_avg.fillna(value=-1)

    

    out = out.drop(columns=['num_correct','num_incorrect','acc_sum','acc_cnt','acc_gr_sum','acc_gr_cnt'])

    

    return out
train_extend = ExtendTrainData(train)
train_extend.info()
train_extend.head()
train_extend.columns
train_extend.describe()
train_extend.to_csv('train_extend.csv', index=False)