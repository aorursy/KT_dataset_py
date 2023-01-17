import pandas as pd

import numpy as np

import time
import matplotlib.pyplot as plt

import os
os.getcwd()
os.listdir()
TrainData=pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
TrainData.shape
TrainData.keys()
TrainData.dtypes
TrainData.head(5)
TrainData.describe(include='all')
TrainData['event_count'].unique()
len(TrainData['event_count'].unique())
min(TrainData['event_count'].unique())
max(TrainData['event_count'].unique())
# frequency of each unique value under event_count (sorted)

TrainData.event_count.value_counts().sort_index()
# the event_count values which only occur once

event_count_freq=TrainData.event_count.value_counts().sort_index()

event_count_freq.loc[event_count_freq==1]
# bar-chart of frequencies of the first the first 100 unique values under event_count

# given that there are 3368 unique values, it is impossible to draw a single bar-chart with 

# all the unique values

plt.figure(figsize=(20,20))

TrainData['event_count'].value_counts()[0:100].plot(kind='bar')

plt.xlabel('event_count')

plt.ylabel('value_counts')

plt.title('distribution of values under the column event_count')

plt.show()
# all the 3368 unique values under event_count can't be put in the same bar-chart,

# therefore, it is best to use collect them into bins and look at these bins.

# We therefore draw a histogram with 300 bins for 'event_count' 

event_count_hist=TrainData.hist(column='event_count', figsize=(20,20), bins=300)
TrainData.type.unique()
# frequency of the distinct 'type' values

plt.figure()

TrainData.type.value_counts().plot(kind='bar')

plt.xlabel('type')

plt.ylabel('number of instances')

plt.title('frequency of the distinct values under the column type ')

plt.show()
TrainDataTpClip=TrainData.loc[TrainData['type']=='Clip']
TrainDataTpClip.title.unique()
# number of distinct titles for 'Clip'

len(TrainDataTpClip.title.unique())
# confirm if there are any missing titles for 'Clip' 

TrainDataTpClip.title.isnull().values.any()
TrainDataTpAct=TrainData.loc[TrainData['type']=='Activity']
TrainDataTpAct.title.unique()
# number of different Activity-titles 

len(TrainDataTpAct.title.unique())
# are there missing entries under Activity-titles?

TrainDataTpAct.title.isnull().values.any()
TrainDataTpGame=TrainData.loc[TrainData.type=='Game']
# different titles for 'type'='Game'

TrainDataTpGame.title.unique()
# number of different titles for 'type' = 'Game'

len(TrainDataTpGame.title.unique())
# entries with 'type'=Assesment

TrainDataTpAssess=TrainData.loc[TrainData.type=='Assessment']
# unique titles for 'Assessment'

TrainDataTpAssess.title.unique()
# no. of unique titles for 'Assessment'

len(TrainDataTpAssess.title.unique())
TrainData.world.unique()
TrainData.loc[TrainData.world=='NONE'].title.unique()
TrainData.loc[TrainData.title=='Welcome to Lost Lagoon!'].world.unique()
# total number of assessment sessions in train.csv

num_assess_tr=len(TrainData.loc[TrainData.type=='Assessment'].game_session.unique())

print('train.csv contains {} assessment sessions'.format(num_assess_tr))
# total number of assessments that were completed 

# i.e. assessment where a solution was submitted resulting in an event with code 4100 or 4110

completed_assessments=len(TrainData.loc[(TrainData.type=='Assessment') 

                                        & ((TrainData.event_code==4100) | (TrainData.event_code==4110) )

                                       ].game_session.unique())

print('train.csv contains {} completed assessments'.format(completed_assessments))
TrainLabels=pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
TrainLabels.keys()
TrainLabels.dtypes
TrainLabels.head()
TrainLabels.describe(include='all')
TrainLabels.title.unique()
TrainLabels['ratio_of_correct']=TrainLabels['num_correct']/(

    TrainLabels['num_correct']+TrainLabels['num_incorrect'])
(TrainLabels['ratio_of_correct']-TrainLabels['accuracy']<(10**(-16))).values.all()
# Delete ratio_of_correct as it is same as accuracy

TrainLabels.drop(['ratio_of_correct'], axis=1, inplace=True)
TrainLabels.keys()
# checking if num_correct was ever greater than 1

(TrainLabels.num_correct>1).values.any()
TrainLabels.num_correct.unique()
len(TrainData.installation_id.unique())
len(TrainLabels.installation_id.unique())
TrainLabels.loc[(TrainLabels.installation_id=='0006a69f') 

                & (TrainLabels.title=='Mushroom Sorter (Assessment)')]
Specs=pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
Specs.keys()
Specs.head()
Specs.describe()
Specs.iloc[0].args
Specs.iloc[0].info
TrainData.shape
TrainData=TrainData.merge(Specs, how='left', on='event_id')
TrainData.shape
TrainData=TrainData.merge(TrainLabels, how='left', 

                          on=['installation_id', 'title', 'game_session'])
TrainData.shape
TrainData.keys()
TrainData.event_code.unique()
len(TrainData.event_code.unique())
TrDatEvtCd2000=TrainData.loc[TrainData.event_code==2000]
TrDatEvtCd2000.shape
len(TrainData.game_session.unique())
# checking that the number of entries with event_code 2000 is same as the number of unique game sessions

len(TrainData.game_session.unique())-TrDatEvtCd2000.shape[0]
# checking that the list of unique sessions in TrDatEvtCd2000 matches exactly with that of unique sessions in TrainData

(TrDatEvtCd2000.game_session.unique()==TrainData.game_session.unique()).all()
TrDatEvtCd2000.head()
TrDtIdSess=TrainData.loc[(TrainData.installation_id=='0006a69f')

              & (TrainData.game_session=='901acc108f55a5a1')]
# Obtaining the title of the game_session above

# Recall, that we had previously established that each game_session is associated 

# with an individual clip, activity, game or assessment. Thus it should have a unique title which

# can be obtained from the title of the very first entry

TrDtIdSess.title.iloc[0]
# just to make sure that indeed there is one and only one title associated 

# with the game_session above, let's check how many unique values are there under

# the column title. If we are correct, 

# then there should be only one i.e. 'Mushroom Sorter (Assessment)'

TrDtIdSess.title.unique()
TrDtIdSess[['event_count', 'event_code', 'event_data', 'info']]
# info

TrDtIdSess.loc[TrDtIdSess.event_count==3]['info']
# event_data 

TrDtIdSess.loc[TrDtIdSess.event_count==3]['event_data']
# info

TrDtIdSess.loc[TrDtIdSess.event_count==6]['info']
# info

TrDtIdSess.loc[TrDtIdSess.event_count==6]['event_data']
# info 

TrDtIdSess.loc[TrDtIdSess.event_code==4100]['info']
# data 

TrDtIdSess.loc[TrDtIdSess.event_code==4100]['event_data']
# game_session corresponding to events with event_count>100

EvCTLargeSess=TrainData.loc[TrainData.event_count>100]

GamesLargeEvCT=EvCTLargeSess.game_session.unique()



# number of game_sessions containing event_count>100

print("There are {} sessions where the event_count exceeds 100".format(len(GamesLargeEvCT)))

session=GamesLargeEvCT[0]
TrainData.loc[TrainData.game_session==session]
# It seems the following code is a very slow implementation 

# I therefore abandoned this approach 

# a better implementation is given in the next cell

# There I use groupby to get the job done

# Therefore do NOT uncomment the code lines in this cell



#session_type=[]

#for session in GamesLargeEvCT:

#    typ=TrainData.loc[TrainData.game_session==session].type.iloc[0]

#    session_type.append([session, typ])



#LargeSessions=pd.DataFrame(session_type, columns=['game_session', 'type']) 

plt.plot()

EvCTLargeSess.groupby(['type']).game_session.unique().apply(lambda x: len(x)).plot(kind='bar')

plt.xlabel('type')

plt.ylabel('number of unique game_sessions')

plt.title('type of game_sessions that had an event_count>100')

plt.show()
session=EvCTLargeSess.loc[EvCTLargeSess.type=='Assessment'].game_session.iloc[0]

TrainData.loc[TrainData.game_session==session]
session=TrainData.loc[TrainData.event_count>3200].game_session.unique()[0]

TrainData.loc[TrainData.game_session==session]
Test=pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
Test.head()
Test.keys()
# let's plot the number of game_session of each type in Test



plt.figure()

Test.groupby(['type']).game_session.unique().apply(lambda x: len(x)).plot(kind='bar')

plt.xlabel('type')

plt.ylabel('number of game_sessions')

plt.title('number of game_session of each type in test.csv')

plt.show()
# let us look at the different kinds of events captured in the test data

# We can do this by looking at the kind of different event_codes

Test.event_code.unique()
len(Test.event_code.unique())
# eye-balling the entries having type 'Activity'

Test.loc[Test.type=="Activity"].head(25)
# eye-balling the entries having type 'Game'

Test.loc[Test.type=="Game"].head(25)
# eye-balling the entries having type 'Assessment'

Test.loc[Test.type=="Assessment"].head(25)
Test.loc[Test.game_session == '8b38fc0d2fd315dc']
# In particular we wish to see the number of events in each assessment attempted by the chosen player

# One way to do this is to group their corresponding data according to their session_ids 

# The number of counts can then be obtained by using the function size()

# The reason this works is because each event has been recorded in a seperate row, so counting 

# the number of rows for each session (i.e. asking for their size) gives the number of events



# In order to obtain various statistics for each group in a pandas.GroupBy object 

# follow this stackoverflow discussion:

# https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby





Test.loc[(Test.installation_id =='00abaee7')

         & (Test.type=='Assessment')].groupby(['game_session']).size()
player=Test.installation_id.unique()[5]

print('The chosen player has installation id: {}'.format(player))


Test.loc[(Test.installation_id ==player)

         & (Test.type=='Assessment')].groupby(['game_session']).size()
player=Test.installation_id.unique()[20]

print('The chosen player has installation id: {}'.format(player))


Test.loc[(Test.installation_id ==player)

         & (Test.type=='Assessment')].groupby(['game_session']).size()
player=Test.installation_id.unique()[35]

print('The chosen player has installation id: {}'.format(player))


Test.loc[(Test.installation_id ==player)

         & (Test.type=='Assessment')].groupby(['game_session']).size()
num_assessments=Test.loc[Test.type=='Assessment'].groupby(['installation_id']).game_session.unique().apply(lambda x: len(x))
# minimum number of assessments taken by any player

print("The minimum number of assessments taken by any player in test data is:{}".

      format(num_assessments.min()))
# maximum number of assessments taken by any player

print("The maximum number of assessments taken by any player in test data is:{}".

      format(num_assessments.max()))
plt.figure()

num_assessments.hist(bins=56)

plt.xlabel('number of assessments')

plt.ylabel('number of players')

plt.title('Chart to show how many assessments were taken by how many players')

plt.show()
num_assessments.value_counts()