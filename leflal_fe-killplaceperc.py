import gc

import numpy as np 
import pandas as pd 
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
#Checking that there are no overlap between the train and test sets matchIds
set.intersection(set(train['matchId'].unique()),set(test['matchId'].unique()))
#Remowing the sample of the train set that has winPlacePerc == np.nan
train.dropna(subset=['winPlacePerc'],inplace=True)
test['winPlacePerc'] = np.nan
df = pd.concat([train,test],axis=0)
df.drop(df.columns.difference(['Id','matchId','groupId','kills','killPlace','maxPlace','numGroups','winPlacePerc']),axis=1,inplace=True)
del train
del test
gc.collect();
#Adding the matchSize feature i.e. the # of Ids that share the same matchId
df = pd.merge(df,df.groupby('matchId')[['Id']].size().to_frame('matchSize').reset_index(),on='matchId',how='left')
#Adding a flag for the test set
df['isTest'] = df['winPlacePerc'].isnull().astype(np.int8)
#Checking that no matchSize is less or equal to 1
df[['matchSize']].describe()
#Adding a killPlacePerc feature
df['killPlacePerc'] = (df['matchSize'] - df['killPlace']) / (df['matchSize'] - 1)
#Checking if our killPlacePerc is between 0 and 1
df[['killPlacePerc']].describe()
#Investigating the negative killPlacePerc
df[df['killPlacePerc'] < 0].sort_values(by='matchSize').head()
#In the first row above, we can see that killPlace == 8 although matchSize == 7, let's zoom on this matchId
df[df['matchId'] == '8799301e853202'].sort_values(by='killPlace')
#We can see that there is no killPlace == 3 in the matchId above 
#We can also see that all players with 0 kills are ranked although they are tied, 
#And that this tiebreak ranking seems to be linked to winPlacePerc
#Let us check that this is indeed the case, namely that:
#For every matchId, Ids that have the same number of kills have their killPlace ranked by decreasing winPlacePerc
#Let us first check below that there aren't any ties for killPlace
df_noties = df.groupby(['matchId','killPlace'])[['Id']].nunique()
df_noties.max()
#Let's now check that all tiebreaks in the train dataset are according to winPlacePerc
df_tiebreak = df[df['isTest'] !=1].sort_values(by=['matchId','kills','killPlace'],ascending=[False,False,True])
df_tiebreak = df_tiebreak.groupby(['matchId','kills','killPlace'])[['winPlacePerc']].mean().groupby(['matchId','kills'])[['winPlacePerc']].diff()
df_tiebreak.sort_index(ascending=[False,False,True],inplace=True)
df_tiebreak.head(15)
df_tiebreak.dropna(inplace=True)
df_tiebreak.max()
#Now that we have proven that the tiebreaking is by non-increasing winPlacePerc,
#Let us investigate another matchId
df[df['matchId'] == '58eb66dd8a0764'].sort_values(by='killPlace')
# On this latter example we see that killPlacePerc tie-breaks for Ids in the same groupId and with the same number of kills 
# This could be misleading for winPlacePerc prediction: we prefer to keep the tie.
# So let us define a more adequate killPlacePerc:
df_killPlace = df.groupby(['matchId','Id'])[['kills','killPlace']].mean()
df_killPlace.sort_values(by=['matchId','kills','killPlace'],ascending=[False,False,True],inplace=True)
df_killPlace = df_killPlace.groupby('matchId')['kills'].rank(method='first',ascending=False).to_frame('killPlacePerc').reset_index()
df.drop('killPlacePerc',axis=1,inplace=True)
df = df.merge(df_killPlace,on=['matchId','Id'],how='left')
df['killPlacePerc'] = df.groupby(['matchId','kills','groupId'])[['killPlacePerc']].transform(np.mean)
df['killPlacePerc'] = (df['matchSize'] - df['killPlacePerc'])/ (df['matchSize'] - 1)
#Let's check the two matchIds we zoomed on hereabove
df.loc[df['matchId'].isin(['8799301e853202','58eb66dd8a0764']),['matchId','groupId','Id','killPlace','kills','winPlacePerc','killPlacePerc']].sort_values(by=['matchId','killPlace'],ascending=[False,True])