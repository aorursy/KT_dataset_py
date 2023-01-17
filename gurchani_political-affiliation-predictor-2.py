import gc

gc.collect()



import pandas as pd

import numpy as np

 # Manually Annotated dataset from the above mentioned paper

df0 = pd.read_csv('../input/dataset-for-french/profilesmanualannotation.csv') 
import collections

df1 = df0[['UserId', 'party']] #Trimming down the first dataset

fr = pd.read_csv('../input/annotatedfriends/manualannotationFriends.csv', names=['id', 'friend']) #Dataset of Friends

fr.drop_duplicates(inplace = True)

Features = pd.read_csv('../input/features/possibleFeatures.csv' , names=['friend'])

#mlp_features = pd.read_csv('../input/mlp-importantfeatures/temp20 (1).csv')

#mlp_features = mlp_features['friend']

#Features
#Seperating all the parties

fnFriends = pd.merge(df0[df0['party'] == 'fn'], fr, how = 'inner', left_on='UserId' , right_on = 'id')

fiFriends = pd.merge(df0[df0['party'] == 'fi'], fr, how = 'inner', left_on='UserId' , right_on = 'id')

lrFriends = pd.merge(df0[df0['party'] == 'lr'], fr, how = 'inner', left_on='UserId' , right_on = 'id')

emFriends = pd.merge(df0[df0['party'] == 'em'], fr, how = 'inner', left_on='UserId' , right_on = 'id')

psFriends = pd.merge(df0[df0['party'] == 'ps'], fr, how = 'inner', left_on='UserId' , right_on = 'id')
fnFriendCount = fnFriends.groupby(['friend']).count()

fiFriendCount = fiFriends.groupby(['friend']).count()

lrFriendCount = lrFriends.groupby(['friend']).count()

emFriendCount = emFriends.groupby(['friend']).count()

psFriendCount = psFriends.groupby(['friend']).count()
#listDf = []

#for i in range(6,7):

#    listDf = []

#    listDf = (fnFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

#    fiFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

#    lrFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

#    emFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

#    psFriendCount.nlargest(i, 'UserId').index.values.tolist())

#    joe = pd.DataFrame({'friend' :listDf})

#    jpott = pd.merge(joe, fr, how = 'inner' , left_on = 'friend', right_on = 'friend' )

#    DicList = []

#    for group, frame in jpott.groupby('id'):

        

#        ak = frame['friend'].tolist()

#        dictOf = dict.fromkeys(ak , 1)

#        DicList.append(dictOf)

#    print(DicList[0])

#    from sklearn.feature_extraction import DictVectorizer

#    dictvectorizer = DictVectorizer(sparse = True)

#    features = dictvectorizer.fit_transform(DicList)

    #print(features)

    #print(features.todense().shape)

#    dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 

#                               index = jpott['id'].unique())

#    dataFrame.index.names = ['UserId']

#    print(dataFrame.head(1))

#    print(jpott['id'].unique())

    #print(jpott['id'] == 1278286086)

    
print(fr[fr['id']==1278286086])
#print(listDf)
joinedDF1 = pd.read_csv('../input/top6features/profile_manual_top6features (1).csv')

joinedDF1.shape
print(joinedDF1.friend.unique())
joinedDF1.friend.unique()
accuracyScore = []

coreFeatures = []

listDf = []

for i in range(1, 50):

    listDf = []

    listDf = (fnFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

    fiFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

    lrFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

    emFriendCount.nlargest(i, 'UserId').index.values.tolist() + 

    psFriendCount.nlargest(i, 'UserId').index.values.tolist())

    joinDF = pd.DataFrame({'friend' :listDf})

    joinedDF = pd.merge(joinDF, fr, how = 'inner' , left_on = 'friend', right_on = 'friend' )

    

    #print(joinedDF.head(1))

    #merged = joinedDF1.merge(joinedDF, indicator=True, how='outer')

    #print(merged[merged['_merge'] == 'right_only'])

    #print(joinedDF.shape)

    DicList = []

    indexTobe = []

    for group, frame in joinedDF.groupby('id'):

        ak = frame['friend'].tolist()

        indexTobe.append(group)

        #break

        dictOf = dict.fromkeys(ak , 1)

        DicList.append(dictOf)

    #print(DicList[0])

    from sklearn.feature_extraction import DictVectorizer

    dictvectorizer = DictVectorizer(sparse = True)

    features = dictvectorizer.fit_transform(DicList)

    features.todense().shape

    dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 

                               index = indexTobe)

    #print(joinedDF['id'].unique())

    dataFrame.index.names = ['UserId']

    #print(dataFrame.head())

    mergedWithParties = pd.merge(dataFrame , df0, left_on = 'UserId', right_on = 'UserId', how= 'inner')

    mergedWithParties.drop(columns=['mediaConnection', 'gender', 'profileType'], inplace = True)

    mergedWithParties.fillna(0, inplace = True)

    #print('Before')

    #print(mergedWithParties.sample(random_state = 2))

    parties = {'fi': 1,'ps': 2,'em': 3,'lr': 4,'fn': 5,'fi/ps': 6,'fi/em': 7, 'fi/lr': 8,'fi/fn': 9, 'ps/em': 10,

    'ps/lr': 11, 'ps/fn': 12, 'em/lr': 13,'em/fn': 14, 'lr/fn': 15}

    #print(df1['party'])



    mergedWithParties['party'] = mergedWithParties['party'].map(parties)

    #print('After')

    #print(mergedWithParties.sample(random_state = 2))

    sanityCheck = pd.concat([mergedWithParties['UserId'],  mergedWithParties['party']], axis = 1)

    sanityCheck2 =  pd.concat([df0['UserId'] ,df0['party']], axis = 1)

    pd.set_option('display.max_columns', None)

    #print(pd.concat([sanityCheck, sanityCheck2], axis = 1))

    #sanity = sanityCheck.merge(sanityCheck2, indicator=True, how='outer')

    #print(merged[merged['_merge'] == 'right_only'])

    #print(merged.shape)

    

    mergedWithParties2 = mergedWithParties[(mergedWithParties['party']==1.0) | (mergedWithParties['party']==2.0)|

                                      (mergedWithParties['party']==3.0)| (mergedWithParties['party']==4.0)

                                      | (mergedWithParties['party']==5.0)]

    

    #print(mergedWithParties2[mergedWithParties2['party'] == 1.0])

    #print(mergedWithParties2[mergedWithParties2['party'] == 2.0])

    #print(mergedWithParties2[mergedWithParties2['party'] == 3.0].shape)

    #print(mergedWithParties2[mergedWithParties2['party'] == 4.0].shape)

    #print(mergedWithParties2[mergedWithParties2['party'] == 5.0].shape)

    

    from sklearn.ensemble import  RandomForestClassifier

    from sklearn.feature_selection import SelectFromModel

    from sklearn.model_selection import train_test_split



    train, test = train_test_split(mergedWithParties2, test_size=0.2, shuffle=True)

    featureSelector = RandomForestClassifier(n_estimators=50)

    featureSelector.fit(train.iloc[:, :-1], train['party'])

    accScore = featureSelector.score(test.iloc[:, :-1],pd.Series(test['party']))

    accuracyScore.append(accScore)

    coreFeatures.append(dictvectorizer.get_feature_names())
print(accuracyScore)

print(coreFeatures)
print(coreFeatures[15])
print(listDf)
joinedDF = pd.read_csv('../input/top6features/profile_manual_top6features (1).csv')

#joinedDF = pd.merge(Features, fr, left_on = 'friend', right_on = 'friend', how= 'inner')

#joinedDF.shape
#mergedWithParties = pd.merge(joinedDF , df0, left_on = 'id', right_on = 'UserId', how= 'inner')

#mergedWithParties[mergedWithParties['party'] == 'fi'].nunique()

DicList = []

for group, frame in joinedDF.groupby('id'):

    ak = frame['friend'].tolist()

    dictOf = dict.fromkeys(ak , 1)

    DicList.append(dictOf)



from sklearn.feature_extraction import DictVectorizer

dictvectorizer = DictVectorizer(sparse = True)

features = dictvectorizer.fit_transform(DicList)

features.todense().shape
dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 

                               index = joinedDF['id'].unique())

dataFrame.index.names = ['UserId']

dataFrame.head()

mergedWithParties = pd.merge(dataFrame , df0, left_on = 'UserId', right_on = 'UserId', how= 'inner')
mergedWithParties.head()
mergedWithParties.drop(columns=['mediaConnection', 'gender', 'profileType'], inplace = True)
mergedWithParties.fillna(0, inplace = True)
mergedWithParties.head()
parties = {'fi': 1,'ps': 2,'em': 3,'lr': 4,'fn': 5,'fi/ps': 6,'fi/em': 7, 'fi/lr': 8,'fi/fn': 9, 'ps/em': 10,

'ps/lr': 11, 'ps/fn': 12, 'em/lr': 13,'em/fn': 14, 'lr/fn': 15}

#print(df1['party'])



mergedWithParties['party'] = mergedWithParties['party'].map(parties)

mergedWithParties.head()
mergedWithParties2 = mergedWithParties[(mergedWithParties['party']==1.0) | (mergedWithParties['party']==2.0)|

                                      (mergedWithParties['party']==3.0)| (mergedWithParties['party']==4.0)

                                      | (mergedWithParties['party']==5.0)]

mergedWithParties2.head()
from sklearn.ensemble import  RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split



train, test = train_test_split(mergedWithParties2, test_size=0.2, shuffle=True)

featureSelector = RandomForestClassifier(n_estimators=50)

featureSelector.fit(train.iloc[:, :-1], train['party'])
featureSelector.score(test.iloc[:, :-1],pd.Series(test['party']))
from sklearn.neighbors import KNeighborsClassifier

clf2 = KNeighborsClassifier(n_neighbors = 800)

clf2.fit(train.iloc[:, :-1], train['party'])

clf2.score(test.iloc[:, :-1],pd.Series(test['party']))

print(test.shape)
Ypred = pd.Series(clf2.predict(test.iloc[:, :-1]))

allLabels = mergedWithParties2['party'].unique()

print(allLabels)
pred = list(pd.Series(clf2.predict(test.iloc[:, :-1])))

print(len(pred))

testY = list(pd.Series(test['party']))

labels = list(allLabels)

from collections import Counter

print(Counter(pred))

print(Counter(testY))
from sklearn.metrics import confusion_matrix

confusion_matrix(pred, testY)

import matplotlib as mpl

import matplotlib.pyplot as plt

import networkx as nx