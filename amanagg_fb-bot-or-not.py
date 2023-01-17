# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
#loading data

data = pd.read_csv('../input/bids.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.head(5)

train.head(5)
data.info()
train.info()
"""NUMBER OF ACTIONS"""
data.head(5)
dataIdList =  data['bidder_id'].unique()
groupedData = data[['bidder_id','auction']].groupby('bidder_id').count()['auction']
def numberofActions(line,dataGrouped,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]
    
train['nbActions'] = train.apply(lambda x: numberofActions(x,groupedData,dataIdList),axis=1) 
test['nbActions'] = test.apply(lambda x: numberofActions(x,groupedData,dataIdList),axis=1)
train.head(5)

"""RESPONSE TIME"""

lastActionDict={}

def timeResponse(line,lastActionDict):
    if line['auction'] in lastActionDict:
        time = line['time'] - lastActionDict[line['auction']]
        lastActionDict[line['auction']] = line['time']
        return time
    else :
        lastActionDict[line['auction']] = line['time']
        return 0

data['timeresponse'] = data.apply(lambda x: timeResponse(x,lastActionDict),axis=1)
data.head(5)


groupedData = data[['bidder_id','timeresponse']].groupby('bidder_id').mean()['timeresponse']
def meanTimeResponse(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['meanTimeResponse'] = train.apply(lambda x: meanTimeResponse(x,groupedData,dataIdList),axis=1)
test['meanTimeResponse'] = test.apply(lambda x: meanTimeResponse(x,groupedData,dataIdList),axis=1)
train.head(5)
groupedData = data[['bidder_id','timeresponse']].groupby('bidder_id').min()['timeresponse']
def minTimeResponse(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['minTimeResponse'] = train.apply(lambda x: minTimeResponse(x,groupedData,dataIdList),axis=1)
test['minTimeResponse'] = test.apply(lambda x: minTimeResponse(x,groupedData,dataIdList),axis=1)
train.head(5)

groupedData = data[['bidder_id','timeresponse']].groupby('bidder_id').max()['timeresponse']
def maxTimeResponse(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['maxTimeResponse'] = train.apply(lambda x: maxTimeResponse(x,groupedData,dataIdList),axis=1)
test['maxTimeResponse'] = test.apply(lambda x: maxTimeResponse(x,groupedData,dataIdList),axis=1)
train.head(5)

lastActionDict={}

def bidValue(line,lastActionDict):
    if line['auction'] in lastActionDict:
        lastActionDict[line['auction']] += 1
    else :
        lastActionDict[line['auction']] = 1
    return lastActionDict[line['auction']]

data['bidValue'] = data.apply(lambda x: bidValue(x,lastActionDict),axis=1)
data.head(5)
groupedData = data[['bidder_id','bidValue']].groupby('bidder_id').mean()['bidValue']
def meanbidValue(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['meanBidValue'] = train.apply(lambda x: meanbidValue(x,groupedData,dataIdList),axis=1)
test['meanBidValue'] = test.apply(lambda x: meanbidValue(x,groupedData,dataIdList),axis=1)
train.head(5)
groupedData = data[['bidder_id','bidValue']].groupby('bidder_id').min()['bidValue']
def minbidValue(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['minBidValue'] = train.apply(lambda x: minbidValue(x,groupedData,dataIdList),axis=1)
test['minBidValue'] = test.apply(lambda x: minbidValue(x,groupedData,dataIdList),axis=1)
train.head(5)

groupedData = data[['bidder_id','bidValue']].groupby('bidder_id').max()['bidValue']
def maxbidValue(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['maxBidValue'] = train.apply(lambda x: maxbidValue(x,groupedData,dataIdList),axis=1)
test['maxBidValue'] = test.apply(lambda x: maxbidValue(x,groupedData,dataIdList),axis=1)
train.head(5)

dataUnique = data[['bidder_id','auction']].drop_duplicates()
groupedData = dataUnique.groupby('bidder_id').count()['auction']

def numberofAuctions(line,dataGrouped,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]
    
train['nbAuctionsPlayed'] = train.apply(lambda x: numberofAuctions(x,groupedData,dataIdList),axis=1)
test['nbAuctionsPlayed'] = test.apply(lambda x: numberofAuctions(x,groupedData,dataIdList),axis=1)
train.head(5)
def last(line,lastActionDict):
    if line['bidValue'] == lastActionDict[line['auction']] :
        return True
    else :
        return False

data['end'] = data.apply(lambda x: last(x,lastActionDict),axis=1)
data.head(5)
winner = data[data['end']].groupby('bidder_id').count()['auction']

def nbOfAuctionWon(line,winner):
    if not line['bidder_id'] in winner:
        return 0
    else:
        return winner[line['bidder_id']]
    
train['nbOfAuctionWon'] = train.apply(lambda x: nbOfAuctionWon(x,winner),axis=1)
test['nbOfAuctionWon'] = test.apply(lambda x: nbOfAuctionWon(x,winner),axis=1)
train.head()
nboftriple = data[['bidder_id','ip','country','url']].drop_duplicates().groupby('bidder_id').count()['ip']

def nbOfTriple(line,nboftriple):
    if not line['bidder_id'] in nboftriple:
        return 0
    else:
        return nboftriple[line['bidder_id']]
    
train['nbOfTriple'] = train.apply(lambda x: nbOfTriple(x,nboftriple),axis=1)
test['nbOfTriple'] = test.apply(lambda x: nbOfTriple(x,nboftriple),axis=1)
train.head()
data['merchandise'].unique()

dicti = {'jewelry':0, 'furniture':1, 'home goods':2, 'mobile':3, 'sporting goods':4,
       'office equipment':5, 'computers':6, 'books and music':7, 'clothing':8,
       'auto parts':9}

train[['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
       'office equipment', 'computers', 'books and music', 'clothing',
       'auto parts']] = pd.DataFrame(np.zeros((train.shape[0],10)),columns=['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
       'office equipment', 'computers', 'books and music', 'clothing',
       'auto parts'])

test[['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
       'office equipment', 'computers', 'books and music', 'clothing',
       'auto parts']] = pd.DataFrame(np.zeros((train.shape[0],10)),columns=['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
       'office equipment', 'computers', 'books and music', 'clothing',
       'auto parts'])
grouped = data[['bidder_id','merchandise']].drop_duplicates()
def findMerchandise(line,grouped,dicti,dataid):
    res = np.zeros(10)
    if line in dataid:
        merch = np.array(grouped[grouped['bidder_id']==line]['merchandise'])[0]
        res[dicti[merch]] = 1
    return tuple(res)

res = train['bidder_id'].map(lambda x: findMerchandise(x,grouped,dicti,dataIdList))
(train['jewelry'], train['furniture'], train['home goods'], 
 train['mobile'], train['sporting goods'],train['office equipment'], 
 train['computers'], train['books and music'], train['clothing'],train['auto parts']) = zip(*res)

res = test['bidder_id'].map(lambda x: findMerchandise(x,grouped,dicti,dataIdList))
(test['jewelry'], test['furniture'], test['home goods'], 
 test['mobile'], test['sporting goods'],test['office equipment'], 
 test['computers'], test['books and music'], test['clothing'],test['auto parts']) = zip(*res)

nbcountry=data[['bidder_id','country']].drop_duplicates().groupby('bidder_id').count()['country']
def country(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]

train['nbCountry'] = train.apply(lambda x: country(x,nbcountry,dataIdList),axis=1)
test['nbCountry'] = test.apply(lambda x: country(x,nbcountry,dataIdList),axis=1)
train.head()
nbIp=data[['bidder_id','ip']].drop_duplicates().groupby('bidder_id').count()['ip']
def ip(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['nbIp'] = train.apply(lambda x: ip(x,nbIp,dataIdList),axis=1)
test['nbIp'] = test.apply(lambda x: ip(x,nbIp,dataIdList),axis=1)
train.head()
nbUrl=data[['bidder_id','url']].drop_duplicates().groupby('bidder_id').count()['url']
def url(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['nbUrl'] = train.apply(lambda x: url(x,nbUrl,dataIdList),axis=1)
test['nbUrl'] = test.apply(lambda x: url(x,nbUrl,dataIdList),axis=1)
train.head()
l = data[['bidder_id','url','bid_id']].groupby(['bidder_id','url']).count()['bid_id']
def actionMainUrl(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    elif 'vasstdc27m7nks3' in groupedData[line['bidder_id']]:
        return float(groupedData[line['bidder_id'],'vasstdc27m7nks3'])/line['nbActions']
    else:
        return 0

train['actionFromMain'] = train[['bidder_id','nbActions']].apply(lambda x :actionMainUrl(x,l,dataIdList),axis=1)
test['actionFromMain'] = test[['bidder_id','nbActions']].apply(lambda x :actionMainUrl(x,l,dataIdList),axis=1)
train.head()
nbDevice = data[['bidder_id','device']].drop_duplicates().groupby('bidder_id').count()['device']
def device(line,groupedData,dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return groupedData[line['bidder_id']]
    
train['nbDevice'] = train.apply(lambda x: device(x,nbDevice,dataIdList),axis=1)
test['nbDevice'] = test.apply(lambda x: device(x,nbDevice,dataIdList),axis=1)
train.head()
HumanList = train[train['outcome'] == 0]['bidder_id'].values
RobotList = train[train['outcome'] == 1]['bidder_id'].values

def flagBids(x,HumanList,RobotList):
    if x in RobotList:
        return 1
    elif x in HumanList:
        return 2
    else:
        return 0

data['flag'] = data['bidder_id'].apply(lambda x : flagBids(x,HumanList,RobotList))
data.head()
dataflagged = data[data['flag']==1]
dataHuman = data[data['flag']==2]
m = dataflagged.groupby('url').count()['bid_id']
n = dataHuman.groupby('url').count()['bid_id']

mkey = m.keys().values
nkey = n.keys().values
k = m.copy().astype(float)
r1 = list(set(mkey).intersection(nkey))
k[r1] = m[r1] / (m[r1]+n[r1])
flag_url = k[r1][k[r1]>0.75].keys().values
def robotURL(x,flag_url):
    if x in flag_url:
        return 1
    else :
        return 0

data['roboturl'] =  data['url'].apply(lambda x : robotURL(x,flag_url))

dataRobot = data[data['roboturl']==1]
dataRobotId = dataRobot['bidder_id'].unique()
groupedData = dataRobot[['bidder_id','auction']].groupby('bidder_id').count()['auction']
def numberofActions(line,dataGrouped,dataid,dataRobotId):
    if not line['bidder_id'] in dataid:
        return 0
    elif not line['bidder_id'] in dataRobotId:
        return 0
    else :
        return float(dataGrouped[line['bidder_id']])/line['nbActions']
    
train['actionsRoboturl_flag'] = train.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)
test['actionsRoboturl_flag'] = test.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)

m = dataflagged.groupby('device').count()['bid_id']
n = dataHuman.groupby('device').count()['bid_id']

mkey = m.keys().values
nkey = n.keys().values
k = m.copy().astype(float)
r1 = list(set(mkey).intersection(nkey))
k[r1] = m[r1] / (m[r1]+n[r1]) 

flag_phone = k[r1][k[r1]>0.75].keys().values

def robotDevice(x,flag_phone):
    if x in flag_phone:
        return 1
    else :
        return 0

data['robotDevice'] =  data['device'].apply(lambda x : robotDevice(x,flag_phone))

dataRobot = data[data['robotDevice']==1]
dataRobotId = dataRobot['bidder_id'].unique()
groupedData = dataRobot[['bidder_id','auction']].groupby('bidder_id').count()['auction']
def numberofActions(line,dataGrouped,dataid,dataRobotId):
    if not line['bidder_id'] in dataid:
        return 0
    elif not line['bidder_id'] in dataRobotId:
        return 0
    else :
        return float(dataGrouped[line['bidder_id']])/line['nbActions']
    
train['actionsRobotDevice_flag'] = train.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)
test['actionsRobotDevice_flag'] = test.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)

m = dataflagged.groupby('country').count()['bid_id']
n = dataHuman.groupby('country').count()['bid_id']

mkey = m.keys().values
nkey = n.keys().values
k = m.copy().astype(float)
r1 = list(set(mkey).intersection(nkey))
k[r1] = m[r1] / (m[r1]+n[r1]) 

flag_country = k[r1][k[r1]>0.75].keys().values

def robotCountry(x,flag_country):
    if x in flag_country:
        return 1
    else :
        return 0

data['robotCountry'] =  data['country'].apply(lambda x : robotCountry(x,flag_country))

dataRobot = data[data['robotCountry']==1]
dataRobotId = dataRobot['bidder_id'].unique()
groupedData = dataRobot[['bidder_id','auction']].groupby('bidder_id').count()['auction']
def numberofActions(line,dataGrouped,dataid,dataRobotId):
    if not line['bidder_id'] in dataid:
        return 0
    elif not line['bidder_id'] in dataRobotId:
        return 0
    else :
        return float(dataGrouped[line['bidder_id']])/line['nbActions']
    
train['actionsRobotCountry_flag'] = train.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)
test['actionsRobotCountry_flag'] = test.apply(lambda x: numberofActions(x,groupedData,dataIdList,dataRobotId),axis=1)

train.head()
data.head()
test.head()
from sklearn.ensemble import RandomForestClassifier


train_r = train

X_train = train_r.drop(['bidder_id','outcome','payment_account','address'],axis=1)
y_train = train_r['outcome']

from sklearn.cross_validation import cross_val_score

rand = RandomForestClassifier(n_estimators=600,max_depth=15,min_samples_leaf=2)
print (cross_val_score(rand,X_train,y_train,cv=3).mean())