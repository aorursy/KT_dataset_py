## Import necessary packages.
import numpy as np ## linear algebra
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt

#######################
# Data handling steps #
#######################

## Import Data from Website ##
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

## Get Training data ##
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()

## Get Testing data.. probably not? ##
# days = env.get_prediction_days()
# (market_obs_df, news_obs_df, predictions_template_df) = next(days)

## list of variables ##
list(market_train_df)
list(news_train_df)

## Classification tree reference 
## https://scikit-learn.org/stable/modules/tree.html
## https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/    

## Check how the 'assetCode' is distributed ##
market_assetCode = market_train_df['assetCode']
news_assetCode = news_train_df['assetCodes']

## Make a crosstab for 'assetName' ##
market_train_tab = pd.crosstab(index=market_assetCode, columns="count") 
news_train_tab = pd.crosstab(index=news_assetCode, columns="count")

## Our Research Target will be a asset whose assetCode is 'AAPL.O' ## 
assetCodelist = ['AMZN.O','EBAY.O','WMT.N','KR.N','COST.O','TGT.N','HD.N','CVS.N','BBY.N','LOW.N']
Market_IDlist = []
for i in range(len(assetCodelist)):
    Market_IDlist = Market_IDlist + list(np.where(market_assetCode == assetCodelist[i])[0])

## Create Market_TargetData... it is subset of market_train_df which contains only information about 'AAPL.O'
Market_TargetData = market_train_df.iloc[Market_IDlist]

## Create Mk_data... it is subset of Market_TargetData which contains 'time' and 'returnsOpenNextMktres10' only.
Mk_data = Market_TargetData[['assetCode','time','returnsOpenNextMktres10']]

## Create time Index.... we will pick only one data point from datameasures
nt = Mk_data.shape[0]
indexT = []
for i in range(nt):
  num = str(Mk_data['time'].values[i])[:10] + Mk_data['assetCode'].values[i]
  indexT = indexT + [num]

## This can be used as Key value of time
Mk_data['index'] = indexT

## Extract Newsdata from training.. this takes quite long time!!
n = len(news_assetCode)
Ns_data = pd.DataFrame()
for j in range(len(assetCodelist)):
    News_ID = []
    for i in range(n):
        if assetCodelist[j] in news_assetCode[i]:
            News_ID = News_ID + [i]

    ## Create Ns_data... it is subset of news_train_df which contains only information about 'AAPL.O'
    # nt_df[nt_df['sourceId']=='d7ad319ee02edea0'] can be used.. but this time assetsCode is string
    Ns_data_temp = news_train_df.iloc[News_ID]

    ## Create time Index.... we will pick only one data point from datameasures
    nt = Ns_data_temp.shape[0]
    indexT = []
    for i in range(nt):
      num = str(Ns_data_temp['time'].values[i])[:10] + assetCodelist[j]
      indexT = indexT + [num]    
    Ns_data_temp['indexT'] = indexT
    Ns_data = pd.concat([Ns_data, Ns_data_temp], ignore_index=True)

## Set a rownumber for Ns_data before we reduce it.
nt = Ns_data.shape[0]
Ns_data['rownumber'] = range(nt)

## Reduce Ns_data...1. summarize time points to be distinct, and take latest point for each.
tmp = np.unique(Ns_data['indexT'].values)
nt = len(tmp)

## Extract the latest row data which correspond to all disticnt time points. 
rowid = []
for i in range(nt):
  a = Ns_data[Ns_data['indexT'] == tmp[i]]    
  k = a.shape[0]
  # Extract the last row number
  rowid = rowid + [a['rownumber'].values[k-1]]
    
## Take subset of NS_data
Ns_data2 = Ns_data.iloc[rowid]

## Left Join NewsData into Market Data... it creates Joined_Data
Joined_Data = pd.merge(Mk_data, Ns_data2, left_on='index', right_on='indexT', how='left')
Joined_Data[['index','returnsOpenNextMktres10']]

## Now, Define new binary variable Y, which will be our response variable 
tmp = Joined_Data['returnsOpenNextMktres10']
N = len(tmp)
Y = np.zeros(N)
for obs in range(N):
  if tmp[obs] >= 0:
    # Y becomes 1 if the return is positive
    Y[obs] = 1
  else:  
    # Y becomes -1 if the return is  negative
    Y[obs] = -1
Y = list(Y)    

## Include Y into my dataframe
Joined_Data['Y'] = list(Y)    

## Now, we can check the dimension of the Joined_Data is (24980, 42)
Joined_Data.shape

Joined_Data['sentimentProportion'] = Joined_Data['sentimentWordCount'] / Joined_Data['wordCount']

# change following column into category: 'takeSequence'
tmp = Joined_Data['takeSequence']
N = len(tmp)
Y = np.zeros(N)
for obs in range(N):
  if tmp[obs] == 1:
    Y[obs] = 1
  else:  
    Y[obs] = 0
Y = list(Y)
Joined_Data['takeSequency'] = list(Y)

# change following columns into category: 'marketCommentary'
tmp = Joined_Data['marketCommentary']
N = len(tmp)
Y = np.zeros(N)
for obs in range(N):
  if tmp[obs] == 'FALSE':
    Y[obs] = 0
  else:  
    Y[obs] = 1
Y = list(Y)
Joined_Data['marketCommentary'] = list(Y)

# change following columns into category: 'sentimentClass'
tmp = Joined_Data['sentimentClass']
N = len(tmp)
Y1 = np.zeros(N)
Y2 = np.zeros(N)
for obs in range(N):
  if tmp[obs] == 1:
    Y1[obs] = 1
    Y2[obs] = 0
  elif tmp[obs] == 0:
    Y1[obs] = 0
    Y2[obs] = 0
  else:  
    Y1[obs] = 0
    Y2[obs] = 1
Y1 = list(Y1)
Y2 = list(Y2)
Joined_Data['sentimentClassPositive'] = list(Y1)
Joined_Data['sentimentClassNegative'] = list(Y2)

Joined_Data2 = Joined_Data.drop(['sentimentClass','provider','returnsOpenNextMktres10', 'indexT', 'time_x','rownumber','assetCode', 'time_y','sourceTimestamp','firstCreated','sourceId', 'headline', 'subjects', 'audiences', 'bodySize', 'headlineTag', 'sentenceCount', 'wordCount', 'assetCodes', 'assetName', 'firstMentionSentence', 'sentimentNeutral', 'sentimentWordCount', 'noveltyCount12H', 'volumeCounts12H'], axis = 1)

# divide into two groups based on 'index'
train_data = pd.DataFrame()
test_data = pd.DataFrame()
N = len(Joined_Data2)
for obs in range(N):
    if Joined_Data2['index'][obs][0:7] < '2016-06':
        train_data = pd.concat([train_data, Joined_Data2[obs:obs+1]], ignore_index=True)
    else:
        test_data = pd.concat([test_data, Joined_Data2[obs:obs+1]], ignore_index=True)
train_data = train_data.drop(['index'], axis = 1)
test_data = test_data.drop(['index'], axis = 1)

train_data.shape
test_data.shape
# individual part

# fill NAs with zero + 'noNews' column
N = len(train_data)
M = len(test_data)
C = train_data.columns
D = test_data.columns

Y1 = np.zeros(N)
for obs in range(N):
    if train_data['takeSequence'].isna()[obs]:
        Y1[obs] = 1       
    else:
        Y1[obs] = 0
for i in C:
    train_data[i] = train_data[i].fillna(0)
Y1 = list(Y1)
train_data['noNews'] = list(Y1)

Y1 = np.zeros(M)
for obs in range(M):
    if test_data['takeSequence'].isna()[obs]:
        Y1[obs] = 1
    else:
        Y1[obs] = 0
for i in D:
    test_data[i] = test_data[i].fillna(0) 
Y1 = list(Y1)
test_data['noNews'] = list(Y1)
# fit svm w/ 'noNews' column

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X_train = train_data.loc[:, train_data.columns != 'Y'].values
X_test = test_data.loc[:, test_data.columns != 'Y'].values
X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = train_data['Y']
Y_test = test_data['Y']
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Create SVM classification object 
model = svm.SVC(kernel='rbf', gamma = 'auto') 

# there is various option associated with it, like changing kernel, gamma and C value.
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(accuracy_score(Y_test, Y_predict))
confusion_matrix(Y_test, Y_predict)

# fit svm w/o 'noNews' column

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X_train = train_data.loc[:, train_data.columns != 'Y']
X_test = test_data.loc[:, test_data.columns != 'Y']
X_train = X_train.loc[:, X_train.columns != 'noNews'].values
X_test = X_test.loc[:, X_test.columns != 'noNews'].values
X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = train_data['Y']
Y_test = test_data['Y']
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Create SVM classification object 
model = svm.SVC(kernel='rbf', gamma = 'auto') 

# there is various option associated with it, like changing kernel, gamma and C value.
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(accuracy_score(Y_test, Y_predict))
confusion_matrix(Y_test, Y_predict)

# fit svm w/o NA rows

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

N = len(train_data)
M = len(test_data)

templist = list(np.where(train_data['noNews'] == 0)[0])
train_data2 = train_data.iloc[templist]
templist = list(np.where(test_data['noNews'] == 0)[0])
test_data2 = test_data.iloc[templist]

X_train = train_data2.loc[:, train_data2.columns != 'Y']
X_test = test_data2.loc[:, test_data2.columns != 'Y']
X_train = X_train.loc[:, X_train.columns != 'noNews'].values
X_test = X_test.loc[:, X_test.columns != 'noNews'].values
X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = train_data2['Y']
Y_test = test_data2['Y']
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Create SVM classification object 
model = svm.SVC(kernel='rbf', gamma = 'auto') 

# there is various option associated with it, like changing kernel, gamma and C value.
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(accuracy_score(Y_test, Y_predict))
confusion_matrix(Y_test, Y_predict)
