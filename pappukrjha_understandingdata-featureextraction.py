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
# --- Read Data ---
trainSample = pd.read_csv('../input/train_sample.csv')
#testSplmnt  = pd.read_csv('../input/test_supplement.csv')
#train       = pd.read_csv('../input/train.csv')
test        = pd.read_csv('../input/test.csv') 
sampleSubmission = pd.read_csv('../input/sample_submission.csv')
# --- Info ---
trainSample.info()
test.info()
# --- View Sample Data ---
print(trainSample.head())
print(test.head())
# --- is_attributed ---
print('Total Attribution :\n', trainSample['is_attributed'].sum())
print('Distribution of is_attributed :\n', trainSample['is_attributed'].value_counts())
# --- Rest of the Variable ---
for col in trainSample.columns:
    if col != 'is_attributed':
        print('\n --- Getting Stats for ' + col + ' --- \n')
        print('Unique Count :', trainSample[col].nunique())
        print('Distribution \n', trainSample[col].value_counts()[:5])
        tmpDf = pd.DataFrame(trainSample.groupby(col)['is_attributed'].sum().reset_index())
        tmpDf.columns = [col, 'isAttributed']
        tmpDf.sort_values('isAttributed', ascending = False, inplace = True)
        print('Top Attributed Values \n', tmpDf.head())
        print('Reverse Distribution \n', tmpDf.groupby('isAttributed')[col].count())
# --- Bar Plot ---
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
sns.set_style('darkgrid')


def barPlot(df, yAxis, xAxis):
    xTicks = df[xAxis]
    yTicks = df[yAxis]
    plt.bar(df[xAxis].index, df[yAxis])
    plt.xticks(df[xAxis].index, xTicks)
    #plt.yticks(yTicks)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()
# --- Attributed Data Only ---

attributedOnly = None
for col in trainSample.columns:
    attributedOnly = trainSample.query('is_attributed==1')
    if col not in ['is_attributed', 'attributed_time', 'click_time'] :
        print('\n --- Getting Stats for ' + col + ' --- \n')
        
        print('Unique Count :', attributedOnly[col].nunique())
        
        
        print('Distribution of Count for ', col, ': \n')
        tmpDf = attributedOnly[col].value_counts()[:5].reset_index()
        tmpDf.columns = [col, 'appInstallCount']
        tmpDf[col] = tmpDf[col].map(str)
        print(tmpDf)
        barPlot(tmpDf, 'appInstallCount', col)
        
        
        print('Distribution of Sum of is Attributed for :', col , '\n')
        tmpDf = pd.DataFrame(attributedOnly.groupby(col)['is_attributed'].sum().reset_index())
        tmpDf.columns = [col, 'isAttributed']
        tmpDf.sort_values('isAttributed', ascending = False, inplace = True)
        headTmpDf = tmpDf.head().reset_index()
        print('Top Attributed Values \n', headTmpDf)
        barPlot(headTmpDf, 'isAttributed', col)
        
        
        print('Reverse Distribution \n')
        tmpDf = tmpDf.groupby('isAttributed')[col].count().reset_index()
        tmpDf.columns = ['isAttributed', 'sumIsAttributed']
        tmpDf.sort_values('sumIsAttributed', inplace = True, ascending = False)
        headTmpDf = tmpDf.head().reset_index()
        del headTmpDf['index']
        print(headTmpDf)
        barPlot(headTmpDf, 'sumIsAttributed', 'isAttributed')        
trainSample['click_time'] = pd.to_datetime(trainSample['click_time'])
trainSample['clickDate'] = trainSample['click_time'].dt.date
trainSample['clickHour'] = trainSample['click_time'].dt.hour
trainSample['clickWOD'] = trainSample['click_time'].dt.dayofweek
# --- Distribution by Day of Week ---
tmpDf = trainSample['clickWOD'].value_counts().reset_index()
tmpDf.columns = ['WOD', 'count']
print(tmpDf)
barPlot(tmpDf, 'count', 'WOD')
# --- Distribution by Hour of Day ---
tmpDf = trainSample['clickHour'].value_counts().reset_index()
tmpDf.columns = ['HOD', 'count']
print(tmpDf)
barPlot(tmpDf, 'count', 'HOD')
# --- Count Level Features ---
trainSampleDict = trainSample.to_dict(orient = 'records')

dictList = []
tmpDict = {}

def getCount(val):
    
    if val in tmpDict:
        tmpDict[val] += 1
    else:
        tmpDict[val] = 1
        
    return tmpDict
 
    
for thisRow in trainSampleDict:
    
    getCount('ip' + str(thisRow['ip']))
    getCount('ap' + str(thisRow['app']))
    getCount('dv' + str(thisRow['device']))
    getCount('os' + str(thisRow['os']))
    getCount('ch' + str(thisRow['channel']))

    
for thisRow in trainSampleDict:
    
    thisRow['ipCount']      = tmpDict['ip' + str(thisRow['ip'])]
    thisRow['appCount']     = tmpDict['ap' + str(thisRow['app'])]
    thisRow['deviceCount']  = tmpDict['dv' + str(thisRow['device'])]
    thisRow['osCount']      = tmpDict['os' + str(thisRow['os'])]
    thisRow['channelCount'] = tmpDict['ch' + str(thisRow['channel'])]
    dictList.append(thisRow)

df = pd.DataFrame(dictList)
df.head(3)
# 1.#App in use from an IP
# 2.#Device in USe from an IP
# 3.#OS from an IP
# 3.Time spent between first and last click from an IP

tmpDict   = {}
appSet    = set()
deviceSet = set()
osSet     = set()

dictList = []
for thisRow in df.to_dict(orient = 'records'):
    if thisRow['app'] in appSet:
        tmpDict['ap' + str(thisRow['ip'])] += 1
    else:
        tmpDict['ap' + str(thisRow['ip'])] = 1
    
    if thisRow['device'] in deviceSet:
        tmpDict['dv' + str(thisRow['ip'])] += 1
    else:
        tmpDict['dv' + str(thisRow['ip'])] = 1   
        
    if thisRow['os'] in deviceSet:
        tmpDict['os' + str(thisRow['ip'])] += 1
    else:
        tmpDict['os' + str(thisRow['ip'])] = 1     
    
    appSet.add(str(thisRow['app']))
    deviceSet.add(str(thisRow['device']))
    osSet.add(str(thisRow['os']))
    
    
for thisRow in df.to_dict(orient = 'records'):
    thisRow['appOnIp'] = tmpDict['ap' + str(thisRow['ip'])]
    thisRow['deviceOnIp'] = tmpDict['dv' + str(thisRow['ip'])]
    thisRow['osOnIp'] = tmpDict['os' + str(thisRow['ip'])]
    dictList.append(thisRow)
    
df = pd.DataFrame(dictList)
trainSample.head()
# Feature - How many OS on a single device

# Time Based Feature

