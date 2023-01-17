# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from datetime import datetime
import time
import sys
# functions

# calculate difference in days between two dates represented by strings
def twoStrDateDifference(date_str_last, date_str_first):
    delta = datetime.strptime(date_str_last, '%Y-%m-%d') - datetime.strptime(date_str_first, '%Y-%m-%d')
    return delta.days
# #######################################################


# read all data sets
train = pd.read_csv('../input/train.csv', nrows = 1000)
print('train is load!')
      
test = pd.read_csv('../input/test.csv', nrows = 1000)
print('test is load!')

dest = pd.read_csv('../input/destinations.csv', nrows = 1000)
print('destinations is load!')
# try to check how many users from train is in test and how many new users is in test
train_users = pd.read_csv('../input/train.csv', usecols = ['user_id'])
print('train_users set is load!')

test_users = pd.read_csv('../input/test.csv',  usecols = ['user_id'])
print('test_users set is load!')
nCommonUsers = len(set(test_users.values[:,0]).intersection(set(train_users.values[:,0])))
nTestUsers = len(set(test_users.values[:,0]))
nTrainUsers = len(set(train_users.values[:,0]))

print ('train set has ', nTrainUsers, ' users')
print ('test set has ', nTestUsers, ' users')

print ('test set has ', nCommonUsers, ' users from train set')
print ('test set has ', nTestUsers - nCommonUsers, ' new users')
train_chunk = pd.read_csv('../input/train.csv', chunksize = 100000)
test_chunk = pd.read_csv('../input/test.csv', chunksize = 100000)
one_user_df = pd.DataFrame(columns=train.keys())
one_user_df_test = pd.DataFrame(columns=test.keys())
t1 = time.time()

for chunk in train_chunk:
    one_user_df = pd.concat([one_user_df, chunk[chunk['user_id'] == 28]])

for chunk in test_chunk:
    one_user_df_test = pd.concat([one_user_df_test, chunk[chunk['user_id'] == 28]])
    
print ('one_user_df and one_user_df_test are calculated!', (time.time() - t1)/60)
one_user_df
one_user_df_test
# find all hotes clusters
hotel_clusters = pd.read_csv('../input/train.csv', usecols=['hotel_cluster'])
hotel_clusters.hist()
hotel_clusters_content = pd.read_csv('../input/train.csv', usecols=['hotel_cluster', 'srch_destination_id', 'hotel_country', 'hotel_continent','hotel_market'])
hotel_clusters_content.head()
sys.getsizeof(hotel_clusters_content)
def getCountryListForCluster(cluster_id):
    res = np.unique(hotel_clusters_content[hotel_clusters_content['hotel_cluster'] == cluster_id]['hotel_country'].values)
    return (res, res.size)

def getContinentListForCluster(cluster_id):
    res = np.unique(hotel_clusters_content[hotel_clusters_content['hotel_cluster'] == cluster_id]['hotel_continent'].values)
    return (res, res.size)

def getMarketListForCluster(cluster_id):
    res = np.unique(hotel_clusters_content[hotel_clusters_content['hotel_cluster'] == cluster_id]['hotel_market'].values)
    return (res, res.size)
getMarketListForCluster(3)