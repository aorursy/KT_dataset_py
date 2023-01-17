# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
basic_path = "../input/finance/data/data/"

test_path = "test.csv"

train_path = "train.csv"

user_taglist_path = "user_taglist.csv"   #用户的画像信息
data = pd.read_csv(basic_path+train_path)
data.head()
data['repay_date'] = data[['due_date', 'repay_date']].apply(

    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1

)

data['repay_amt'] = data['repay_amt'].apply(

    lambda x: float(x) if x!= '\\N' else 0)

data['tag'] = data['repay_amt'] == 0

data['days'] = pd.to_datetime(data['due_date']) - pd.to_datetime(data['repay_date'])

data['days'] = data['days'].dt.days - data['tag']
for column in ['listing_id','auditing_date','due_date','due_amt','repay_date','repay_amt','tag']:

    del data[column]
data['days'] += 1
data = data.groupby(data['user_id']).mean()

data['user_id'] = data.index

data['days'] = np.array(data['days']+0.5).astype(int)
#使用

user_taglist_data = pd.read_csv(basic_path+user_taglist_path)

user_taglist_data.head()
#保留最新信息

user_taglist_data = user_taglist_data.sort_values(by='insertdate', ascending=False)

user_taglist_data = user_taglist_data.drop_duplicates('user_id').reset_index(drop=True)

del user_taglist_data['insertdate']
user_taglist_data['taglist'] = user_taglist_data['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

cntVector = CountVectorizer(max_features=1000)

cntTf = cntVector.fit_transform(user_taglist_data['taglist']).toarray()
user_taglist = pd.DataFrame(cntTf)

user_taglist['user_id'] = user_taglist_data['user_id']

data = pd.merge(user_taglist,data,on='user_id',how='left')
del cntVector,user_taglist,cntTf,user_taglist_data

gc.collect()
data_ = data.dropna()

train_tag = np.array(data_['days'])

del data_['days']

del data_['user_id']

train_data = np.array(data_)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



k_model = SelectKBest(chi2, k=100)

k_model.fit(train_data,train_tag)

del data_,train_data,train_tag

gc.collect()
user_id = data['user_id']

del data['user_id']

del data['days']
data = np.array(data)

process_data = k_model.transform(data)
del data

gc.collect()
data = pd.DataFrame(process_data)

data['user_id'] = user_id

data.head()
data.to_csv('user_tag_K.csv',index=False)