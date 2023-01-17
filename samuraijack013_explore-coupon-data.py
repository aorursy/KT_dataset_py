# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.preprocessing import LabelEncoder

from scipy.stats import mode



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

from sklearn.metrics import classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.metrics import auc



import os

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth',500)

pd.set_option('display.max_columns',5000)

from IPython.display import Image

import os

!ls ../input/

encoder = LabelEncoder()
train = pd.read_csv('../input/train.csv')

campaign = pd.read_csv('../input/campaign_data.csv')

items = pd.read_csv('../input/item_data.csv')

coupons = pd.read_csv('../input/coupon_item_mapping.csv')

cust_demo = pd.read_csv('../input/customer_demographics.csv')

cust_tran = pd.read_csv('../input/customer_transaction_data.csv')

test = pd.read_csv('../input/test.csv')
train.head()
campaign.tail()
items.head()
coupons.head()
cust_demo.head()
cust_tran.head()
test.head()
# check count of customers by campaings

train.groupby(['campaign_id']).agg({'customer_id':'nunique','coupon_id':'unique'})



test.groupby(['campaign_id']).agg({'customer_id':'nunique','coupon_id':'unique'})
# campaign to coupon mapping

map_camp_coup = train.groupby(['campaign_id','coupon_id'],as_index = False).first()[['campaign_id','coupon_id']]

map_camp_coup[map_camp_coup['coupon_id'] == 470]
# print(map_camp_coup['coupon_id'].value_counts()[:100])

print("Uniqe Coupons: ",map_camp_coup.coupon_id.nunique())
# map_camp_coup.coupon_id.unique()
train[(train['customer_id'] == 1053) & (train['redemption_status'] == 1)]
cust_tran[(cust_tran['customer_id'] == 188) & (cust_tran['coupon_discount'] > 0)]
campaign[campaign['campaign_id'] == 13]
campaign.sort_values(by = ['campaign_id'],axis = 0)
campaign.campaign_id.unique
# data processing



campaign.head()

# campaign.dtypes

# convert to datetime



# filter for campaings in train data

campaign = campaign[campaign['campaign_id'].isin(train.campaign_id)]



campaign['start_date'] = pd.to_datetime(campaign['start_date'], format = "%d/%m/%y")

campaign['end_date'] = pd.to_datetime(campaign['end_date'], format = "%d/%m/%y")



# unique campaing and coupons



map_camp_coup = train.groupby(['campaign_id','coupon_id'],as_index = False).first()[['campaign_id','coupon_id']]

#QC

# print(map_camp_coup.shape)

map_camp_coup = pd.merge(map_camp_coup,campaign, on = ['campaign_id'], how = 'left')

#QC

# print(map_camp_coup.shape)

map_camp_coup_item = pd.merge(coupons,map_camp_coup, on = ['coupon_id'], how = 'left')



# merge it to customer - campaign - coupon data to bring item information

cust_coup_item_map = pd.merge(train,map_camp_coup_item, on = ['campaign_id','coupon_id'])

cust_tran['date'] = pd.to_datetime(cust_tran['date'], format = "%Y-%m-%d")
x = cust_coup_item_map.merge(cust_tran, how='left', on=['customer_id', 'item_id'])
def map_coupon(x):

    if (x['start_date'] <= x['date']) & (x['end_date'] >= x['date']):

        return 1

    else :

        return 0
x.head()
x.head()

x['coupon_redeemed'] = x.apply(map_coupon,axis = 1)
x_cust = pd.merge(x,cust_demo,on = ['customer_id'], how = 'left')

x_item = pd.merge(x_cust,items,on = ['item_id'], how = 'left')
x_item.sample(n = 100)
x[x['eligible_coupon_bool'] > 0]['coupon_discount'].hist()