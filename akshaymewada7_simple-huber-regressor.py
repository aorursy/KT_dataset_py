# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import string

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv',sep='\t')

test = pd.read_csv('../input/test.tsv',sep='\t')
def Fill_Null_Values(data):

    data = data.fillna(method='backfill')

    data = data.fillna(method='ffill')

    return data
train = Fill_Null_Values(train)

test = Fill_Null_Values(test)
train.head()
Price = train['price']

feature_train = train.drop(['name','train_id','price','item_description'],axis =1)

feature_test = test.drop((['name','test_id','item_description']),axis=1)
def Cat_Data(data):

    d = data.category_name.str.split('/|')

    d2 = pd.DataFrame(d.values.tolist())

    data[['cat1','cat2','cat3']]=  d2[[0,1,2]]

    data = data.drop(['category_name'],axis=1)

    return data
feature_train  = Cat_Data(feature_train)

feature_test = Cat_Data(feature_test)
from sklearn.preprocessing import LabelEncoder

def Encoding(data,data2):    

    a = data.unique().tolist()

    b = data2.unique().tolist()

    c  = np.sort(list(set(a + b))) 

    lb = LabelEncoder()

    lb.fit(c)

    return lb.transform(data),lb.transform(data2)
feature_train['brand_name'],feature_test['brand_name']  = Encoding(feature_train['brand_name'],feature_test['brand_name'])

feature_train['cat1'],feature_test['cat1'] = Encoding(feature_train['cat1'],feature_test['cat1'])

feature_train['cat2'],feature_test['cat2'] = Encoding(feature_train['cat2'],feature_test['cat2'])

feature_train['cat3'],feature_test['cat3'] = Encoding(feature_train['cat3'],feature_test['cat3'])
feature_train.head()
from sklearn.linear_model import HuberRegressor

s = HuberRegressor()
s.fit(feature_train,Price)
p = s.predict(feature_test)
sub =  pd.DataFrame(test['test_id'],columns=['test_id'])
sub['price'] = p
sub.to_csv('submission.csv',index=False)