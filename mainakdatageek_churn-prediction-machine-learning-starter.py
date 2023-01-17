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
import pandas as pd 

import numpy as np

train = pd.read_csv('../input/train.csv')

test_zero = pd.read_csv('../input/sample_submission_zero.csv')

members = pd.read_csv('../input/members.csv')

transactions = pd.read_csv('../input/transactions.csv')

#user_logs = pd.read_csv('../input/user_logs.csv',nrows = 2e7)

# train + members 

training = pd.merge(left = train,right = members,how = 'left',on=['msno'])

training.head()
training_inner  = pd.merge(left = train,right = members,how = 'inner',on=['msno'])

training_inner.shape
training_inner.describe(include='all') 
# test + members

testing = pd.merge(left = test_zero,right = members,how = 'left',on=['msno'])

testing.head()
training.shape, testing.shape
training.describe(include='all')
training.head(10)
testing.describe(include='all')
training['city'] = training.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

training['registered_via'] = training.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

training['gender']=training['gender'].fillna("NAN")

training.info()
testing['city'] = testing.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

testing['registered_via'] = testing.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

testing['gender']=testing['gender'].fillna("NAN")

testing.info()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time

from datetime import datetime

from collections import Counter

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
training['registration_init_time'] = training.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )

training['expiration_date'] = training.expiration_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

training.head()
testing['registration_init_time'] = testing.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )

testing['expiration_date'] = testing.expiration_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

testing.head()
training.is_churn.value_counts()
training.city.value_counts()
training.gender.value_counts()
training.registration_init_time.value_counts()
training.expiration_date.value_counts()
63471/(929460 + 63471)*100 ###Huge imbalance class 
testing.is_churn.value_counts() 
print (testing.shape)

print (training.shape)
training.registered_via.value_counts()
import seaborn as sns 
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

#my_imputer.fit_transform(training)
#training['expiration_date'] = training['expiration_date'].fillna(0,inplace=True)
#training['expiration_date']
d = len(pd.date_range(start=training.registration_init_time,end=training.expiration_date,freq='M'))

d
training.describe(include='all')
training.head()
training['registration_init_time'] = training['registration_init_time'].fillna('blankk')