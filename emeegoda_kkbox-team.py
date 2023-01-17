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
members = pd.read_csv('../input/members_v2.csv')
train = pd.read_csv('../input/train.csv')
transactions = pd.read_csv('../input/transactions.csv')
#user_logs = pd.read_csv('../input/user_logs.csv')
members.head()
members.groupby('msno')
transactions.head() #use parse dates in next import
train.head()
train.shape
transactions.shape
members.shape
df = pd.merge(train, transactions, how = 'inner', on = 'msno')
df.shape
df = pd.merge(df, members, how = 'left', on = 'msno')
df.shape
df.head()
df.msno.nunique()
df.dtypes
df[['payment_plan_days', 'plan_list_price']].agg(['min', 'average', 'max'])
df['membership_expire_date'] = pd.to_datetime(df['membership_expire_date'], format='%Y%m%d')
df.head()



#df = df.drop('membership_expire_date_clean', axis=1)
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')
df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d')
df.head()
df[['transaction_date', 'membership_expire_date', 'registration_init_time']].describe()
df[df.membership_expire_date == '2023-08-17']
df[df.msno == 'f/CqixCvjsoTwQRY8A09SMBMsM0cRcG8BSUe48Bd2Mg='].sort_values('membership_expire_date')
df.groupby('payment_method_id').count()
df[df.payment_plan_days == 30].groupby('plan_list_price').count()
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')



%config InlineBackend.figure_format = 'retina'

%matplotlib inline

#import datetime as dt



#df['membership_expire_date_tuple'] = dt.timetuple(df.membership_expire_date)
import matplotlib.pyplot as plt





ax = plt.gca()

ax.hist(df['membership_expire_date'].values, bins= 30)
df['membership_expire_date'].groupby([df['membership_expire_date'].dt.year, df['membership_expire_date'].dt.month]).count()
df.groupby(df.msno).max().reset_index()