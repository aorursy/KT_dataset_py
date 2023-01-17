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
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
#Read the data

cust_data = pd.read_csv('../input/CustomerLoanData.csv')
#sample the data

cust_data.head()
#Dimension of Data

cust_data.shape
#Column ames of data

cust_data.columns
#Structure of Data

cust_data.dtypes
#summary of data

cust_data.describe()
trainx,testx,trainy,testy = train_test_split(cust_data.iloc[:,:-1],cust_data.iloc[:,-1],test_size=0.3,random_state=1)

print(cust_data.shape)

print(trainx.shape)

print(testx.shape)
cat_cols = ["edu","securities","cd","online","cc","infoReq"]

num_cols = trainx.columns.difference(cat_cols)

num_cols
trainx[cat_cols] = trainx[cat_cols].apply(lambda x: x.astype('category'))

trainx[num_cols] = trainx[num_cols].apply(lambda x: x.astype('float'))

trainx.dtypes
trainx.isnull().sum()
num_data = trainx.loc[:,num_cols]

cat_data = trainx.loc[:,cat_cols]
# Numeric columns imputation

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

num_data = pd.DataFrame(imp.fit_transform(num_data),columns=num_cols)



# Categorical columns imputation

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

cat_data = pd.DataFrame(imp.fit_transform(cat_data),columns=cat_cols)





print(num_data.isnull().sum())

print(cat_data.isnull().sum())
standardizer = StandardScaler()

standardizer.fit(num_data)

num_data= pd.DataFrame(standardizer.transform(num_data),columns=num_cols)



trainx = pd.concat([num_data,cat_data], axis=1)

bins = [trainx.ccAvg.min(),np.median(trainx.ccAvg),trainx.ccAvg.max()]

group_names = ['low', 'high']

trainx['cat_cc'] = pd.cut(trainx['ccAvg'],bins, labels=group_names)

trainx.head()
trainx=pd.get_dummies(trainx,columns=cat_cols.extend(['cat_cc']))
trainx.head()