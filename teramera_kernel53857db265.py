# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tr=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

tr.head()
ts=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

ts.head()
Ts=ts.fillna(method='ffill')

Ts=Ts.fillna(method='bfill')
Tr=tr.fillna(method='ffill')

Tr=Tr.fillna(method='bfill')
Train_data=Tr

Test_data=Ts
cat=Tr.drop(['Id','SalePrice'],axis=1).columns[Tr.drop(['Id','SalePrice'],axis=1).dtypes=='object']

len(cat)
for i in cat:

    dum=pd.get_dummies(Tr[i])

    dum.columns=str(i)+'_'+dum.columns

    Train_data=pd.concat([Train_data,dum],axis=1)

    Train_data.drop(i,axis=1,inplace=True)

    dum=pd.get_dummies(Ts[i])

    dum.columns=str(i)+'_'+dum.columns

    Test_data=pd.concat([Test_data,dum],axis=1)

    Test_data.drop(i,axis=1,inplace=True)
Train_data.head()
Test_data.head()
for i in Train_data.drop('SalePrice',axis=1).columns:

    if i not in Test_data.columns:

        Test_data[i]=np.zeros(len(Test_data))
len(Test_data.columns)
for i in Test_data.columns:

    if i not in Train_data.columns:

        Train_data[i]=np.zeros(len(Train_data))
len(Train_data.columns)
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(Train_data.drop(['Id','SalePrice'],axis=1),Train_data['SalePrice'])
xtr.shape
ytr.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(xtr,ytr)
reg.score(xts,yts)
df=reg.predict(Test_data.drop(['Id'],axis=1))
submission=pd.DataFrame(data=df,columns=['SalePrice'])
submission['Id']=Test_data['Id']
submission.set_index('Id',inplace=True)
submission.to_csv('submission.csv')
sam=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.shape