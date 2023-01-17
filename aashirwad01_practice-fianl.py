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
import matplotlib.pyplot as plt
import seaborn as sns
r_data=pd.read_csv('../input/reviews.csv')
c_data=pd.read_csv('../input/calendar.csv')
l_data=pd.read_csv('../input/listings.csv')
r_data.head(10)
r_data.info()
r_data.isnull().sum()
y=np.array([0,19])
plt.yticks(np.arange(y.min(), y.max(), 3))
r_data.isnull().sum().plot(kind='bar')
l_data.head()
l_data.info()
l_data.describe(include='all')

l_data.isnull().sum()[l_data.isnull().sum().nonzero()[0]]

l_data.isnull().sum()[l_data.isnull().sum().nonzero()[0]].index
type(l_data.isnull().sum()[l_data.isnull().sum().nonzero()[0]])
plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
plt.yticks(np.arange(0,4000,200))
sns.barplot(x=l_data.isnull().sum()[l_data.isnull().sum().nonzero()[0]].index,y=l_data.isnull().sum()[l_data.isnull().sum().nonzero()[0]].values)
c_data.head(10)
c_data.info()
c_data.isnull().sum()

c_data.isnull().sum().plot(kind='bar')
#df = df.filter(df.col_X. isNotNull())
c_data.dropna(axis=0,subset=['price'],inplace=True)
r_data.dropna(axis=0,subset=['comments'],inplace=True)
cr_data=pd.merge(c_data,r_data,on='listing_id')
cr_data.head()

#df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
l_data=l_data.rename(columns={'id':'listing_id'})
#l_data.columns
crl_data=pd.merge(cr_data,l_data,on='listing_id')
#cr_data['month']=int(cr_data['date'].split('-')[1])
#cr_data['year']=int(cr_data['date'].split('-')[0])