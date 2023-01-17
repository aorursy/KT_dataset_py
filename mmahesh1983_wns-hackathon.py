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
from sklearn.linear_model import LogisticRegression

random_seed = 25
df_viewlog = pd.read_csv('/kaggle/input/wns wizard 2019/Train/view_log.csv')

df_itemdata = pd.read_csv('/kaggle/input/wns wizard 2019/Train/item_data.csv')

df_train=pd.read_csv('/kaggle/input/wns wizard 2019/Train/train.csv',nrows=5000)
df_test=pd.read_csv('/kaggle/input/wns wizard 2019/test.csv')
df_traindata = df_train.set_index('user_id').join(df_viewlog.set_index('user_id'),lsuffix='_m').reset_index()

df_testdata = df_test.set_index('user_id').join(df_viewlog.set_index('user_id'),lsuffix='_m').reset_index()
#df_traindata.head()
df_fulldata = df_traindata.set_index('item_id').join(df_itemdata.set_index('item_id'),lsuffix='_m').reset_index()

#df_fulldata.head()
df_fulltestdata = df_testdata.set_index('item_id').join(df_itemdata.set_index('item_id'),lsuffix='_m').reset_index()

#df_fulltestdata.head()
df_fulldata.nunique()


#df_featureData = df_fulldata[features]
osversion = {'latest':0,'intermediate':1,'old':2}
df_fulldata['os_versionnum'] = df_fulldata.os_version.apply(lambda x: osversion[x])

#df_fulldata.head()
df_fulltestdata['os_versionnum'] = df_fulltestdata.os_version.apply(lambda x: osversion[x])
devicetype = {'android':1,'iphone':2,'web':3}
df_fulldata['devicetype_code'] = df_fulldata.device_type.apply(lambda x: devicetype[x])

#df_fulldata.head()
df_fulltestdata['devicetype_code'] = df_fulltestdata.device_type.apply(lambda x: devicetype[x])
#df_featuredata = df_fulldata.drop(columns=['device_type','os_version'],axis=1)
df_fulldata = df_fulldata.fillna(value=0)

df_fulltestdata = df_fulltestdata.fillna(value =0)
id_column = 'impression_id'

label = 'is_click'

features = ['app_code','os_versionnum','is_4G','devicetype_code','item_price','category_1','category_2','category_3','product_type']
#df_featuredata.head()
model = LogisticRegression(max_iter=500,solver='lbfgs')
model.fit(df_fulldata[features],df_fulldata[label])
df_fulltestdata[label] = model.predict(df_fulltestdata[features])
#df_fulltestdata.head()
df_fulltestdata[[id_column,label]].to_csv('WNS_Submission.csv',index=False)