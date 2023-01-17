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
data = pd.read_csv('../input/students-2-shopee-code-league-order-brushing/order_brush_order.csv', index_col=0)

data.event_time = data.event_time.astype('datetime64')
# print(data)
sorted = data.sort_values(by=['shopid', 'event_time', 'userid'])
# print(sorted)
group = sorted.groupby(['shopid', 'userid'])
# print(group.head())
aggData = group.event_time.agg(['min', 'max', 'count'])

aggData = aggData.reset_index()
aggData

morethan3 = aggData[aggData['count']>=3]
morethan3.userid = morethan3.userid.astype('str')
print(morethan3)
morethan3res = morethan3.groupby(['shopid'])["userid"].agg("&".join)
morethan3res.drop(columns=['min','max','count'])
morethan3res  = pd.DataFrame((morethan3res))
morethan3res= morethan3res.reset_index()
morethan3res
zero = aggData[aggData['count']<3]
zero.loc['userid']=0
zero.drop(columns=['min','max','count'],inplace=True)
zeronew = zero.loc[~zero['shopid'].isin(morethan3res['shopid'])]
zero.userid = zero.userid.astype('str')
zero
morethan3res =morethan3res[['shopid','userid']]
res = pd.concat((zeronew,morethan3res))
res
res.drop_duplicates(subset='shopid',inplace=True)
res.to_csv('result4.csv',index=False)
len(data.shopid.unique())
res
res = res.loc[res['shopid']!=0]
len(res.shopid.unique())
zero.set_index('shopid')

filter1 = aggData[aggData['count'] >= 3]
filter2 = filter1[filter1['max'] - np.timedelta64(1,'h') <= filter1['min']]

# filter2.reset_index(filter2["shopid"])
# filter2.groupby("shopid").head()



output = pd.DataFrame([filter2['shopid'], filter2['userid']]).transpose()
output.userid = output.userid.astype('str')
output = output.groupby(['shopid'])["userid"].agg("&".join)
print(output)
output.to_csv('predictions.csv')