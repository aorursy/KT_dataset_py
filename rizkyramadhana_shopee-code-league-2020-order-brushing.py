# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/shopee-code-league-20/_DA_Order_Brushing/order_brush_order.csv')

data.head()

data.event_time = pd.to_datetime(data.event_time)
shopid = data['shopid'].unique()

userid = ['0']*len(data['shopid'].unique())

submission = pd.DataFrame({'shopid':shopid, 'userid':userid})
data['is_brushing'] = [False]*len(data)
for x in data.shopid.unique() :

    orders_x = data[data['shopid']==x] #filtering orders in shopid x only

    orders_x.reset_index(drop = True, inplace = True)

    for n in range(0,len(orders_x)) :   

        mask = orders_x[(orders_x['event_time'] >= orders_x.loc[n,'event_time']) & (orders_x['event_time'] <= (orders_x.loc[n,'event_time'] + timedelta(seconds=3600))) ] #filtering orders which is in one hour period

        conc_rate = len(mask)/len(mask['userid'].unique()) #calculating concentrate rate

        if conc_rate >=3 :

            selector = data['orderid'].isin(mask['orderid']) #creating boolean list. Its value will be True if certain orderid is considered as a brushing order

            data.loc[selector, 'is_brushing'] = True #assigning True for column 'is_brushing'

    brushing_orders_x = data[(data['shopid']==x) & (data['is_brushing']==True)] #list of orders in shopid x that is considered as a brushing order

    brushing_userid_list = [str(i) for i in brushing_orders_x['userid'].mode().sort_values()] #create a string list of userid that did brushing orders

    brushing_userids = '&'.join(brushing_userid_list) #join the list

    submission.loc[submission['shopid']==x, 'userid'] = brushing_userids
submission.to_csv('../input/output/submission.csv', index=False)