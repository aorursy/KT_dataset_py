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
import pandas

order_brush=pandas.io.parsers.read_csv('/kaggle/input/order-brush-order/order_brush_order.csv')

orderid = order_brush['orderid']

shopid = order_brush['shopid']

userid= order_brush['userid']

event_time = order_brush['event_time']

print(order_brush)

#print(order_brush.groupby('shopid').count())

grouped = order_brush.groupby('shopid').count()

#print(grouped)

for i in order_brush['shopid']:

  a = order_brush.loc[order_brush['shopid'] == i]

  for e in a['userid']:

    b = a.loc[a['userid'] == e]

    print(b)

    c = order_brush.index(b)

    num = len(c)

    if num < 3: order_brush.drop(c)

#  group = a.groupby('userid').count()

 # print(group)

order_brush.tocsv('/kaggle/working/')