# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
events_dt = pd.read_csv("/kaggle/input/ecommerce-dataset/events.csv")
sp1 = events_dt.sample(n = 100, random_state = 1)

sp2 = events_dt.sample(n=100, random_state = 2)

sp3 = events_dt.sample(n=100, random_state = 3)

sp4 = events_dt.sample(n=100, random_state = 4)

sp5 = events_dt.sample(n=100, random_state = 5)

sp6 = events_dt.sample(n=100, random_state = 6)

sp7 = events_dt.sample(n=100, random_state = 7)

sp8 = events_dt.sample(n=100, random_state = 8)

sp9 = events_dt.sample(n=100, random_state = 9)

sp10 = events_dt.sample(n=100, random_state = 10)
sp1['nitemid'] =  (sp1['itemid']%10)

sp2['nitemid'] =  (sp2['itemid']%10)

sp3['nitemid'] =  (sp3['itemid']%10)

sp4['nitemid'] =  (sp4['itemid']%10)

sp5['nitemid'] =  (sp5['itemid']%10)

sp6['nitemid'] =  (sp6['itemid']%10)

sp7['nitemid'] =  (sp7['itemid']%10)

sp8['nitemid'] =  (sp8['itemid']%10)

sp9['nitemid'] =  (sp9['itemid']%10)

sp10['nitemid'] =  (sp10['itemid']%10)
sp1 = sp1.groupby(['nitemid','event']).size()

sp1 = sp1.unstack()

sp2 = sp2.groupby(['nitemid','event']).size()

sp2 = sp2.unstack()

sp3 = sp3.groupby(['nitemid','event']).size()

sp3 = sp3.unstack()

sp4 = sp4.groupby(['nitemid','event']).size()

sp4 = sp4.unstack()

sp5 = sp5.groupby(['nitemid','event']).size()

sp5 = sp5.unstack()

sp6 = sp6.groupby(['nitemid','event']).size()

sp6 = sp6.unstack()

sp7 = sp7.groupby(['nitemid','event']).size()

sp7 = sp7.unstack()

sp8 = sp8.groupby(['nitemid','event']).size()

sp8 = sp8.unstack()

sp9 = sp9.groupby(['nitemid','event']).size()

sp9 = sp9.unstack()

sp10 = sp10.groupby(['nitemid','event']).size()

sp10 = sp10.unstack()
sp1.plot(kind='bar')

sp2.plot(kind='bar')

sp3.plot(kind='bar')

sp4.plot(kind='bar')

sp5.plot(kind='bar')

sp6.plot(kind='bar')

sp7.plot(kind='bar')

sp8.plot(kind='bar')

sp9.plot(kind='bar')

sp10.plot(kind='bar')
view1 = sp1['view']

view2 = sp2['view']

view3 = sp3['view']

view4 = sp4['view']

view5 = sp5['view']

view6 = sp6['view']

view7 = sp7['view']

view8 = sp8['view']

view9 = sp9['view']

view10 = sp10['view']

allview = [view1, view2, view3, view4, view5, 

           view6, view7, view8, view9, view10]

box1 = plt.boxplot(allview,vert=0,patch_artist=True, 

                   labels = ['view1', 'view2', 'view3', 'view4', 'view5',

                             'view6', 'view7', 'view8', 'view9', 'view10'])