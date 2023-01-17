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

dt = pd.read_excel("../input/race4-dosc/race4.xlsx")

#dt

#dt['boatname'].value_counts()
dt = dt.drop_duplicates(subset=['boatname', 'date'], keep='first')

dt['boatname'].value_counts()



dt['time'] = pd.DatetimeIndex(dt['hour'])

dt["delta"] = dt['time'] - min(dt['time'])



dt["delta2"] = 0 + pd.DatetimeIndex( dt["delta"] ).minute  + pd.DatetimeIndex( dt["delta"] ).hour*60 

dt = dt.drop_duplicates(subset=['boatname', 'delta2'], keep='first')

dt['delta2'].value_counts()

dd = dt.loc[dt['delta2'] % 10 ==0,]

dd = dd.loc[dd['delta2']> 0]

xx = dd['delta2'].value_counts().to_frame()

ff = xx.loc[xx['delta2']>1]

my = list(ff.index.values)

my

d2 = dd[dd['delta2'].isin( my)]

#d2['delta2'].value_counts()

import plotly_express as px

px.scatter(d2, x="lon", y="Lat", animation_frame="delta2", animation_group="boatname",

        color="boatname", hover_name="boatname", log_x = False, 

           size_max=4, range_x=[55.100,55.240], range_y=[25.155,25.19])

        
