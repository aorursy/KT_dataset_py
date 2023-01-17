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

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("/kaggle/input/montcoalert/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





d.sort_values(by=['timeStamp'], ascending=False, inplace=True)

d.head()

t=d[d['timeStamp']>= '2015-01-01']





p = pd.pivot_table(t, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)





# h 1 hour intervals

# d 1 day 

# w 1 week

pp = p.resample('1w').apply(np.sum).reset_index()





pp.columns = pp.columns.get_level_values(0)



pp.fillna(0, inplace=True)

pp.sort_values(by=['timeStamp'], ascending=False, inplace=True)

pp[['timeStamp','EMS: FEVER']].head(9)
t=pp[pp['timeStamp']>= '2016-01-02']





from matplotlib.ticker import FuncFormatter



fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False) 







ax.plot(t['timeStamp'],t['EMS: FEVER'],color = 'red')



ax.set_title("EMS: FEVER /week", 

             fontsize=12, color='darkslateblue')





ax.grid(b=True, which='major', color='wheat', linestyle='-')





plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")  





 



fig.autofmt_xdate()

plt.show()