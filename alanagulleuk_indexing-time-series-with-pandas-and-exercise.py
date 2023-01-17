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
df_humidity=pd.read_csv('../input/humidity.csv')

df_humidity.head()



df_humidityIndexed=pd.read_csv('../input/humidity.csv',parse_dates=True,index_col='datetime')

#df_humidityIndexed.info()   
df_humidityIndexed.loc['2012-10-01 ',('Portland','Seattle')]



 
#Selecting sigle day

df_humidityIndexed.loc['February 2,2013'].head()



#Selecting whole month 

df_humidityIndexed.loc['2015-2']



#or

df_humidityIndexed.loc['2015-Feb']



#or

df_humidityIndexed.loc['February,2015']

df_humidityIndexed.loc['2015-2-16':'2015-2-18']