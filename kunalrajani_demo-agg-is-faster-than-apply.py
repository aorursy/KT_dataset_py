# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# let's read the 'City_time_series.csv' 

# df_city_time_seris = pd.read_csv('../input/City_time_series.csv')

# print top 10 item 

df_city_time_seris.Date = pd.to_datetime(df_city_time_seris.Date)

df_city_time_seris.head()



## The following runs in 5 secs

df_city_time_seris.groupby('RegionName')['Date'].agg({'num_rows':len,'period':lambda x: (x.max()-x.min()).days})



## The following runs in 15 secs

df_city_time_seris.groupby('RegionName').apply(lambda x: pd.Series({'num_rows':x.shape[0],'period':(x.Date.max()-x.Date.min()).days}))
