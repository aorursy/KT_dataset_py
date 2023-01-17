# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)}



pd.set_option('max_rows',100)
df = pd.read_csv('/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2018.csv')

df = df[df['ORIGIN'].isin(['ATL','LAX','ORD','DFW','DEN','JFK','SFO','SEA','LAS','MCO'])]



df_2017 = df.append( pd.read_csv('/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2017.csv'))

df_2017 = df_2017[df_2017['ORIGIN'].isin(['ATL','LAX','ORD','DFW','DEN','JFK','SFO','SEA','LAS','MCO'])]

df = df.append(df_2017)



df_2016 = df.append( pd.read_csv('/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2016.csv'))

df_2016 = df_2016[df_2016['ORIGIN'].isin(['ATL','LAX','ORD','DFW','DEN','JFK','SFO','SEA','LAS','MCO'])]

df = df.append(df_2016)



#df_2015 = df.append( pd.read_csv('/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2015.csv'))

#df_2015 = df_2015[df_2015['ORIGIN'].isin(['ATL','LAX','ORD','DFW','DEN','JFK','SFO','SEA','LAS','MCO'])]

#df = df.append(df_2015)
df.groupby('FL_DATE')[['ARR_DELAY']].median().sort_values(by='ARR_DELAY',ascending=False)[:10]