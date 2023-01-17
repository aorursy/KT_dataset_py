import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_ghg = pd.read_csv('/kaggle/input/co2-and-ghg-emission-data/emission data.csv',index_col=[0])

df_ghg[~df_ghg.index.isin(['World','EU-28','Asia and Pacific (other)','Americas (other)','Middle East','Europe (other)','Africa'])][['2017']].sort_values(by='2017',ascending=False)[:15]