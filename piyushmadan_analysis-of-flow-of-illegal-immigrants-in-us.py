# Loading library and importing data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting graphs



df = pd.read_csv('../input/arrests.csv')
# df_usa stores total number of arrests in United States by year and total/mexican migrants

df_usa = df.loc[df['Border']=="United States"]



# Total immigration arrests since 2000

df_usa_all_immigrants= df_usa.filter(regex=r'All')

df_usa_all_immigrants.index = list(['All immigrants'])

df_usa_all_immigrants.columns = [int(line[0:4]) for line in df_usa_all_immigrants.columns]



# Mexican immigration arrests since 2000

df_mexican_immigrants= df_usa.filter(regex=r'Mexican')

df_mexican_immigrants.index = list(['Mexican immigrants'])

df_mexican_immigrants.columns = [int(line[0:4]) for line in df_mexican_immigrants.columns]
%matplotlib inline

pd.concat([df_mexican_immigrants,df_usa_all_immigrants]).T.plot(title='Illegal immigrants arrests')
df_border = df.loc[df['Sector']=="All"]

df_border.index = list(df_border['Border'])



# Total immigration arrests since 2000 , borderwise

df_borderwise= df_border.filter(regex=r'All')

df_borderwise.columns = [int(line[0:4]) for line in df_borderwise.columns]

df_borderwise
%matplotlib inline

df_borderwise.T.plot(title='Number of arrests')