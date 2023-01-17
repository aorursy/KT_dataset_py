import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/planecrashinfo_20181121001952.csv", encoding='ISO-8859-1')
df = df.replace('?', np.nan)
df = df.drop(['time', 'flight_no', 'cn_ln'], axis=1)
df.loc[428, 'location'] = 'Selby, North Yorkshire, United Kingdom'

df.loc[605, 'location'] = 'Zhengzhou-Xinzheng, China'

df.loc[620, 'location'] = 'Muhlberg, Germany'

df.loc[966, 'location'] = 'Yulin, China'

df.loc[3155, 'location'] = 'Griffin, USA'

df.loc[4244, 'location'] = 'Dori, Burkina Faso'
df['year'] = [int(each.split(', ')[1]) for each in df['date']]

df['date'] = [each.split(', ')[0] for each in df['date']]

df['month'] = [str(each.split(' ')[0]) for each in df['date']]

df['day'] = [int(each.split(' ')[1]) for each in df['date']]

df = df.drop(['date'], axis=1)
df['crashed_in_country'] = [str(each.split(' ')[-1]) for each in df['location']]

df = df.drop(['location'], axis=1)
df['fatalities'] = [each.split(' ')[0] for each in df['fatalities']]



df.loc[101, 'fatalities'] = 1

df.loc[345, 'fatalities'] = 2

df.loc[377, 'fatalities'] = 2

df.loc[440, 'fatalities'] = 16

df.loc[566, 'fatalities'] = 25

df.loc[605, 'fatalities'] = 16

df.loc[628, 'fatalities'] = 28

df.loc[724, 'fatalities'] = 25

df.loc[821, 'fatalities'] = 25

df.loc[1413, 'fatalities'] = 33

df.loc[4912, 'fatalities'] = 28



df['fatalities'] = [int(x) for x in df['fatalities']]
df['aboard'] = [each.split(' ')[0] for each in df['aboard']]



for x in df[df['aboard']=='?'].index:

    if df.loc[x, 'year']>1950:

        df.loc[x, 'aboard'] = 150

    else:

        df.loc[x, 'aboard'] = 50

        

df['aboard'] = [int(x) for x in df['aboard']]        
for each in df[df['ground'].isnull()].index:

    df.loc[each, 'ground'] = 0
df.columns
out_data = pd.DataFrame(df)

out_data.columns= df.columns

out_data.to_excel('Extra File for Tableau for plane crashes.xlsx', header=True)