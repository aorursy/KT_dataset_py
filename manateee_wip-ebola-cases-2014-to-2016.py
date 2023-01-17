import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/ebola.csv')

df.head()
import re

def covertIndicators(ind):

    result = 'Unknown : Unknown'

    lst_ind = [

        { 'regx': r'Case fatality rate \(CFR\) of (.*) Ebola cases', 'abbv': 'CFR'},

        { 'regx': r'Cumulative number of (.*) Ebola deaths', 'abbv': 'Cumulative Deaths'},

        { 'regx': r'Cumulative number of (.*) Ebola cases', 'abbv': 'Cumulative Cases'},

        { 'regx': r'Proportion of (.*) Ebola deaths that are from the last 21 days', 'abbv': 'Proportion Deaths 21'},

        { 'regx': r'Proportion of (.*) Ebola cases that are from the last 21 days', 'abbv': 'Proportion Cases 21'},

        { 'regx': r'Proportion of (.*) Ebola cases that are from the last 7 days', 'abbv': 'Proportion Cases 07'},

        { 'regx': r'Number of (.*) Ebola deaths in the last 21 days', 'abbv': 'Number Deaths 21'},

        { 'regx': r'Number of (.*) Ebola cases in the last 21 days', 'abbv': 'Number Cases 21'},

        { 'regx': r'Number of (.*) Ebola cases in the last 7 days', 'abbv': 'Number Cases 07'},

    ]

    for i in lst_ind:

        match = re.search(i['regx'], ind)

        if match:

            tmp = str(match.group(1))

            result = i['abbv'] + ' : ' + (tmp if len(tmp) < 20 else 'All')

            break

            

    return result

    
df['ind'] = df['Indicator'].apply(covertIndicators)

df['ind_class'] = df['ind'].apply(lambda x: x.split(' : ')[0])

df['ind_sub'] = df['ind'].apply(lambda x: x.split(' : ')[1])

df.drop(['Indicator', 'ind'], axis=1, inplace=True)

df.head()
df_pv = pd.pivot_table(df, values='value', index=['Country', 'Date'], 

                    columns=['ind_class', 'ind_sub'], aggfunc=np.sum)

df_pv.index.names = ['Country', 'Date']

df_pv.head()
df_pv.loc['Guinea']