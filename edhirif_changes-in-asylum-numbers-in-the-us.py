# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt



df_d = pd.read_csv('../input/defensive_asylum.csv', sep=',', thousands=',', engine='python')
df_da_total = df_d[df_d['Continent/Country of Nationality'] == 'Total']
df_da_total = df_da_total.set_index('Continent/Country of Nationality')
df_da_total = df_da_total.transpose()

df_da_total['Total'] = df_da_total['Total'].astype('float')
df_da_total.plot(title='Total number of individuals granted defensive asylum', legend=False, ylim=(0, 20000))
df_da = df_d

df_da = df_da[:6]

df_da = df_da.set_index('Continent/Country of Nationality')

df_da = df_da.transpose()

df_da = df_da.astype('float')
df_da.plot(title='Origin of individuals granted defensive asylum')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
df_a = pd.read_csv('../input/affirmative_asylum.csv', sep=',', thousands=',', engine='python')

df_aa_total = df_a[df_a['Continent/Country of Nationality'] == 'Total']

df_aa_total = df_aa_total.set_index('Continent/Country of Nationality')

df_aa_total = df_aa_total.transpose()

df_aa_total = df_aa_total.astype('float')



df_aa_total.plot(title='Total number of individuals granted affirmative asylum', legend=False, ylim=(0, 19000))

df_aa = df_a

df_aa = df_aa[:6]

df_aa = df_aa.set_index('Continent/Country of Nationality')

df_aa = df_aa.transpose()

df_aa = df_aa.astype('float')



df_aa.plot(title='Origin of individuals granted affirmative asylum')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))