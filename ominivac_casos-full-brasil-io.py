# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/covid-full-data-brasil-io/caso_full.csv", sep=',')
df.head() 
df_group = df.groupby(["state", "city"])["new_deaths"].count()
df_group
df_group_estado = df.groupby(["state"])["new_deaths"].count()
df_group_estado
df_group_estado.plot( kind='bar' )
group_estado = df.groupby(["state"])
group_estado

df_estado = pd.DataFrame(group_estado.new_deaths.sum())
df_estado.head()
print(df_estado.index.name)
df_estado = df_estado.set_index('state')
df_estado = df_estado.reset_index(drop=True)
df_estado.plot( kind='bar')
df_estado = df_estado.reset_index(drop=True)
df_estado

df_estado.plot('state','new_deaths', kind='bar',fontsize = 16)
df_neg  = df[df.new_deaths < 0 ]
df_neg
pd.isnull(df)
group_state = df.groupby(["state"])
group_state

df_graf = pd.DataFrame(group_state.new_deaths.sum())
df_graf
df_neg  = df[df.new_deaths < 0 ]
df_neg
df_null = df[pd.isnull(df).any(axis=1)]
df_null
df_ce  = df[df.state == 'CE']
df_ce
df_municipios = df_ce.groupby(['city'], sort='True').sum()
df_municipios = df_ce.sort_values(by = ['new_deaths'], ascending=[False])
df_municipios
df1 = df_municipios[['city', 'new_deaths']]
df1
df2 = df_municipios.groupby(['city'], sort='True').sum().reset_index()
df2
group_city = df_ce.groupby(["city"])
group_city

pd.DataFrame(group_city.new_deaths.sum())
%matplotlib inline
plt.rcParams["figure.figsize"] = [40, 15]
plt.legend(loc=2, prop={'size': 10})
df2.plot('city','new_deaths', kind='bar',fontsize = 16)