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

pd.set_option('display.max_rows', 10000)

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

import matplotlib
df=pd.read_csv('/kaggle/input/covid19indiadata/Covid_India_Data_Consolidated.csv')
df.dtypes
df1=df.fillna(0)
df1.columns=['Confirmed','CountryReg','Deaths','LastUpdate','Latitude','Longitude','Province','Recovered','week']
columns=['Confirmed','Deaths','Recovered']

for cln in columns:

    df1[cln]=df1[cln].astype(int)
final=df1.sort_values('LastUpdate')
final=final[final.CountryReg.isin(['India','Italy','Spain','Germany'])]
final
df1['LastUpdate']=df1.LastUpdate.astype(str)
df12=df1[df1.LastUpdate=='2020-03-20']
df13=df12.groupby(['CountryReg'])['Confirmed','Recovered','Deaths'].sum().sort_values('Confirmed',ascending=False)
df13.reset_index(inplace=True)
df14=df13[0:10]

df15=df13[0:30]
import matplotlib.pyplot as plt

import seaborn as sns

f,ax=plt.subplots(figsize=(20,7))

sns.set(style='whitegrid')

ax=sns.barplot(x='Confirmed',y='CountryReg',data=df14,palette='Greens_r',capsize=.5)

ax.set_xlabel('Total Confirmed Cases')

ax.set_ylabel('Top 10 Countries')

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

f,ax=plt.subplots(figsize=(20,7))

sns.set(style='whitegrid')

ax=sns.lineplot(x='Confirmed',y='week',hue='CountryReg',data=final,legend="full",sizes=(.25, 2.5))

ax.set_xlabel('Total Confirmed Cases')

ax.set_ylabel('Week Numbers')

plt.show()
from IPython.display import HTML

def color_negative_red(value):

  """

  Colors elements in a dateframe

  green if positive and red if

  negative. Does not color NaN

  values.

  """



  if value < 0:

    color = 'red'

  elif value > 0:

    color = 'green'

  else:

    color = 'black'



  return 'color: %s' % color

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', 'Black'),

  ('background-color', '#f7f7f9')

  ]



# Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px')

  ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]

cm = sns.light_palette("green", as_cmap=True)



(df15.style

  .background_gradient(cmap=cm, subset=['Confirmed','Deaths'])

  .highlight_max(subset=['Confirmed','Deaths'])

  .set_caption('This is a custom caption.')

  .set_table_styles(styles))

#df14.style.applymap(color_negative_red, subset=['Confirmed','Deaths'])