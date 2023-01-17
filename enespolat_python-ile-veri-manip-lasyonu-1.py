# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/flight.csv')
print(df.columns)
df = df.drop(['tailnum','dep_time','sched_dep_time','dep_delay'],axis=1)
print('Data Types:')
print(df.dtypes.value_counts())
print()
print(df.info())
df.head()
plt.figure(figsize=(8,8))
sns.heatmap(pd.isnull(df.T), cbar=False)

pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
cols = ['arr_time', 'arr_delay']
df[cols] = df[cols].fillna(df[cols].median())
df['distance(miles)'] = round(df.distance.apply(lambda x: x*0.62137),2)
df.head()
mylist1 = np.unique(df.dest.values)
sub = 'i'
print ("\n".join(s for s in mylist1 if sub.lower() in s.lower()))
print()
mylist2 = [idx for idx, val in enumerate(mylist1) if sub.lower() in val.lower()]
print('Index')
print(mylist2)
mylist3 = [val for idx, val in enumerate(mylist1) if sub.lower() in val.lower()]
print()
print('Values')
print(mylist3)
mylist1= np.unique([str(i) for i in df.flight.values])
sub = '151'
print ("\n".join(s for s in mylist1 if sub in s))
print()
mylist2 = [idx for idx, val in enumerate(mylist1) if sub in val]
print('Index')
print(mylist2)
mylist3 = [val for idx, val in enumerate(mylist1) if sub in val]
print()
print('Values')
print(mylist3)
df['time_hour'] = pd.to_datetime(df['time_hour'])
df['date'] = df['time_hour'].map(lambda x: x.strftime('%Y-%m-%d'))
df['month'] = df['time_hour'].dt.month
df.head()
pd.crosstab(df['origin'], df['month'])
pd.pivot_table(df, values='arr_delay', index=['origin'],
                  columns=['month'], aggfunc=np.mean)
df['origin'].replace({'EWR':'ewr', 'JFK':'jfk', 'LGA':'lga'}, inplace=True)
df.loc[df.origin=='ewr', ['dest']] ='BAY'
df['flight'] = df['flight'].astype('float')
df.head()
print(df.groupby('origin')['dest'].nunique().to_frame())
df['sum_arr_delay'] = df.groupby('carrier')['arr_delay'].transform(pd.Series.cumsum)
df[df.carrier=='UA'].head(10)
df1 = df[~np.isnan(df['arr_delay'])]
df2 = df.loc[:,~df.columns.duplicated()]
df3 = df.drop_duplicates(subset=['origin'], keep='first')
df4 = pd.concat([df, pd.get_dummies(df['origin'])], axis=1)
df5 = pd.concat([df, pd.get_dummies(df.select_dtypes(include='object'))], axis=1)
df = df.drop('sum_arr_delay', axis = 1)
df = df.iloc[50:]
df.head()
#df.columns
df['delay_total'] = df['arr_time'] + df['arr_delay']
df = df.append(df.iloc[:6])
df.tail(6)
pd.concat([df[df['distance']>2000], df[df.arr_delay<10]],axis =0,
          ignore_index =True).head(6) 
pd.concat([df.arr_time, df.arr_delay],axis =1).head(6)
df[(df.arr_delay >50) & (df.arr_delay<100) & ((df.flight == 1545) | (df.flight == 1141))].head(6)
from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')