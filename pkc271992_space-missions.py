# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.style.use('ggplot')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
pd.set_option('display.max_columns',85)
pd.set_option('display.max_rows',85)
df.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
df['Datum'] = pd.to_datetime(df['Datum'])
df['Date'] = df['Datum'].apply(lambda x:x.date())
df['Date'] = pd.to_datetime(df['Date'])
year_all = df['Date'].apply(lambda x: x.year).astype(str)
df.set_index('Date',inplace=True)
#df.loc[106,'Datum'] = 'Thu Aug 29, 2019 14:00 UTC'
#df['Datum'] = pd.to_datetime(df['Datum'],format='%a %b %d, %Y')
#df['Date'] = df['Datum'].apply(lambda x: pd.to_datetime(x))
year_list = year_all.unique()
year_list.sort()
#df.set_index('Datum',inplace=True)
df.sort_index(inplace=True)
#gr = df.groupby('Company Name')
#gr['Status Mission'].apply(lambda x: x.str.contains('Prelau').sum())



plt.figure(figsize=(15,10))
ax = sns.countplot(df['Company Name'])
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+50))
plt.show()
comp_lst = (','.join(df['Company Name'])).split(',')
company_cnt = Counter(comp_lst)
company_dict = company_cnt.most_common(10)
plt.figure(figsize=(30,15))
for k,v in company_dict:
    r_cnt = []
    df_temp = df[(df['Company Name'] == k)]
    for y in year_list:
        c = df_temp[:y]['Company Name'].count()
        r_cnt.append(c)
    plt.plot(year_list,r_cnt,label=k,linewidth=5)
    plt.legend()
    plt.xticks(rotation=45,fontsize=17)
    plt.xlabel('Year',fontsize=30)
    plt.ylabel('Number of total launches',fontsize=30)
    plt.title('Satellite Launches')
plt.show()
#df.head(50)
        
plt.figure(figsize=(20,10))
company_list = []
for k,v in company_dict:
    company_list.append(k)
company_list.append('ISRO')
x_axis = np.arange(0,len(company_list))
x_axis
success_cnt     = []
f_cnt           = []
partial_f_cnt   = []
prelaunch_f_cnt = []
for cmp in company_list:
    df_temp = df[(df['Company Name'] == cmp)]
    success_cnt.append(df_temp['Status Mission'].str.contains('Success').sum())
    f_cnt.append(df_temp['Status Mission'].str.contains('Failure').sum())
    partial_f_cnt.append(df_temp['Status Mission'].str.contains('Partial Failure').sum())
    prelaunch_f_cnt.append(df_temp['Status Mission'].str.contains('Prelaunch Failure').sum())
width = 1
plt.bar(x_axis-width/2,success_cnt,width/4,color='g',label='Success')
plt.bar(x_axis-width/4,f_cnt,width/4,color='r',label='Failure')
plt.bar(x_axis,partial_f_cnt,width/4,color='b',label='Partial Failure')
plt.bar(x_axis+width/4,prelaunch_f_cnt,width/4,color='k',label='Prelaunch Failure')
plt.xticks(x_axis,company_list,rotation=90)
plt.xlabel('Companies')
plt.ylabel('Count')
plt.yscale('log')
plt.title('Final result of launch')
plt.legend()
plt.savefig('/kaggle/working/pic.png')
plt.show()
df.head(50)
plt.figure(figsize=(30,30))
df['Status Rocket'].value_counts()
x = df['Datum'].apply(lambda x : x.year)
sns.swarmplot(x='Company Name',y=x,data=df,hue='Status Rocket')
sns.set_style('darkgrid')   
plt.rc('xtick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.xticks(rotation = 90)
plt.show()

