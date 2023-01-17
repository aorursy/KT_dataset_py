# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go 
import cufflinks as cf

# py.offline().init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_df = pd.read_csv('../input/BlackFriday.csv')
data_df.describe()
data_df.head(10)

data_df = data_df.drop(['User_ID', 'Product_ID'], axis =1)
# data without User_ID and Product_ID column
# data_df.describe()

data_df.head(10)
pd_data = data_df.copy()
pd_data['Gender'] = pd_data['Gender'].map({'F':0, 'M':1}).astype(int)
# pd_data
pd_data.head()
pd_data['City_Category'].value_counts()
pd_data['City_Category'] = pd_data['City_Category'].map({'A':0, 'B':1, 'C':2}).astype(int)
pd_data.head()
pd_data['Age'].value_counts()
age_ranges = pd_data['Age'].astype('category').cat.categories.tolist()
age_ranges
replace_map_comp = {'Age' : {k: v for k,v in zip(age_ranges,list(range(1,len(age_ranges)+1)))}}
replace_map_comp
pd_data.replace(replace_map_comp, inplace= True)
pd_data.head()
pd_data.isnull().sum()
pd_data = pd_data.fillna(0)
# pd_data.describe()
pd_data.head()
sns.set(rc={'figure.figsize':(8,8)})
corr = pd_data.corr()
ax = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
ax.set_title('Heat map of the columns and ther relationships to each other')

pd_data.head(10)
df_gend = pd_data[['Gender', 'Purchase']]
df_gend.head(10)
df_gend_mean = df_gend.groupby('Gender')['Purchase'].mean().reset_index()
df_gend_mean['Gender'] = df_gend_mean['Gender'].map({0:'Female', 1:'Male'}) 
df_gend_mean
sns.set(rc={'figure.figsize':(6,6)},style= 'darkgrid')
ax = sns.barplot(x="Gender", y="Purchase", data = df_gend_mean)
ax.legend(loc = 'upper left')
rs_df = pd_data[['Gender','Marital_Status', 'Purchase']]
rs_df.head()
rs_f = rs_df.loc[rs_df['Gender']==0]
rs_f.head(10)
rs_f_mean = rs_f.groupby('Marital_Status')['Purchase'].mean().reset_index()
rs_f_mean
labels_ =('Single', 'Married')
y_pos = np.arange(len(labels_))
fig = plt.figure(figsize=(6,6))
plt.bar(rs_f_mean['Marital_Status'], rs_f_mean['Purchase'], align = 'center', alpha = 0.7)
plt.xticks(y_pos,labels_)
plt.ylabel('Purchase')
plt.xlabel('Relationship Status')

plt.show()
rs_m = rs_df.loc[rs_df['Gender']==1]
rs_m.head()
rs_m_mean = rs_m.groupby('Marital_Status')['Purchase'].mean().reset_index()
rs_m_mean
sns.set(rc={'figure.figsize':(6,6)},style= 'darkgrid')
ax = sns.barplot(x="Marital_Status", y="Purchase", data = rs_m_mean)
ax.legend(loc = 'upper left')
pd_data.head()
city_df = pd_data[['City_Category', 'Purchase']]
city_df.head()
city_mean = city_df.groupby('City_Category')['Purchase'].mean().reset_index()
city_mean
city_mean['City_Category'] = city_mean['City_Category'].map({0:'A',1:'B', 2:'C'})
city_mean
fig= plt.figure()
sns.set(rc= {'figure.figsize':(6,6)},style = 'darkgrid')
ax = sns.barplot(x= 'City_Category', y = 'Purchase', data = city_mean)
plt.title('Average amount of money spent by each city')
plt.show(fig)