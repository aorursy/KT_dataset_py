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
# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style="darkgrid")
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots

# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

# Counter
from collections import Counter
# Importing Projects.csv
Projects_DF = pd.read_csv('/kaggle/input/io/Projects.csv')
Projects_DF.dropna(subset=['Project Subject Category Tree','Project Title','Project Essay','Project Short Description','Project Need Statement','Project Resource Category'],inplace = True)
#Projects_DF.isnull().sum()

# Importing Donors.csv
#Donors_DF = pd.read_csv("/kaggle/input/io/Donors.csv")

# Importing Donations.csv
#Donations_DF = pd.read_csv('/kaggle/input/io/Donations.csv')
Projects_DF.head(5)
project_type_count = pd.DataFrame(Projects_DF.groupby('Project Type')['Project ID'].count())
project_type_count.columns = ['Count']

project_type_pie = go.Pie(labels=project_type_count.index,values=project_type_count['Count'],name="Project Type",hole=0.4,domain={'x': [0,0.46]})


layout = dict(title = 'Project Types', font=dict(size=10), legend=dict(orientation="h"))
fig = dict(data=[project_type_pie], layout=layout)
py.iplot(fig)
# Extract Project Subject Category Tree

categories = Counter()
def extract_category(category):
        cat = category.split(',')
        for c in cat:
            categories[c.strip()]+=1

Projects_DF['Project Subject Category Tree'].apply(extract_category)
project_categories_count = pd.DataFrame.from_dict(categories,orient = 'index')
project_categories_count.columns = ['Count']

project_category_pie = go.Pie(labels=project_categories_count.index,values=project_categories_count['Count'],name="Project Category",hole=0.4,domain={'x': [0,0.46]})


layout = dict(title = 'Project Categories', font=dict(size=10), legend=dict(orientation="h"))
fig = dict(data=[project_category_pie], layout=layout)
py.iplot(fig)
# Extract Project Subject Subcategory Tree

sub_categories = Counter()
def extract_sub_category(sub_category):
        sub_cat = sub_category.split(',')
        for c in sub_cat:
            sub_categories[c.strip()]+=1

Projects_DF['Project Subject Subcategory Tree'].apply(extract_sub_category)
#print(sub_categories)
project_sub_categories_count = pd.DataFrame.from_dict(sub_categories,orient = 'index')
project_sub_categories_count.columns = ['Count']
project_sub_categories_count.sort_values(by = ['Count'],inplace = True,ascending = False)
plt.figure(figsize=(20,10))
sns.barplot(x = 'Count', y = project_sub_categories_count.index,data = project_sub_categories_count).set_title('Project Subject Subcategories')
project_resource_count = pd.DataFrame(Projects_DF.groupby(['Project Resource Category'])['Project ID'].count())
project_resource_count.columns = ['Count']
project_resource_count.sort_values(by=['Count'], inplace = True,ascending = False)
plt.figure(figsize=(15,10))
sns.barplot(x = 'Count',y = project_resource_count.index,data = project_resource_count)
#Top_5_resources = Projects_DF[Projects_DF['Project Resource Category'].isin('')]
project_resource_count = pd.DataFrame(Projects_DF.groupby(['Project Resource Category','Project Grade Level Category'])['Project ID'].count())
project_resource_count.columns = ['Count']
project_resource_count.sort_values(by=['Count'], inplace = True,ascending = False)
project_resource_count.reset_index(inplace = True)
project_resource_count = project_resource_count[project_resource_count['Project Resource Category'].isin(['Supplies','Technology','Books','Other','Computers & Tablets'])]
plt.figure(figsize=(20,10))
sns.barplot(x = 'Count',y = 'Project Resource Category',hue = 'Project Grade Level Category',data = project_resource_count)
posted_date = pd.DataFrame(Projects_DF['Project Posted Date'].value_counts())
posted_date.columns = ['Count']
posted_date['Day of Week'] = pd.DatetimeIndex(posted_date.index).dayofweek
posted_date['Month'] = pd.DatetimeIndex(posted_date.index).month
posted_date['Year'] = pd.DatetimeIndex(posted_date.index).year

posted_date_month_df = pd.DataFrame(posted_date.groupby('Month')['Count'].sum())
posted_date_month_df.columns = ['Count']


funded_date = pd.DataFrame(Projects_DF['Project Fully Funded Date'].value_counts())
funded_date.columns = ['Count']
funded_date['Day of Week'] = pd.DatetimeIndex(funded_date.index).dayofweek
funded_date['Month'] = pd.DatetimeIndex(funded_date.index).month
funded_date['Year'] = pd.DatetimeIndex(funded_date.index).year

funded_date_month_df = pd.DataFrame(funded_date.groupby('Month')['Count'].sum())
funded_date_month_df.columns = ['Count']


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
labels = ['January','February','March','April','May','June','July','August','September','October','November','December']
fig.add_trace(go.Pie(labels=labels, values=posted_date_month_df['Count'], name="Month Posted"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=funded_date_month_df['Count'], name="Month Funded"),
              1, 2)
fig.update_traces(hole=.4)
fig.update_layout(
    title_text="Project Posted vs Project Funded: Monthy breakdown")
fig.show()
posted_date_dow_df = pd.DataFrame(posted_date.groupby('Day of Week')['Count'].sum())
posted_date_dow_df.columns = ['Count']

funded_date_dow_df = pd.DataFrame(funded_date.groupby('Day of Week')['Count'].sum())
funded_date_dow_df.columns = ['Count']


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig.add_trace(go.Pie(labels=labels, values=posted_date_dow_df['Count'], name="Day Posted"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=funded_date_dow_df['Count'], name="Day Funded"),
              1, 2)
fig.update_traces(hole=.7)
fig.update_layout(
    title_text="Project Posted vs Project Funded: Daily breakdown")
fig.show()
project_status_df = pd.DataFrame(Projects_DF['Project Current Status'].value_counts())
project_status_df.columns = ['Count']
fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
fig.update_layout(
    title_text="Project Current Status")
fig.add_trace(go.Pie(labels=project_status_df.index, values=project_status_df['Count'], name="Project Status"),
              1, 1)
fig.show()
fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))
funded_count = funded_date.groupby(['Year','Month'])['Count'].sum()
posted_count = posted_date.groupby(['Year','Month'])['Count'].sum()
posted_count.plot(legend=True,ax=axis1,marker='o',title="Projects Posted")
funded_count.plot(legend=True,ax=axis2,marker='o',title="Projects Funded")


posted_count.sort_values(ascending = False)
funded_amount = pd.DataFrame(Projects_DF.groupby('Project Fully Funded Date')['Project Cost'].sum())
funded_amount.columns = ['Cost']
funded_amount['Day of Week'] = pd.DatetimeIndex(funded_amount.index).dayofweek
funded_amount['Month'] = pd.DatetimeIndex(funded_amount.index).month
funded_amount['Year'] = pd.DatetimeIndex(funded_amount.index).year
funded_amount_sum = pd.DataFrame(funded_amount.groupby(['Year','Month'])['Cost'].sum())
funded_amount_sum.reset_index(inplace = True)
funded_amount_sum = funded_amount_sum[funded_amount_sum['Year'].isin(['2015','2016','2017','2018'])]
funded_amount_sum['Month'] = funded_amount_sum['Month'].map({1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'})

plt.figure(figsize=(20,10))
sns.barplot(x = 'Year',y = 'Cost',hue = 'Month',data = funded_amount_sum)
funded_amount_sum.sort_values(by=['Cost'],ascending = False).head(3)
Schools_DF = pd.read_csv("/kaggle/input/io/Schools.csv")
project_school_df = pd.DataFrame(Projects_DF['School ID'].value_counts())
project_school_df.reset_index(inplace = True)
project_school_df.columns = ['School ID','Count']
project_school_df = project_school_df.head(10)
project_school_df = project_school_df.merge(Schools_DF[['School ID','School Name']], on = ['School ID'],how = 'left')
project_school_df
plt.figure(figsize=(20,10))
sns.barplot(x = 'Count',y = 'School Name', data = project_school_df)