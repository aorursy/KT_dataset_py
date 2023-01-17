# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.figure_factory as ff
import cufflinks
cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.info()
data.describe()
print(data.isnull().sum())
data = data.dropna()
data = data.drop('Unnamed: 0', axis=1)
old_columns = data.columns
new_columns = ['job_title', 'salary', 'description','rating', 'comp_name','location',
              'headquarters','comp_size','founded','ownership_type','industry','sector','revenue',
              'competitors','easy_apply']
data = data.rename(columns=dict(zip(old_columns, new_columns)))
data
data.salary.unique()
data[data.salary=='-1']
data[data.salary=='-1'] = data[data.salary=='-1'].replace('-1', '$0K-$0K (Glassdoor est.)')
salary_min = data.salary.apply(lambda x: x[0:x.index('-')])
salary_min = salary_min.apply(lambda x: x[1:(x.find('K'))])
salary_max = data.salary.apply(lambda x: x[x.find('-')+1:-1])
salary_max = salary_max.apply(lambda x: x[1:(x.find('K'))])
salary_min = pd.to_numeric(salary_min)
salary_max = pd.to_numeric(salary_max)
salary = (pd.DataFrame(salary_max) + pd.DataFrame(salary_min))/2
data['mean_salary'] = salary.salary.apply(lambda x: int(x))
data.comp_name.head()
data['comp_name'] = data['comp_name'].apply(lambda x: x.splitlines()[0])
data.location.unique()
data.headquarters.unique()
data[data.headquarters=='-1']
data = data.replace('-1', 'no information')
print('BEFORE REPLACING')
print('Company size uniques')
print(data.comp_size.unique())
print('--------------------')
print('Revenue uniques')
print(data.revenue.unique())
print('--------------------')
data['comp_size'] = data['comp_size'].replace('Unknown', 'no information')
data['revenue'] = data['revenue'].replace('Unknown / Non-Applicable', 'no information')
print('AFTER REPLACING')
print('Company size uniques')
print(data.comp_size.unique())
print('--------------------')
print('Revenue uniques')
print(data.revenue.unique())
print('Ownership type uniques')
print(data.ownership_type.unique())
print('Industry uniques')
print(data.industry.unique())
print('Sector uniques')
print(data.sector.unique())
print(data.easy_apply.unique())
print('There is {:.2f} % rows with such values in these columns'.format(
    len(data[(data.competitors=='$0K-$0K (Glassdoor est.)')|(data.easy_apply=='no information')|
             (data.easy_apply=='$0K-$0K (Glassdoor est.)')])/len(data)*100))
data = data.drop(['competitors', 'easy_apply'], axis=1)
data = data.replace(-1, 0)
data.head()
px.bar(data.job_title.value_counts().reset_index().head(30), x='index', y='job_title', labels={'index':'job title', 'job_title':'amount of vacancies'},
                                                                                             title = 'Names of vacancies distribution',
      color = 'job_title')
px.bar(data.salary.value_counts().reset_index(), x='index', y='salary', labels={'index':'salary', 'salary':'amount of vacancies'},
                                                                                             title = 'Salary distribution',
      color = 'salary')
salary_df = pd.DataFrame({'minn':salary_min, 'maxx':salary_max, 'meann':data.mean_salary})
salary_df = salary_df[(salary_df.minn!=0)|(salary_df.maxx!=0)|(salary_df.meann!=0)]
fig = go.Figure()
fig.add_trace(go.Box(y=salary_df['minn'].values, name = 'Min salary boxplot'))
fig.add_trace(go.Box(y=salary_df['maxx'].values, name = 'Max salary boxplot'))
fig.add_trace(go.Box(y=salary_df['meann'].values, name = 'Mean salary boxplot'))
px.histogram(data[data.sector!='no information'], x='sector', color='sector', title = 'Amount of vacancies in each sector')
data['state']=data['location'].apply(lambda x: x.split(',')[1])
data['location']=data['location'].apply(lambda x: x.split(',')[0])
px.histogram(data, x='state', color='state', title = 'Amount of vacancies in each state')
v=pd.DataFrame(data.groupby('revenue').mean_salary.value_counts()).rename(columns={'mean_salary':'amount of vacancies'}).reset_index()
v=v[v.revenue!='no information']
dictionary={'Less than $1 million (USD)':1,'$1 to $5 million (USD)':2,
                                                    '$5 to $10 million (USD)':3,'$10 to $25 million (USD)':4,
                                                    '$25 to $50 million (USD)':5,'$50 to $100 million (USD)':6,
                                                    '$100 to $500 million (USD)':7,'$500 million to $1 billion (USD)':8,
                                                    '$1 to $2 billion (USD)':9,'$2 to $5 billion (USD)':10,'$5 to $10 billion (USD)':11,
                                                    '$10+ billion (USD)':12,'no information':0}
v['rang'] = v['revenue'].map(dictionary)
v=v.sort_values('rang')
px.scatter(v[v.mean_salary!=0], x='mean_salary', y='revenue', size='amount of vacancies', color='mean_salary',
          title = 'Amount of vacancies with each salaries in companies grouped by revenue')
v=pd.DataFrame(data.groupby('comp_size').mean_salary.value_counts()).rename(columns={'mean_salary':'amount of vacancies'}).reset_index()
v=v[v.comp_size!='no information']
dictionary={'1 to 50 employees':1, '51 to 200 employees':2, '201 to 500 employees':3,
                                                        '501 to 1000 employees':4,'1001 to 5000 employees':5, '5001 to 10000 employees':6,
                                                         '10000+ employees':7,'no information':0}
v['rang'] = v['comp_size'].map(dictionary)
v=v.sort_values('rang')
px.scatter(v[v.mean_salary!=0], x='mean_salary', y='comp_size', size='amount of vacancies', color='mean_salary',
           title = 'Amount of vacancies with each mean salarie in companies grouped by company size')
y=pd.DataFrame(data.groupby(['industry','sector']).mean_salary.mean().reset_index())
px.scatter(y, x='industry', y='sector', color='mean_salary', size='mean_salary')
px.scatter(data[data.rating!=0], x='state', y='sector', color='rating', hover_data=['comp_name'], size='mean_salary',
          title = 'Vacancy Map')
top = data[(data.rating>4.6)&(data.mean_salary>69)].sort_values(['mean_salary','rating'],ascending=False )
px.histogram(top, x='state', title = 'State distribution for top vacancies', color='state')
px.histogram(top, x='industry', title = 'Industry distribution for top vacancies', color='sector')
px.box(top.mean_salary.values, title = 'Salary description for top vacancies')
junior = data[(data['job_title'].str.contains('junior'))|(data['job_title'].str.contains('Junior'))]
junior.head(10)
px.box(junior, junior.mean_salary.values, hover_data=['comp_name'], title = 'Junior Analysts salary')
px.scatter(junior, x='state', y='sector', color='rating', hover_data=['comp_name'], size='mean_salary',
          title = 'Vacancy Map for Juniors')
px.histogram(junior, x='state', color='mean_salary')
