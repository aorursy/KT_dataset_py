# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')
data.head()
data.info()
data.describe()
data.columns
data= data.rename(columns={'Cumulative number of case(s)':'cases_cum', \
                        'Number of deaths': 'Deaths',\
                       'Number recovered': 'Recovered'})
data.head()
data.Date=pd.to_datetime(data.Date)
data.Country.unique()
data.Country.value_counts()
countries=data.groupby('Country').Deaths.count().reset_index(name='Count').sort_values(by='Count',ascending=False)
sns.set(style='whitegrid')
plt.figure(figsize=(19,9))
a=sns.barplot(x='Country',y='Count',data=countries,palette='mako')
plt.xticks(rotation=90,size=12)
plt.title('Apperance Off Countries',size=30)
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
cases=data.groupby('Country',as_index=False).cases_cum.max().sort_values(by='cases_cum',ascending=False)
cases['percent']=round((cases.cases_cum/cases.cases_cum.sum())*100,2)
cases
values=list(cases.cases_cum.head(6))
labels=list(cases.Country.head(6))
values.append(cases[cases.cases_cum<206]['cases_cum'].sum())
labels.append('Other Countries')
a=plt.pie(values,labels=labels,radius=3,autopct='%0.2f%%',shadow=True,explode=[0,0,0.5,0,0,0,0])

data['week']=data.Date.dt.week
week_cases=data.groupby(['Country','week']).cases_cum.max().reset_index(name='week_cases')
week_deaths=data.groupby(['Country','week']).Deaths.max().reset_index(name='week_deaths')
week_recovered=data.groupby(['Country','week']).Recovered.max().reset_index(name='week_recovered')
## country_stats is a function that make us visualize the evolution of cases,deaths and recovered for each country over the weeks
def country_stats(country):
    plt.figure(figsize=(12,9))
    a=sns.lineplot(x='week',y='week_cases',data=week_cases[week_cases.Country==country],marker=True,ci=20,label='Cases',color='c')
    sns.lineplot(x='week',y='week_recovered',data=week_recovered[week_recovered.Country==country],marker=True,ci=20,ax=a,label='Recovered',color='g')
    sns.lineplot(x='week',y='week_deaths',data=week_deaths[week_deaths.Country==country],marker=True,ci=20,ax=a,label='Death',color='r')
    plt.xlabel('Weeks',fontsize=20)
    plt.ylabel('Count',fontsize=20)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=15)
country_stats('China')
country_stats('Hong Kong SAR, China')
country_stats('Taiwan, China')
country_stats('Canada')
country_stats('United States')
country_stats('Singapore')
