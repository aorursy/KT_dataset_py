# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from wordcloud import WordCloud



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')

data.head(2)
data.drop(['Unnamed: 0'], axis =1, inplace = True)
data.notnull()
data.replace(to_replace=-1,value=np.nan,inplace= True)

data.replace(to_replace=-1.0,value=np.nan,inplace= True)

data.replace(to_replace='-1',value=np.nan,inplace= True)

data.replace(to_replace='-1.0',value=np.nan,inplace= True)

data.replace(to_replace='',value=np.nan,inplace= True)

data['Rating'].replace(to_replace="nan",value="0",inplace= True)

data['Easy Apply'] = data['Easy Apply'].fillna(False).astype('bool')

data.drop(['Competitors'],axis=1,inplace=True)

data.drop(['Founded'],axis=1,inplace=True)
data_salary = data['Salary Estimate']

data_sal = data_salary.astype(str)
x = []

for i in data_sal:

    min_val = i.split('(')[0].split('-')[0].lstrip('$').rstrip('K')

    x.append(min_val)

data['Min Salary'] = pd.DataFrame(x)



data['Min Salary'].replace(to_replace="nan",value="0",inplace= True)

data['Min Salary'].astype('int')
y = []

for i in data_sal:

    max_val = i.split('(')[0].split('-')[-1].lstrip('$').split("K")[0]

    y.append(max_val)

data['Max Salary'] = pd.DataFrame(y)



data['Max Salary'].replace(to_replace="nan",value="0",inplace= True)

data['Max Salary'].astype('int')
data_company = data['Company Name']

data_comp = data_company.astype(str)



y = []

for i in data_comp:

    company = i.split('\n')[0]

    y.append(company)

data['Company'] = pd.DataFrame(y)



data.drop(['Company Name'],axis=1,inplace=True)
df_easy_apply = data[data['Easy Apply']==True]

df = df_easy_apply.groupby('Company')['Easy Apply'].count().reset_index()

company_opening_df = df.sort_values('Easy Apply',ascending=False)
plt.figure(figsize=(10,5))

chart = sns.barplot(

    data = company_opening_df.head(10),

    x = 'Company',

    y = 'Easy Apply',

    palette = 'Set1'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
data_analyst = data[data['Job Title']=="Data Analyst"]

sns.set(style="white", palette = "muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True)

sns.despine(left=True)

sns.distplot(data_analyst['Min Salary'], color="b", ax=axes[0])

sns.distplot(data_analyst['Max Salary'], color="r",ax=axes[1])



plt.setp(axes, yticks=[])

plt.tight_layout()
job_title = data['Job Title'][~pd.isnull(data['Job Title'])]

wordCloud = WordCloud(width=450,height= 300).generate(' '.join(job_title))

plt.figure(figsize=(19,9))

plt.axis('on')

plt.title(data['Job Title'].name,fontsize=20)

plt.imshow(wordCloud)

plt.show()