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
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.columns = data.columns.str.replace(' ', '_')
data.info()
data['Size'].unique()
data['Size']=data['Size'].replace('Unknown', None)

data['Size']=data['Size'].replace('-1', None)

data['Size'].unique()
data['Salary_Estimate']=data['Salary_Estimate'].apply(lambda x:x.split(' ')[0])

data['Salary_Estimate']=data['Salary_Estimate'].str.replace('$','')

data['Salary_Estimate']=data['Salary_Estimate'].str.replace('K','000')
data['Salary_Estimate'].unique()
data['Min_Salary']=data['Salary_Estimate'].apply(lambda x:x.split('-')[0])

data['Max_Salary']=data['Salary_Estimate'].apply(lambda x:x.split('-')[1])
data["Min_Salary"]=data["Min_Salary"].replace('','0')
data["Min_Salary"] = data["Min_Salary"].astype('float64')

data["Max_Salary"] = data["Max_Salary"].astype('float64')
data['Average_Salary']=data[['Min_Salary','Max_Salary']].mean(axis=1)
data.head(50)
data['Location']=data['Location'].apply(lambda x:x.split(',')[1])
data.groupby(['Location']).mean()
data['Size'].unique()
data['Size']=data['Size'].replace('10000+ employees', '10000 to 10000 employees')
data['Min_Size']=data['Size'].apply(lambda x:x.split(' to ')[0])

data['Max_Size']=data['Size'].apply(lambda x:x.split(' to ')[1])

data['Max_Size']=data['Max_Size'].apply(lambda x:x.split(' ')[0])

data["Min_Size"] = data["Min_Size"].astype('int32')

data["Max_Size"] = data["Max_Size"].astype('int32')
data['Average_Size']=data[['Min_Size','Max_Size']].mean(axis=1)
data["Rating"] = data["Rating"].astype('float64')
data['Revenue'].unique()
data['Revenue']=data['Revenue'].replace('Unknown / Non-Applicable', None)

data['Revenue']=data['Revenue'].replace('-1', None)

data['Revenue']=data['Revenue'].replace('$10+ billion (USD)', '$10billion to $10billion (USD)')

data['Revenue']=data['Revenue'].replace('Less than $1 million (USD)', '$0 to $1million (USD)')

data['Revenue']=data['Revenue'].replace('$100 to $500 million (USD)', '$100million to $500million (USD)')

data['Revenue']=data['Revenue'].replace('$2 to $5 billion (USD)', '$2billion to $5billion (USD)')



#data['Revenue']=data['Revenue'].str.replace('$50 to $100 million (USD)', '$50million to $100million (USD)')

#data['Revenue']=data['Revenue'].str.replace('$1 to $2 billion (USD)', '$1billion to $2billion (USD)')



data['Revenue']=data['Revenue'].replace('$5 to $10 billion (USD)', '$5billion to $10billion (USD)')

data['Revenue']=data['Revenue'].replace('$1 to $5 million (USD)', '$1million to $5million (USD)')



data['Revenue']=data['Revenue'].replace('$25 to $50 million (USD)', '$25million to $50million (USD)')

data['Revenue']=data['Revenue'].replace('$10 to $10 billion (USD)', '$10billion to $10billion (USD)')



data['Revenue']=data['Revenue'].replace('$0 to $1 million (USD)', '$0 to $1million (USD)')

data['Revenue']=data['Revenue'].replace('$10 to $25 million (USD)', '$10million to $25million (USD)')



data['Revenue']=data['Revenue'].replace('$500 million to $1 billion (USD)', '$500million to $1billion (USD)')

data['Revenue']=data['Revenue'].replace('$5 to $10 million (USD)', '$5million to $10million (USD)')



data['Revenue']=data['Revenue'].str.replace('million', '000000' )

data['Revenue']=data['Revenue'].str.replace('billion', '000000000')

data['Revenue']=data['Revenue'].str.replace('$','')

data['Revenue']=data['Revenue'].str.replace('(USD)','')



data['Revenue']=data['Revenue'].replace('50 to 100 000000 ()', '50000000 to 100000000 ()')

data['Revenue']=data['Revenue'].replace('1 to 2 000000000 ()', '1000000000 to 2000000000 ()')



data['Min_Revenue']=data['Revenue'].apply(lambda x:x.split(' to ')[0])

data['Max_Revenue']=data['Revenue'].apply(lambda x:x.split(' to ')[1])

data['Max_Revenue']=data['Max_Revenue'].apply(lambda x:x.split(' ')[0])

data['Revenue'].unique()
data["Min_Revenue"] = data["Min_Revenue"].astype('int32')

data["Max_Revenue"] = data["Max_Revenue"].astype('int32')

data['Average_Revenue']=data[['Min_Revenue','Max_Revenue']].mean(axis=1)

data["Average_Revenue"] = data["Average_Revenue"].astype('float64')
data.head()
data.info()
data['Job_Title'] = data['Job_Title'].str.lower()

data['Job_Title']=data['Job_Title'].str.replace('sr.', 'senior' )

data['Job_Title']=data['Job_Title'].str.replace('jr.', 'junior' )
arr1 = data.Job_Title.unique()



for i in arr1:

    print(i)
import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

titles=Counter(data['Job_Title'])

most_common_titles=titles.most_common(5)

x,y= zip(*most_common_titles)

x,y= list(x), list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.hls_palette(len(x)))

plt.xlabel('Most Common Job Titles')
location=Counter(data['Location'])

most_common_location=location.most_common(15)

x,y= zip(*most_common_location)

x,y= list(x), list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.hls_palette(len(x)))

plt.xlabel('Most Common Locations')
ax = sns.scatterplot(x="Average_Salary", y="Average_Size", data=data)
import plotly.express as px

fig = px.pie(data, values='Average_Revenue', names='Sector', title='Average Revenues According to Sector',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
fig = px.pie(data, values='Average_Revenue', names='Type_of_ownership', title='Average Revenues According to Type of Ownership',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
df1= data 

df1= df1.drop(columns = ['Unnamed:_0','Size','Salary_Estimate','Competitors','Revenue', 'Job_Title','Job_Description','Company_Name','Headquarters','Founded','Easy_Apply','Min_Salary','Max_Salary','Min_Size','Max_Size','Min_Revenue','Max_Revenue'])
df1.head()