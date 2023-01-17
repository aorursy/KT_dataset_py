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
# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go
# upload dataset

data= pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")



# dataset shape

print(data.shape)



data.head()
# check missing values

data.isnull().sum()
data.describe()
# remove Unnamed:0 column

data= data.drop(columns= ["Unnamed: 0"], axis=1)
# separating state name from Location column

new= data["Location"].str.split(", ", expand=True)



# creating new column with the state name

data["state"]= new[1]

data.drop(data.loc[data['state']=="Arapahoe"].index, inplace=True)
# create 2 columns for minimum salary and maximum salary

data["Salary Estimate"]= data["Salary Estimate"].str.rstrip("(Glassdoor est.)")

df1= data["Salary Estimate"].str.split("-", expand=True)



# minimum salary

data["minimum_salary"]= df1[0]

data["minimum_salary"] = data["minimum_salary"].str.lstrip("$")

data["minimum_salary"] = data["minimum_salary"].str.rstrip("K")

# data["minimum_salary ($)"] = data["minimum_salary ($)"].astype("str").astype("int")

data["minimum_salary"] = pd.to_numeric(data["minimum_salary"])



# maximum salary

data["maximum_salary"]= df1[1]

data["maximum_salary"] = data["maximum_salary"].str.lstrip("$")

data["maximum_salary"] = data["maximum_salary"].str.rstrip("K")

data["maximum_salary"]= pd.to_numeric(data["maximum_salary"])
from wordcloud import WordCloud, STOPWORDS



text= ', '.join(data['Job Description'])



stopwords = set(STOPWORDS)



wordcloud = WordCloud( background_color="black").generate(text)





plt.figure(figsize=[15,20])

plt.title("skills a data analyst job demands",size= 30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
job_title=data["Job Title"].value_counts().nlargest(10)

job_title_df= pd.DataFrame(job_title)

job_title_df



fig = px.bar(job_title_df, x="Job Title", y=job_title_df.index, orientation='h',

             color='Job Title',

             labels= {"job_title_df.index": "job title",

                       "Job Title": "no. of jobs"},

             title= "Number of jobs for job titles",

             height=500)

fig.update_layout(title_x=0.5)

fig.show()
series= data["state"].value_counts()

df1= pd.DataFrame(series).reset_index().rename(columns={"index": "state", "state": "count"})

df1



fig = px.treemap(df1, path=['state'], values='count',color='count',color_discrete_sequence = px.colors.qualitative.Set1,

                title=('States with Number of Jobs'))

fig.update_layout(title_x=0.5)

fig.show()
df= data.groupby("state")["Location"].value_counts()

df_= pd.DataFrame(df).rename(columns={"Location": "count"}).reset_index()

df_

fig = px.treemap(df_, path=['state', 'Location'], values='count',color='count', 

                 color_continuous_scale='mint',

                title=('States and Cities with Number of Jobs'))

fig.update_layout(title_x=0.5)

                

fig.show()

#  )
salary = data.groupby('Location')[['minimum_salary','maximum_salary']].mean().sort_values(['minimum_salary','maximum_salary'],

                                                                                          ascending=False).head(20)



fig = go.Figure()

fig.add_trace(go.Bar(x= salary.index, y= salary['minimum_salary'], name= 'Min Salary', marker= dict(color= 'pink')))

fig.add_trace(go.Bar(x= salary.index, y= salary['maximum_salary'], name= 'Max Salary', marker= dict(color= 'grey')))

fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=12)

fig.update_layout(title= 'Job Location with their Salary Range', barmode= 'stack',title_x=0.5)



fig.show()
data["Sector"].unique()
# replacing -1 to "unknown" in Sector

data["Sector"].replace({"-1": "unknown"}, inplace=True)
sector= data['Sector'].value_counts().nlargest(n=10)

sector_df= pd.DataFrame(sector).reset_index().rename(columns= {"index":"sector", "Sector":"job count"})



fig = px.pie(sector_df, 

       values = "job count", 

       names = "sector",

       labels= "sector",       

       title="Top 10 Sectors with number of jobs", 

       color=sector.values,

       color_discrete_sequence=px.colors.qualitative.Prism)



fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, textposition='inside', textinfo='percent+label')



fig.update_layout(title_x=0.45)

fig.show()
salary = data.groupby('Sector')[['minimum_salary','maximum_salary']].mean().sort_values(['minimum_salary','maximum_salary'],

                                                                                          ascending=False)



fig = go.Figure()

fig.add_trace(go.Bar(x = salary.index, y = salary['minimum_salary'], name = 'Min Salary', marker = dict(color = 'pink')))

fig.add_trace(go.Bar(x = salary.index, y = salary['maximum_salary'], name = 'Max Salary', marker = dict(color = 'grey')))

fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=12)

fig.update_layout(title = 'Sectors with their Salary Range', barmode = 'stack',title_x=0.5)



fig.show()
sns.pairplot(data[["Rating", "minimum_salary", "maximum_salary"]])
# replacing -1 to "unknown" in Size

data["Size"].replace({"-1": "Unknown"}, inplace=True)

data["Size"].unique()
# removing data for which Employee Size is not given(unknown)

df= data.drop(data.loc[data['Size']=="Unknown"].index)



df_size=df.groupby("Size")["Company Name"].count()

df_size



fig = px.bar(y=df_size.index,

       x=df_size.values,

       orientation='h',

       color=df_size.index,

       text=df_size.values,

       color_discrete_sequence= px.colors.qualitative.Bold)



fig.update_traces(texttemplate='%{text:.2s}', 

                  textposition='outside', 

                  marker_line_color='rgb(8,48,107)', 

                  marker_line_width=1.5, 

                  opacity=0.7)



fig.update_layout(width=800, 

                  showlegend=False, 

                  xaxis_title="No. of companies",

                  yaxis_title="Company Size",

                  title="Company Size and their Company count", title_x=0.5)

fig.show()
series2= data.groupby("Size")["Company Name"].value_counts()

df2= pd.DataFrame(series2).rename(columns={"Company Name": "count"}).reset_index()



# removing data for which Employee Size is not given(unknown)

df2.drop(df2.loc[df2['Size']=="Unknown"].index, inplace=True)



# removing data for which the number of job is only 1

df2.drop(df2.loc[df2['count'] ==1].index, inplace=True)

df2



fig = px.treemap(df2, path=['Size', 'Company Name'], values='count',color='count', 

                 color_discrete_sequence = px.colors.qualitative.Set1,

                title=('Companies with their Size and Job Counts'))

fig.update_layout(title_x=0.5)

                

fig.show()