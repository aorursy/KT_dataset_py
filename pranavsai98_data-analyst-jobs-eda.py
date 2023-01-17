import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from wordcloud import WordCloud

import string

import nltk

from nltk.corpus import stopwords

!pip install chart-studio

import chart_studio.plotly as py 

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from collections import defaultdict

import warnings

warnings.filterwarnings("ignore")

nltk.download('stopwords')

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# reading .csv file

df=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

df.shape
df.head()
df.describe(include='all')
def miss_val(df):

    #number of missing values

    miss_val = df.isnull().sum()

    

    #percent missing values

    miss_val_per = miss_val/ len(df)*100

    

    #creating df with the results

    miss_val_df = pd.concat([miss_val, miss_val_per], axis=1)

    

    #renaming the columns

    miss_val_df = miss_val_df.rename(columns = {0 : 'missing_values', 1 : '%_of_total_values'})

    

    #sorting the table

    miss_val_df=miss_val_df[miss_val_df['missing_values']!=0]

    miss_val_df = miss_val_df.sort_values('%_of_total_values', ascending=False)

    return miss_val_df
miss_val(df)
df['Job Title'].value_counts()
#1 is given as max split i.e split only once

df['Job Title'],df['Department']=df['Job Title'].str.split(',', 1).str

df.head()
df['Industry'].value_counts().head(10)
df['Sector'].value_counts().head(10)
df['Competitors'].value_counts().head(10)
df['Easy Apply'].value_counts().head(10)
df['Revenue'].value_counts().head(10)
num_lst=[-1,-1.0,'-1']

for num in num_lst:

    df=df.replace(num,np.nan)
df['Salary Estimate'].value_counts().head(10)
df['Salary Estimate'],_=df['Salary Estimate'].str.split('(', 1).str

df['salary_min'],df['salary_max']=df['Salary Estimate'].str.split('-').str

df['salary_max']=df['salary_max'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')

df['salary_min']=df['salary_min'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')

df.drop('Salary Estimate',axis=1,inplace=True)
df['Size'].value_counts()
df['Size'],_=df['Size'].str.split(" e").str
df['Company Name'].value_counts().head(5)
df['Company Name'],_=df['Company Name'].str.split('\n', 1).str
df['City'],df['State']=df['Location'].str.split(', ', 1).str

df['State']=df['State'].replace("Arapahoe, CO","CO")

df.drop('Location',axis=1,inplace=True)

df.head()
df['Size'].fillna('Unknown',inplace=True)



df['Founded']=df['Founded'].fillna("0").astype(int)



df['Type of ownership'].fillna('Unknown',inplace=True)



df['Sector'].fillna('Unknown',inplace=True)



df['Easy Apply']=df['Easy Apply'].fillna("False").astype("bool")
def filter_revenue(x):

    revenue=0

    if(x== 'Unknown / Non-Applicable' or type(x)==float):

        revenue=0

    elif(('million' in x) and ('billion' not in x)):

        maxRev = x.replace('(USD)','').replace("million",'').replace('$','').strip().split('to')

        if('Less than' in maxRev[0]):

            revenue = float(maxRev[0].replace('Less than','').strip())

        else:

            if(len(maxRev)==2):

                revenue = float(maxRev[1])

            elif(len(maxRev)<2):

                revenue = float(maxRev[0])

    elif(('billion'in x)):

        maxRev = x.replace('(USD)','').replace("billion",'').replace('$','').strip().split('to')

        if('+' in maxRev[0]):

            revenue = float(maxRev[0].replace('+','').strip())*1000

        else:

            if(len(maxRev)==2):

                revenue = float(maxRev[1])*1000

            elif(len(maxRev)<2):

                revenue = float(maxRev[0])*1000

    return revenue
df['Max_revenue']=df['Revenue'].apply(lambda x: filter_revenue(x))

df.head()
job_title=df['Job Title'][~pd.isnull(df['Job Title'])]

wordCloud = WordCloud(background_color='white',width=500,height= 200).generate(' '.join(job_title))

plt.figure(figsize=(20,7))

plt.axis('off')

plt.title(df['Job Title'].name,fontsize=20)

plt.imshow(wordCloud)

plt.show()
# Removing stopwords

def removing_stopwords(text):

   #removing some important stopwords from stopwords

    Stopwords=set(stopwords.words('english'))

    return " ".join([word for word in str(text).split() if word not in Stopwords])
#removing stopwords

df['Job Description']=df['Job Description'].apply(lambda text:removing_stopwords(text))
Job_Description=df['Job Description'][~pd.isnull(df['Job Description'])]

wordCloud = WordCloud(background_color='white',width=500,height= 200).generate(' '.join(Job_Description))

plt.figure(figsize=(20,7))

plt.axis('off')

plt.title(df['Job Description'].name,fontsize=20)

plt.imshow(wordCloud)

plt.show()
pg_lan = ["python","c++","java","matlab",".net","c#","javascript","html","bash"]

big_data = ["big data","hadoop","spark","impala","cassandra","kafka","hdfs","hbase","hive"]

job = df["Job Description"].tolist()

job = [x.lower() for x in job]
pg_lan_required = defaultdict()

for item in pg_lan:

    counter = 0

    for it in job:

        if item in it:

            counter = counter + 1

    pg_lan_required[item] = counter



pg_lan_df = pd.DataFrame(list(pg_lan_required.items()),columns = ['Programming Langauge','count']) 

pg_lan_df.sort_values(["count"], axis=0, ascending=False, inplace=True)
plt.figure(figsize = (20,7))

x = pg_lan_df["Programming Langauge"]

y = pg_lan_df["count"]

plt.bar(x,y,color= "#4090c5")

plt.title("Top programming languages required",fontsize=17)

plt.xlabel("Programming Languages",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
counter = 0

big_data_required = defaultdict()

for item in big_data:

    counter = 0

    for it in job:

        if item in it:

            counter = counter + 1

    big_data_required[item] = counter



big_data_df = pd.DataFrame(list(big_data_required.items()),columns = ['Big Data Technologies','count']) 

big_data_df.sort_values(["count"], axis=0, ascending=False, inplace=True)
plt.figure(figsize = (20,7))

x = big_data_df["Big Data Technologies"]

y = big_data_df["count"]

plt.bar(x,y,color= "#4090c5")

plt.title("Top Big Data Technologies requrired ",fontsize=17)

plt.xlabel("Skills",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
plt.figure(figsize=(20,7))

sns.countplot(df['Rating'],color='#4090c5')

plt.xlabel("Rating",fontsize=15)

plt.ylabel("# of companies",fontsize=15)

plt.title("Rating v/s number of companies",fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
statewise_count=pd.DataFrame(df.groupby('State').count().iloc[:,1]).reset_index()





data = dict(type='choropleth',

            locations = statewise_count['State'],

            locationmode = 'USA-states',

            colorscale='blues',

            z = statewise_count['Job Description'],

            colorbar = {'title':"number of jobs"}

            )



layout = dict(title = 'Data Analyst jobs per state',geo = dict(scope='usa'))



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
salary_df=df.copy()

salary_df['mean_salary']=(df['salary_max']+df['salary_min'])/2

state_salary=pd.DataFrame(salary_df.groupby('State')['mean_salary'].mean()).reset_index()

state_salary



data = dict(type='choropleth',

            locations = state_salary['State'],

            locationmode = 'USA-states',

            colorscale = 'blues',

            z = state_salary['mean_salary'],

            colorbar = {'title':"salary"}

            )



layout = dict(title = 'mean salary based on state',geo = dict(scope='usa'))



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
plt.figure(figsize=(20,7))

sns.distplot(salary_df.loc[salary_df['mean_salary']!=0,'mean_salary'],color="#4090c5")

plt.xlabel('mean_salary',fontsize=15)

plt.ylabel('probability density',fontsize=15)

plt.title('distribution of mean salary',fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
temp_df=salary_df[salary_df['Max_revenue']!=0]

size_revenue=pd.DataFrame(temp_df.groupby('Size')['mean_salary'].mean().sort_values()).reset_index()
plt.figure(figsize=(20,7))

order=['Unknown','51 to 200','201 to 500','501 to 1000','1001 to 5000','5001 to 10000','10000+']

sns.barplot(x='Size',y='mean_salary',data=size_revenue,order=order,color='#4090c5')

plt.title("company size v/s salary",fontsize=17)

plt.ylabel('mean salary in thousands of $',fontsize=15)

plt.xlabel('Size of the Company',fontsize=15)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
ownership_df=pd.DataFrame(temp_df['Type of ownership'].value_counts()).reset_index()

ownership_df.rename(columns={'index':'Type of ownership','Type of ownership':'value_counts'},inplace=True)

ownership_salary=pd.DataFrame(temp_df.groupby('Type of ownership')['mean_salary'].mean()).reset_index()

ownership_df=pd.merge(ownership_df,ownership_salary,how='left',left_on="Type of ownership",right_on='Type of ownership')
plt.figure(figsize=(20,7))

plt.xticks(rotation=60,horizontalalignment='right')

sns.barplot(x=ownership_df['Type of ownership'],y=ownership_df['mean_salary'],color='#4090c5')

plt.title('mean salary for different types of ownership', fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.xlabel('Type of ownership',fontsize=15)

plt.ylabel('mean salary',fontsize=15)
ownership_df_2=ownership_df.set_index('Type of ownership')

fig = plt.figure(figsize=(20,7)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



width = 0.4



ownership_df_2['mean_salary'].plot(kind='bar', color='#ee854a', ax=ax, width=width, position=1)

ownership_df_2['value_counts'].plot(kind='bar', color='#4090c5', ax=ax2, width=width, position=0)



ax.set_ylabel('mean salary in thousands of $',fontsize=15)

ax.legend(['mean_salary'],bbox_to_anchor=(0.45, 1))

ax2.set_ylabel('number of companies',fontsize=15)

ax2.legend(['number of companies'],bbox_to_anchor=(0.65, 1))

plt.title("Type of ownership v/s mean salary v/s number of companies",fontsize=17)

plt.xlabel('Type of ownership',fontsize=15)

plt.yticks(fontsize=13)
sector_df=pd.DataFrame(temp_df['Sector'].value_counts()).reset_index()

sector_df.rename(columns={'index':'Sector','Sector':'value_counts'},inplace=True)

sector_salary=pd.DataFrame(temp_df.groupby('Sector')['mean_salary'].mean()).reset_index()

sector_df=pd.merge(sector_df,sector_salary,how='left',left_on="Sector",right_on='Sector')

sector_df_2=sector_df.set_index('Sector')
fig = plt.figure(figsize=(20,7)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



width = 0.4



sector_df_2['mean_salary'].plot(kind='bar', color='#ee854a', ax=ax, width=width, position=1)

sector_df_2['value_counts'].plot(kind='bar', color='#4090c5', ax=ax2, width=width, position=0)



ax.set_ylabel('mean salary in thousands of $',fontsize=15)

ax.legend(['mean_salary'],bbox_to_anchor=(0.45, 1))

ax2.set_ylabel('number of companies',fontsize=15)

ax2.legend(['number of companies'],bbox_to_anchor=(0.65, 1))

plt.xlabel('Sector',fontsize=15)

plt.title("sector v/s mean salary v/s number of companies",fontsize=17)
plt.figure(figsize=(20,7))

sns.countplot(df.loc[df['Max_revenue']!=0,'Max_revenue'],color='#4090c5')

plt.ylabel('number of companies',fontsize=15)

plt.xlabel("Revenue in millions of $",fontsize=15)

plt.title('Number of companies with given revenue',fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
city_salary=pd.DataFrame(temp_df.groupby('City')['mean_salary'].mean()).reset_index()
top_city_df=pd.DataFrame(df['City'].value_counts()).reset_index()

top_city_df.rename(columns={'City':'count','index':'city'},inplace=True)

top_city_df=pd.merge(top_city_df,city_salary,how='left',left_on='city',right_on='City')

top_city_df.drop('City',axis=1,inplace=True)

top_city_df=top_city_df[top_city_df['count']>=5]

top_city_df=top_city_df.sort_values(by='mean_salary',ascending=False)



top_city_df=top_city_df.head(20)

plt.figure(figsize=(20,7))

sns.barplot(x=top_city_df['city'],y=top_city_df['mean_salary'],color='#4090c5')

plt.xticks(rotation=60,horizontalalignment='right')

plt.ylabel('mean salary in thousands of $',fontsize=15)

plt.xlabel("city",fontsize=15)

plt.title('top 20 cities with highest salaries',fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)
Title_salary=pd.DataFrame(temp_df.groupby('Job Title')['mean_salary'].mean()).reset_index().sort_values('mean_salary',ascending=False)

plt.figure(figsize=(20,7))

sns.barplot(x=Title_salary['Job Title'].head(20),y=Title_salary['mean_salary'].head(20),color='#4090c5')

plt.xticks(rotation=60,horizontalalignment='right')

plt.ylabel('mean salary in thousands of $',fontsize=15)

plt.xlabel("Job Title",fontsize=15)

plt.title('top 20 Job Titles with highest salaries',fontsize=17)

plt.xticks(fontsize=11)

plt.yticks(fontsize=13)
sector_revenue=pd.DataFrame(temp_df.groupby('Sector')['Max_revenue'].sum()).reset_index().sort_values('Max_revenue',ascending=False)

sector_revenue.rename(columns={'Max_revenue':"Total_revenue"},inplace=True)

plt.figure(figsize=(20,7))

sns.barplot(x=sector_revenue['Sector'],y=sector_revenue['Total_revenue'],color='#4090c5')

plt.xticks(rotation=60,horizontalalignment='right')

plt.ylabel('Total revenue in millions of $',fontsize=15)

plt.xlabel("Sector",fontsize=15)

plt.title('Sector wise total revenue generated',fontsize=17)

plt.xticks(fontsize=11)

plt.yticks(fontsize=13)