import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head(2)
df.drop(['Unnamed: 0'], axis=1,inplace=True)
df.isna().sum()
df[df.isnull().any(axis=1)]
df.dropna(axis=0 , subset=['Company Name'], inplace=True)
missing_val_dict = {

    -1 : np.nan,

    -1.0 : np.nan,

    '-1' : np.nan

}
df.replace(missing_val_dict, inplace=True)
df['Easy Apply'].replace(np.nan, False, inplace=True)
df.isna().sum()
df['Job Title'], df['Department'] = df['Job Title'].str.split(',', 1).str
df['Job Title'].value_counts()[:20]
df['Job Title'] = df['Job Title'].replace(['Sr. Data Analyst', 'Sr Data Analyst'], 'Senior Data Analyst')
df['Job Title'].value_counts()[:20]
df['Salary Estimate']
df['Salary Estimate'],_ = df['Salary Estimate'].str.split('(', 1).str

df['Min Salary'], df['Max Salary'] = df['Salary Estimate'].str.split('-').str

df.dropna(axis=0 , subset=['Max Salary'], inplace=True)
df['Max Salary'] = df['Max Salary'].str.extract('(\d+)')

df['Min Salary'] = df['Min Salary'].str.extract('(\d+)')



df['Min Salary'] = df['Min Salary'].astype(str).astype(int)

df['Max Salary'] = df['Max Salary'].astype(str).astype(int)
del df['Salary Estimate']
df['Company Name'], temp = df['Company Name'].str.split('\n', 1).str
df['Location'].value_counts()[:20]
df['City'], df['State'] = df['Location'].str.split(',', 1).str
df['State'] = df['State'].replace([' Arapahoe, CO'], ' CO')
df['State'] = df['State'].str.strip()

df['City'] = df['City'].str.strip()
df['State'].value_counts()
df['Industry'] = df['Industry'].fillna('Others')
df['Sector'] = df['Sector'].fillna('Others')
df['Rating'] = df['Rating'].fillna(round(df['Rating'].mean(), 1))
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
df['Revenue'] = df['Revenue'].apply(lambda x: filter_revenue(x))
important_column = ['Job Title', 'Rating', 'Company Name', 'State', 'City','Size', 'Industry', 'Sector', 'Min Salary', 'Max Salary', 'Revenue']
df[important_column].head()
top_20_job = pd.DataFrame(df['Job Title'].value_counts()[:20]).reset_index()

top_20_job.rename(columns={'index': 'Job Title', 'Job Title': 'No. of Openings'}, inplace=True)
fig = go.Figure(go.Bar(

    x=top_20_job['Job Title'],

    y=top_20_job['No. of Openings'],

))

fig.update_layout(title_text='Current openings in different Roles',xaxis_title="Job Title",yaxis_title="Number of openings")

fig.show()
top_20_industry = pd.DataFrame(df['Industry'].value_counts()[1:21]).reset_index()

top_20_industry.rename(columns={'index': 'Industry', 'Industry': 'No. of Openings'}, inplace=True)
fig = go.Figure(go.Bar(

    x=top_20_industry['Industry'],

    y=top_20_industry['No. of Openings'],

))

fig.update_layout(title_text='Current openings in different Industry',xaxis_title="Industry",yaxis_title="Number of openings")

fig.show()
top_20_city = pd.DataFrame(df['City'].value_counts()[:20]).reset_index()

top_20_city.rename(columns={'index':'City', 'City':'No. of Openings'}, inplace=True)
fig = go.Figure(go.Bar(

    x=top_20_city['City'],

    y=top_20_city['No. of Openings'],

))

fig.update_layout(title_text='Current openings in different City',xaxis_title="City",yaxis_title="Number of openings")

fig.show()
top_20_company = pd.DataFrame(df['Company Name'].value_counts()[:20]).reset_index()

top_20_company.rename(columns={'index':'Company Name' , 'Company Name':'No. of Openings'},inplace=True)
companies = top_20_company['Company Name'].values

revenue_rating = df[df['Company Name'].isin(companies)][['Company Name','Rating', 'Revenue']]

revenue_rating = revenue_rating.groupby('Company Name').mean()
fig = go.Figure(go.Bar(

    x=top_20_company['Company Name'],

    y=top_20_company['No. of Openings'],

))

fig.update_layout(title_text='Current openings in different City',xaxis_title="Company",yaxis_title="Number of openings")

fig.show()
df.dropna(axis=0 , subset=['Max Salary','Min Salary'], inplace=True)
grp_job_title = df[['Job Title','Min Salary', 'Max Salary']].groupby('Job Title').mean().reset_index()

grp_job_title = grp_job_title[grp_job_title['Job Title'].isin(top_20_job['Job Title'].values)].reset_index()

del grp_job_title['index']
grp_job_title['Min Salary'] = grp_job_title['Min Salary'].round(2)

grp_job_title['Max Salary'] = grp_job_title['Max Salary'].round(2)
fig = go.Figure(data=[

    go.Bar(name='Min Salary', x=grp_job_title['Job Title'], y=grp_job_title['Min Salary'],marker_color='indianred'),

    go.Bar(name='Max Salary', x=grp_job_title['Job Title'], y=grp_job_title['Max Salary'],marker_color='lightsalmon'),

])

# Change the bar mode

fig.update_layout(barmode='group', title='Min and Max salary of top 20 Job openings',

                 yaxis=dict(

                    title='USD (_K)',

                    titlefont_size=16,

                    tickfont_size=14,

                ),

                xaxis=dict(

                    title='Job Title',

                    titlefont_size=16,

                    tickfont_size=14,

                ))



fig.show()