# libraries for data exploration and manipulation

import pandas as pd

import numpy as np



# libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

data.head()
# converting column names to snake case

colname_list = []

for col in data.columns:

    col = col.strip()

    col = col.lower()

    col = col.replace(' ','_')

    colname_list.append(col)

    

data.columns = colname_list  

data.head(2)
# dropping the unnecessary column

data.drop('unnamed:_0', axis=1, inplace=True)



# creating numerical features

salary_range = data.salary_estimate.str.split('(').str[0]

salary_range = salary_range.str.strip().str.replace('$','').str.replace('K','').str.split('-')

data['min_salary'], data['max_salary'] = salary_range.str[0], salary_range.str[1]
min_mode = data['min_salary'].mode()[0]



# replacing '' with the mode value

data['min_salary'] = data['min_salary'].replace('',min_mode)

data['min_salary'] = data['min_salary'].str.strip().astype(int)*1000

data['max_salary'] = data['max_salary'].str.strip().astype(int)*1000



# drop the salary_estimate column

data.drop('salary_estimate', axis=1, inplace=True)
data.company_name = data.company_name.str.split('\n').str[0]

data.headquarters = data.headquarters.str.split(',').str[0]

data.location = data.location.str.split(',').str[0]
for val in ['-1', -1, -1.0]:

    data = data.replace(val, np.nan)



# list of columns containing null values    

cols_with_null = [col for col in data.columns 

                 if data[col].isnull().sum()>0]  



# columns with missing values and displaying the dimension of data

cols_with_null, data.shape  
data[cols_with_null].isnull().sum().sort_values(ascending=False)
data[cols_with_null].describe(include='all')
# filling missing rating with median rating

data.rating.fillna(3.7, inplace=True)



# filling missing headquarters with the location

data.loc[data.headquarters.isnull(), 'headquarters'] = data.location



# filling missing size

mode_size = data['size'].mode()[0]

data['size'].fillna(mode_size, inplace=True)



# filling type of owenership

mode_ownership = data.type_of_ownership.mode()[0]

data.type_of_ownership.fillna(mode_ownership, inplace=True)



# filling type of owenership

mode_industry = data.industry.mode()[0]

data.industry.fillna(mode_industry, inplace=True)



# filling type of owenership

mode_sector = data.sector.mode()[0]

data.sector.fillna(mode_sector, inplace=True)



# filling type of owenership

mode_revenue = data.revenue.mode()[0]

data.revenue.fillna(mode_revenue, inplace=True)



# filling easy apply

data.easy_apply = data.easy_apply.replace({

    'True':True,

    np.nan:False

})



# dropping one row where company name is not given

data.dropna(subset=['company_name'], inplace=True)
data.job_title = data.job_title.str.split(',').str[0]

data.job_title = data.job_title.str.title()

data.job_title = data.job_title.str.replace('.', '')

data.job_title = data.job_title.str.replace('Sr', 'Senior')

data.job_title = data.job_title.str.replace('Data Analyst Junior', 'Junior Data Analyst')
# making a new feature to show maximum revenue of companies

rev = data.revenue.str.split().str[-3:-1]

rev_series = []

for val in rev:

    if val[1] == 'million': 

        price = val[0].replace('$','').replace('+','').strip()

        price = int(price)*1000000  # multiply the value with 1000000 if price is in million

        rev_series.append(price)

    elif val[1] == 'billion':

        price = val[0].replace('$','').replace('+','').strip()

        price = int(price)*1000000000  # multiply the value with 1000000000 if price is in billion

        rev_series.append(price)

    else:

        rev_series.append(np.nan)  # if price is not given then fill null value

        

max_rev = pd.Series(rev_series)



# add new column to the data

data['max_revenue'] = max_rev 

data.max_revenue.fillna(data.max_revenue.median(), inplace=True)
data.head(3)
# frequency of each sector

sectors = data.sector.value_counts()

sector_job = pd.DataFrame({'sector':sectors.index,

                         'jobs':sectors.values})

# creating bar graph for the frequency of each sector

fig = px.bar(

    sector_job,

    x='sector',

    y='jobs',

    title='Number of jobs in each sectors',

    color_discrete_sequence =['#EB5377'],

    template='plotly_white'

           )



# updating the ylabel and xticks rotation

fig.update_yaxes(title_text='Number of jobs')

fig.update_xaxes(tickangle=45)

fig.show()
# avg. maximum revenue of each sector

grouped_sector = data.groupby('sector')['max_revenue'].mean()

sector_df = pd.DataFrame({'sector':grouped_sector.index,

                         'max_revenue':grouped_sector.values})



# creating bar graph

fig = px.bar(

    sector_df,

    x='sector',

    y='max_revenue',

    title='Average maximum revenue(USD) in each sector',

    color_discrete_sequence =['#03D6C1'],

    template='plotly_white'

           )



# updating ylabel and xticks rotation

fig.update_yaxes(title_text='Maximum revenue')

fig.update_xaxes(tickangle=45)

fig.show()
# avg minimum and maximum salary in each sector

salary = data.groupby('sector')[['min_salary','max_salary']].mean()

sal_df = pd.DataFrame({'sector':salary.index,

                      'min_salary':salary.min_salary,

                      'max_salary':salary.max_salary})



# creating bar graph

fig = px.bar(sal_df,

             x='sector',

             y=['min_salary','max_salary'],

            color_discrete_sequence=['#273746','#A9DFBF'],

             title='Average salary(USD) in sectors',

             template='plotly_white'

            )



# updating ylabel and xticks rotation

fig.update_yaxes(title_text='Salary')

fig.update_xaxes(tickangle=45)

fig.show()
# taking data only of easy apply

df = data[data.easy_apply==True]

companies = df.company_name.value_counts()[:10]

comp_df = pd.DataFrame({'company':companies.index,

                       'total_job':companies.values})



# creating bar chart

fig = go.Figure(data=[go.Pie(

    labels=comp_df.company,

    values=comp_df.total_job,

    hole=.5)])

fig.update_layout(title='Top 10 recruiters providing easy apply')

fig.show()
job_counts = data.job_title.value_counts(ascending=False)[:15]

job_df = pd.DataFrame({'job_title':job_counts.index,

                       'total_job':job_counts.values})



# creating bar chart

fig = px.bar(

    job_df,

    y='total_job',

    x='job_title',

    title='Number of jobs for each title',

    color_discrete_sequence =['#EC70CC'],

    template='plotly_white'

           )



# updating ylabel

fig.update_yaxes(title_text='Number of jobs')

fig.show()
job_list = job_df.job_title.tolist()

job_df2 = data[data.job_title.isin(job_list)]

job_grp = job_df2.groupby('job_title')[['min_salary','max_salary']].median()



# creating bar graph

fig = px.bar(job_grp,

             x=job_grp.index,

             y=['min_salary','max_salary'],

            color_discrete_sequence=['#15CC6F','#94EB53'],

             title='Average salary(USD) in top 15 job roles',

             template='plotly_white'

            )



# updating xlabel and ylabel

fig.update_yaxes(title_text='Salary')

fig.update_xaxes(tickangle=45)

fig.show()
star_grp = job_df2.groupby('job_title')['rating'].mean()



# creating bar chart

fig = go.Figure(go.Bar(

    x=star_grp.index,y=star_grp.values,

    marker={'color': star_grp.values, 

    'colorscale': 'agsunset'}

))



# updating title and labels

fig.update_layout(title_text='Top 15 job roles rating',

                  xaxis_title="Job roles",

                  yaxis_title="Ratings")

fig.show()
owner_grp = data.groupby('type_of_ownership')['max_revenue'].mean()

fig = go.Figure(data=[go.Scatter(

    x=owner_grp.index, y=owner_grp.values,

    mode='markers',

    marker=dict(

        color=owner_grp.values,

        size=owner_grp.values*0.000000025,

        showscale=True))])



# updating the labels and title

fig.update_layout(template='plotly_white',

                  title='Maximum avg. revenue in differnt types of ownership',

                  xaxis_title="Type of ownership",

                  yaxis_title="Maximum avg. revenue")

fig.show()
top_location = data.location.value_counts()[:20]

loc_df = pd.DataFrame({'location':top_location.index,

                       'total_job':top_location.values})



# creating pie chart

fig = go.Figure(data=[go.Pie(

    labels=loc_df.location,

    values=loc_df.total_job,

    hole=.5)])

fig.update_layout(title='Top 20 locations providing analytics jobs')

fig.show()
# number of jobs in New York

ny = data[data.location == 'New York']

ny.job_title.value_counts()[:10]