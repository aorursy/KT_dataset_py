import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sb



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

from plotly import graph_objs as go

import plotly.express as px

# For Notebooks

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
space = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv", index_col=0)
def preview():

    '''

    This method will render a preview of space dataset.

    '''

    return space.head(5)
preview()
space.drop('Unnamed: 0.1', axis=1, inplace=True)
space.columns = space.columns.str.lower().str.strip().str.replace(" ", "_")
preview()
space['datum'] = pd.to_datetime(space['datum'])
space['year'] = space['datum'].apply(lambda x: x.year)
space['month'] = space['datum'].apply(lambda x: x.month)
space['country'] = space['location'].apply(lambda x : x.split(",")[-1].strip())
space.isnull().sum()
space.drop('rocket', axis=1, inplace = True)
preview()
plt.figure(figsize=(18,9))

plt.plot(space['year'].unique(), space['year'].value_counts())

plt.xticks(rotation=90)

x_ticks = np.arange(1957, 2020, 2)

plt.xticks(x_ticks)

plt.xlabel('Year', fontsize=16)

plt.ylabel('Number of launches', fontsize=16)

plt.tick_params(labelsize=12)

plt.title('Total number of launches each year', fontsize=16)
def compare_by_company(first_company, second_company):

    '''

    This method will group the data for the give company name and will render the compring line chart for the same.

    '''

    df1 = pd.DataFrame({'count_company_1' : space[space['company_name']==first_company].groupby(by='year').size()})

    df2 = pd.DataFrame({'count_company_2' : space[space['company_name']==second_company].groupby(by='year').size()})

    plt.figure(figsize=(18,9))

    plt.plot(df1['count_company_1'])

    plt.plot(df2['count_company_2'], 'r')

    plt.xticks(rotation=90)

    x_ticks = np.arange(1957, 2020, 3)

    plt.xticks(x_ticks)

    plt.xlabel('Year', fontsize=16)

    plt.ylabel('Number of launches', fontsize=16)

    plt.tick_params(labelsize=12)

    plt.title(f'Total number of launches each year : {first_company} vs {second_company}' , fontsize=16)



    plt.legend([first_company, second_company])
compare_by_company('ISRO', 'NASA') # Go on give it a try change the company_name
#Use one of these company names with compare_by_company method.

print(space.company_name.unique().tolist())
plt.figure(figsize=(10,10))

ax = sb.countplot(y="country", data=space, order=space['country'].value_counts().index)

ax.set_xscale("log")

ax.axes.set_title("Countries against their number of launches", fontsize=16)

ax.set_xlabel("Number of launches (log scale)", fontsize=16)

ax.set_ylabel("Country", fontsize=16)

ax.tick_params(labelsize=12)

plt.show()
space['country'].unique()
def compare_by_country2(first_country, second_country):

    '''

    This method will group the data for the give country name and will render the compring line chart for the same.

    '''

    df1 = pd.DataFrame({first_country : space[space['country']==first_country].groupby(by='year').size()})

    df2 = pd.DataFrame({second_country : space[space['country']==second_country].groupby(by='year').size()})

    df = df1.join(df2)

    df.fillna(0)

    _fig = df[[first_country, second_country]].iplot(kind='scatter', xTitle="year", yTitle = "Number of launches",

                 title='Launches based on country', asFigure=True)

    _fig.iplot()
compare_by_country2("USA", "Russia")
print(space['status_mission'].unique().tolist())
#Here we have four list each dedicated for a status for all the countries.

success_list = []

failure_list = []

prelaunch_failure = []

partial_failure = []

for country in space['country'].unique():

    temp = space[space['country']==country]

    success_list.append(temp[temp['status_mission']=='Success']['status_mission'].count())

    failure_list.append(temp[temp['status_mission']=='Failure']['status_mission'].count())

    prelaunch_failure.append(temp[temp['status_mission']=='Prelaunch Failure']['status_mission'].count())

    partial_failure.append(temp[temp['status_mission']=='Partial Failure']['status_mission'].count())
plt.figure(figsize=(18,9))



plt.bar(x=space['country'].unique(), height=success_list)

plt.bar(x=space['country'].unique(), height=failure_list)

plt.bar(x=space['country'].unique(), height=prelaunch_failure)

plt.bar(x=space['country'].unique(), height=partial_failure)



plt.xticks(rotation=90, fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('Country', fontsize=16)

plt.ylabel('Number', fontsize=16)

plt.legend(['Success', 'Failure', 'Prelaunch Failure', 'Partial Failure'], title='Mission Status')

plt.title('Mission status for each country', fontsize=16)
#Here we have four list each dedicated for a status for all the countries.

success_list = []

failure_list = []

prelaunch_failure = []

partial_failure = []

for country in space['company_name'].unique():

    temp = space[space['company_name']==country]

    success_list.append(temp[temp['status_mission']=='Success']['status_mission'].count())

    failure_list.append(temp[temp['status_mission']=='Failure']['status_mission'].count())

    prelaunch_failure.append(temp[temp['status_mission']=='Prelaunch Failure']['status_mission'].count())

    partial_failure.append(temp[temp['status_mission']=='Partial Failure']['status_mission'].count())

    

plt.figure(figsize=(18,9))



plt.bar(x=space['company_name'].unique(), height=success_list)

plt.bar(x=space['company_name'].unique(), height=failure_list)

plt.bar(x=space['company_name'].unique(), height=prelaunch_failure)

plt.bar(x=space['company_name'].unique(), height=partial_failure)



plt.xticks(rotation=90, fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('Company', fontsize=16)

plt.ylabel('Number', fontsize=16)

plt.legend(['Success', 'Failure', 'Prelaunch Failure', 'Partial Failure'], title='Mission Status')

plt.title('Mission status for each company', fontsize=16)
#Top 5 Companies with highest number of success'

temp = space[space['status_mission']== 'Success']

success =  pd.DataFrame({'count' : temp.groupby( "company_name").size()}).reset_index()

success.sort_values(by='count', ascending=False).head()['company_name']
#Top 5 Companies with highest number of failures

temp = space[space['status_mission']== 'Failure']

fail =  pd.DataFrame({'count' : temp.groupby( "company_name").size()}).reset_index()

fail.sort_values(by='count', ascending=False).head()['company_name']
fig, axes = plt.subplots(1,2,figsize=(18,6))

sb.barplot(x=success.sort_values(by='count', ascending=False).head()['company_name'], y = success['count'], ax = axes[0])

axes[0].set_xlabel("Companies", fontsize=12)

axes[0].set_ylabel("Number of success", fontsize=12)

axes[0].set_title("Top 5 companies with highest number of success.")

sb.barplot(x=fail.sort_values(by='count', ascending=False).head()['company_name'], y = fail['count'], ax = axes[1])

axes[1].set_xlabel("Companies", fontsize=12)

axes[1].set_ylabel("Number of failure", fontsize=12)

axes[1].set_title("Top 5 companies with highest number of failure.")

plt.xticks(rotation=90)
preview()
plt.figure(figsize=(18,9))

sb.countplot(x='status_rocket', data=space)

plt.xlabel("Rocket Status",fontsize=16)

plt.ylabel("Number of active/retired",fontsize=16)
preview()
temp = space['status_mission'].value_counts().reset_index()

temp.columns  = ['status_mission', 'count']

fig = px.pie(temp, values='count', names='status_mission')

fig.show()
temp = space.groupby(by='month').count().reset_index()[['month','detail']]

temp.columns = ['month', 'count']

temp.iplot(kind='bar', x='month', y='count', color='red')
def mission_status_percentages(company_name):

    success_query = space.query(f"company_name=='{company_name}' and status_mission=='Success'")

    all_scenarios = space.query(f"company_name=='{company_name}'") #all_scenario -> success and failure all data

    print(f'has success percentage of {success_query.shape[0] / all_scenarios.shape[0] * 100}')
print("Enter the name of the compant to get the success percentage.\n")

print(f"For reference you can use company name from this list : {space.company_name.unique().tolist()}")
mission_status_percentages('SpaceX')
space
space.groupby(by = ['status_rocket', 'company_name']).count()
space[space['status_rocket']=='StatusActive'].groupby('company_name').size().sort_values(ascending=False).head(5)
def active_by_year(year):

    return space[(space['status_rocket']=='StatusActive') & (space['year']==year)].groupby('company_name').size().sort_values(ascending=False).head(5)
active_by_year(2020)