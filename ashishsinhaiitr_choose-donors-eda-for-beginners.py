# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

resources=pd.read_csv('../input/Resources.csv')
donors=pd.read_csv('../input/Donors.csv')
schools=pd.read_csv('../input/Schools.csv')
teachers=pd.read_csv('../input/Teachers.csv')
projects=pd.read_csv('../input/Projects.csv')
donations=pd.read_csv('../input/Donations.csv')
resources.head()
donors.head()
schools.head()
teachers.head()
projects.head()
donations.head()
donations.shape
teachers.shape
schools.shape
donors.shape
projects.shape
resources.shape
donations.describe()
donations.isnull().sum().apply(lambda x: x/donations.shape[0])
donors.describe()
donors.isnull().sum().apply(lambda x: x/donors.shape[0])
donors[['Donor City','Donor Zip']].nunique()
resources.describe()
resources.dtypes
resources['Resource Vendor Name'].nunique()
resources.isnull().sum().apply(lambda x: x/resources.shape[0])
schools.describe()
schools.isnull().sum().apply(lambda x: x/schools.shape[0])
projects['Teacher Project Posted Sequence'].describe()
projects['Teacher Project Posted Sequence'].nunique()
projects.isnull().sum().apply(lambda x: x/projects.shape[0])
teachers.describe()
teachers['Teacher Prefix'].unique()
teachers.isnull().sum().apply(lambda x: x/teachers.shape[0])
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations.head()
donations.dropna(axis=0,how='any',inplace=True)
donors.dropna(axis=0,how='any',inplace=True)
resources.dropna(axis=0,how='any',inplace=True)
projects.dropna(axis=0,how='any',inplace=True)
schools.dropna(axis=0,how='any',inplace=True)
teachers.dropna(axis=0,how='any',inplace=True)
df = donors.groupby("Donor State")['Donor City'].count().to_frame().reset_index()
X = df['Donor State'].tolist()
Y = df['Donor City'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Count of Donor Cities' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Count of Donor Cities",data=data)
sns.countplot(x='School Metro Type',data = schools)
donors_state_amount=donors_donations.groupby('Donor State')['Donation Amount'].sum().reset_index()
donors_state_amount['Donation Amount']=donors_state_amount['Donation Amount'].apply(lambda x: format(x, 'f'))

df = donors_state_amount[['Donor State','Donation Amount']]
X = df['Donor State'].tolist()
Y = df['Donation Amount'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Total Donation Amount' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Total Donation Amount",data=data)
data = donors_donations["Donor State"].value_counts().head(25)
plt.figure(figsize=(20,10))
sns.barplot(data=da,x='Donor State')
state_count = data.to_frame(name="number_of_projects").reset_index()
state_count = state_count.rename(columns= {'index': 'Donor State'})
# merging states with projects and amount funded
donor_state_amount_project = state_count.merge(donors_state_amount, on='Donor State', how='inner')

val = [x/y for x, y in zip(donor_state_amount_project['Donation Amount'].apply(float).tolist(),donor_state_amount_project['number_of_projects'].tolist())]
state_average_funding = pd.DataFrame({'Donor State':donor_state_amount_project['Donor State'][-5:][::-1],'Average Funding':val[-5:][::-1]})
sns.barplot(x="Donor State",y="Average Funding",data=state_average_funding)
schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()
projects_resources = projects.merge(resources, on='Project ID', how='inner')
projects_resources.head()
sns.barplot(projects_resources['Resource Vendor Name'])
project_title = projects_resources['Project Title'].value_counts()[:5]
sns.barplot(data=project_title).set_title('Unique project title')
school_project = schools.merge(projects, on='School ID', how='inner')
project_open_close=school_project[['Project Resource Category','Project Posted Date','Project Fully Funded Date']]
project_open_close['Project Posted Date'] = pd.to_datetime(project_open_close['Project Posted Date'])
project_open_close['Project Fully Funded Date'] = pd.to_datetime(project_open_close['Project Fully Funded Date'])

time_gap = []
for i in range(school_project['School ID'].count()):
    if school_project['Project Current Status'][i] =='Fully Funded':
        time_gap.append(abs(project_open_close['Project Fully Funded Date'][i]-project_open_close['Project Posted Date'][i]).days)
    else:
        time_gap.append(-1)

project_open_close['Time Duration(days)'] = time_gap
project_open_close.head()

project_open_close_resource=project_open_close.groupby('Project Resource Category')['Time Duration(days)'].mean().reset_index()
df = project_open_close_resource[['Project Resource Category','Time Duration(days)']]
X = df['Project Resource Category'].tolist()
Y = df['Time Duration(days)'].apply(int).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Project Resource Category' : Z[0:5], 'Total Time Duration(days)' : sorted(Y)[0:5] })
sns.barplot(x="Total Time Duration(days)",y="Project Resource Category",data=data)



