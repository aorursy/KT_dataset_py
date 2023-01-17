# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import cufflinks as cf
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
donors_data = pd.read_csv("../input/Donors.csv")
donors_data.count()
# Loading the donations dataset
donations_data = pd.read_csv("../input/Donations.csv")
donations_data.head()
donor_is_teacher = donors_data['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': donor_is_teacher.index,
                   'values': donor_is_teacher.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Not Teacher vs Teacher')
donors_donations = donations_data.merge(donors_data, on='Donor ID', how='inner')
state_wise = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})
state_wise.sort_values(by="Donation Amount",ascending=False).head(10)
city_wise = donors_donations.groupby('Donor City', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})
city_wise.sort_values(by="Donation Amount",ascending=False).head(10)
temp = donors_data["Donor State"].value_counts().head(30)
total = donors_data.shape[0]
temp= temp.apply(lambda x: (x/total)*100)
#for x in temp.values:
#    print(x)
temp = donors_donations["Donor City"].value_counts().head(30)
total = donors_donations.shape[0]
temp = temp.apply(lambda x : (x/total)*100)
temp.iplot(kind='bar', xTitle = 'City name', yTitle = "Count", title = 'Top Percentage Donor cities')
donations_data.head()
#donations_data[["Donor ID","Donation Amount"]]
donations_sum = donations_data["Donation Amount"].sort_values()
donations_data["Donation Amount"].sort_values()[-10:].plot(kind="Bar",title="Top Donation Amount by a Donor")
donation_10 = donations_data[donations_data["Donation Amount"] <= 10]
total= donations_data.shape[0]
donation_100 = donations_data[(donations_data["Donation Amount"] > 10) & (donations_data["Donation Amount"] <= 100)] 
donation_1000 = donations_data[(donations_data["Donation Amount"] > 100) & (donations_data["Donation Amount"] <= 1000)] 
donations_percent = {"10":(donation_10.shape[0]/total)*100, "100":(donation_100.shape[0]/total)*100,"1000":(donation_1000.shape[0]/total)*100}
donation_amount = donations_data[(donations_data["Donation Amount"] > 1000) & (donations_data["Donation Amount"] <= 10000)]
donations_percent["10000"]=(donation_amount.shape[0]/total)*100
donation_amount = donations_data[(donations_data["Donation Amount"] > 10000) & (donations_data["Donation Amount"] <= 30000)]
donations_percent["30000"]=(donation_amount.shape[0]/total)*100
donation_amount = donations_data[(donations_data["Donation Amount"] > 30000) & (donations_data["Donation Amount"] <= 50000)]
donations_percent["50000"]=(donation_amount.shape[0]/total)*100
donation_amount = donations_data[(donations_data["Donation Amount"] > 50000)]
donations_percent["100000"]=donation_amount.shape[0]
print(donations_percent)
total = donations_data.shape[0]
donations_percent
df = pd.Series(data=donations_percent)
df.plot(kind="bar")
df.head(10)
#df.iplot(kind='pie',title='Distribution of Donation Amount wise ')
donations_data[donations_data["Donor ID"]== "8f70fc7370842e0709cd9af3d29b4b0b"]["Donation Amount"].sum()
projects_data = pd.read_csv("../input/Projects.csv")
projects_data.head()

donations_merge_projects = donations_data.merge(projects_data,how="inner",on="Project ID")
donations_merge_projects.head()
donations_merge_projects.groupby("Project Subject Category Tree").count()["Donation Amount"].plot(kind="Bar",figsize=(15,10),title="Donations Towards different Projects")
donations_merge_projects.groupby("Project Subject Category Tree").sum()["Donation Amount"].sort_values(ascending=False)[0:30].plot(kind="Bar",figsize=(15,10),title="Donations Towards different Projects")
resources_data = pd.read_csv("../input/Resources.csv")
resources_data.head()
temp = resources_data["Resource Item Name"].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="blue"),
)]
layout = go.Layout(
    title='Top Resource Requests',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="baic-bar")
total_resources = resources_data.shape[0]
vendor_names = temp.index.values[0:15]
#print(vendor_names)
temp = temp.head(15)
vendor_percent = [(x) for x in temp]
temp.head(10)
temp = resources_data["Resource Vendor Name"].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="light blue"),
)]
layout = go.Layout(
    title='Top Resource Vendors ',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="baic-bar")
total_resources = resources_data.shape[0]
vendor_names = temp.index.values[0:15]
#print(vendor_names)
temp = temp.head(15)
vendor_percent = [(x/total_resources)*100 for x in temp]
#print(vendor_percent)
temp.head(10)
teachers = pd.read_csv("../input/Teachers.csv")
teachers.head()
temp = teachers["Teacher Prefix"].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Distribution of Teacher Prefix')

from datetime import datetime
teachers["date_year"] = teachers['Teacher First Project Posted Date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").year)
temp = teachers.groupby("date_year").count()
temp = temp["Teacher First Project Posted Date"]

fig  = {
    'data':[ {
        'x': temp.index,
        'y': temp.values
    }
    ],
    'layout' : {
       'title' : 'Trends of Number of Teachers Posted  their First Project (2002 - 2018)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Teachers Posted First Project"}
    }
    
    
}
#fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="baic-bar")