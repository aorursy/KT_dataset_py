# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hospdata=pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")



hospdata=hospdata.drop(['Unnamed: 12', 'Unnamed: 13'], axis=1)

hospdata.rename(columns = {'NumPrimaryHealthCenters_HMIS':'Primary Health Center', 'NumCommunityHealthCenters_HMIS':'Community Health Center','NumSubDistrictHospitals_HMIS':'Sub District Hospital', 'NumDistrictHospitals_HMIS':'District Hospitals'}, inplace = True) 

hospdata.rename(columns = {'TotalPublicHealthFacilities_HMIS':'Total Public Health Facility', 'NumPublicBeds_HMIS':'Public Beds','NumRuralHospitals_NHP18':'Rural Hospitals', 'NumRuralBeds_NHP18':'Rural Hosp Beds','NumUrbanHospitals_NHP18':'Urban Hospitals','NumUrbanBeds_NHP18':'Urban Hosp Beds'}, inplace = True) 

hospdata1=hospdata.drop([36,37], axis=0)
import plotly.express as px

fig = px.bar(hospdata1.sort_values('Urban Hospitals', ascending=False).sort_values('Urban Hospitals', ascending=True), 

             x="Urban Hospitals", y="State/UT", title='Total Urban Health Centers', text='Urban Hospitals', orientation='h',width=1000, height=700, range_x = [0, max(hospdata1['Urban Hospitals'])]) 

            

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()


fig = px.bar(hospdata1.sort_values('Rural Hospitals', ascending=False).sort_values('Rural Hospitals', ascending=True), 

             x="Rural Hospitals", y="State/UT", title='Total Rural Health Centers', text='Rural Hospitals', orientation='h',width=1000, height=700, range_x = [0, max(hospdata1['Rural Hospitals'])]) 

            

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt







sns.set_style("white")

sns.set_context({"figure.figsize": (24, 10)})





sns.barplot(x = hospdata1['Urban Hosp Beds'], y = hospdata1['State/UT'], color = "red")





bottom_plot = sns.barplot(x = hospdata1['Rural Hosp Beds'], y = hospdata1['State/UT'], color = "#0000A3", )





topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')

bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')

l = plt.legend([bottombar, topbar], ['Rural Hosp Beds', 'Urban Hosp Beds'], loc=1, ncol = 2, prop={'size':16})

l.draw_frame(False)





sns.despine(left=True)

bottom_plot.set_ylabel("States")

bottom_plot.set_xlabel("Hospital Beds")





for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +

             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):

    item.set_fontsize(16)


f, ax = plt.subplots(figsize=(15,10))



data = hospdata1[['State/UT','Community Health Center']]

data.sort_values(['Community Health Center'])

sns.set_color_codes("muted")

sns.barplot(x="Community Health Center", y="State/UT",data=data, label="Community Health Center", color="g")

fig = px.bar(hospdata1, x="Primary Health Center", y="State/UT", color='Primary Health Center', orientation='h', height=800,

             title='Primary Health Center', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()

fig = px.bar(hospdata1, x="Total Public Health Facility", y="State/UT", color='Total Public Health Facility', orientation='h', height=800,

             title='Total Health Facility in India', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()



fig = px.scatter(hospdata1, x="Sub District Hospital", y="District Hospitals", color="State/UT", marginal_y="rug", marginal_x="histogram")

fig


fig = px.scatter(hospdata1, x="Total Public Health Facility", y="Public Beds", color="State/UT", marginal_y="rug", marginal_x="histogram")

fig