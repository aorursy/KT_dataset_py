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
path="/kaggle/input/covid19-in-india/"

def load_data(data):

    return pd.read_csv(path+data)
cvd=pd.read_csv('../input/covid19-in-india/covid_19_india.csv') # Reading Covid-19 Csv

print(cvd.head())

age_group=load_data('AgeGroupDetails.csv') #Age Group Details Loading data using load_data func!!

ind_det=load_data('IndividualDetails.csv') #Individual details Loading data using load_data func!!



# Similary Loading other csv using load_data func



state_Test=load_data("StatewiseTestingDetails.csv")



bed=load_data('HospitalBedsIndia.csv') # Loading bed details



print(ind_det.current_status.unique())
# Creating a custom table for better understanding of the data 

state_details = pd.pivot_table(cvd, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')

# Calculating the recovery rate which is Cured/Confirmed rounding to 2 digits

state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],2)

# Similarly, for Death Rate

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)



state_details = state_details.sort_values(by='Confirmed', ascending= False).reset_index(level=0)



state_details.style.background_gradient(cmap='plasma_r')
# Renaming State details

state_details.rename(columns={'State/UnionTerritory':'STUT'}, inplace=True)

state_details.head()
import plotly.express as px

df = px.data.tips()

# Plotting the Density_Contour

fig = px.density_contour(state_details, x="Confirmed", y="Deaths") # Plotting Contour for Confirmed Vs Deaths Comparasion

fig.update_traces(contours_coloring="fill", contours_showlabels = True)

fig.show()
# Similary for Confirmed Vs Cured Comparasion

fig = px.density_contour(state_details, x="Confirmed", y="Cured")

fig.update_traces(contours_coloring="fill", contours_showlabels = True)

fig.show()
import plotly.graph_objs as go

import plotly.offline as pyo # Setting Notebook to work Offline with Plotly

import plotly

pyo.init_notebook_mode()



# Acessing the values from state_details

x = state_details.STUT



trace1 = {

  'x': x,

  'y': state_details.Confirmed,# Created a trace variable to store confirmed cases as a bar per state wise ,similarly for Cured and Deaths

  'name': 'Confirmed',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': state_details.Cured,

  'name': 'Cured',

  'type': 'bar'

};



trace3 = {

  'x': x,

  'y': state_details.Deaths,

  'name': 'Deaths',

  'type': 'bar'

};



data = [trace1, trace2,trace3]; # A singleton row matrix to store the trace1,trace2,trace3



layout = {

  'xaxis': {'title': ' State-Data '},

  'barmode': 'relative',

  'title': 'Case Wise Disturbution'

};



fig = go.Figure(data = data, layout = layout)#Plotting the bar plot form the above data 

pyo.iplot(fig)
import plotly.express as px

df = px.data.wind()

# Plotting bar_polar plot

fig = px.bar_polar(state_details, r=state_details.Confirmed, theta=state_details.Cured, color=state_details.STUT, template="plotly_dark",color_discrete_sequence= px.colors.sequential.Plasma_r)

fig.show()
locations = ind_det.groupby(['detected_state', 'detected_district', 'detected_city'])['government_id'].count().reset_index()

locations['country'] = 'India'

# Plotting a Tree map!

fig = px.treemap(locations, path=["country", "detected_state", "detected_district", "detected_city"], values="government_id", height=700,title='State ---> District --> City', color_discrete_sequence = px.colors.qualitative.Prism)



fig.data[0].textinfo = 'label+text+value+percent entry+percent root'

fig.show()