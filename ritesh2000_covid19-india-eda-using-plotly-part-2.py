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
#Import Libraries

import plotly.graph_objs as go

import plotly.express as px

import plotly

import plotly.offline as pyo
fig = go.Figure() # Using Plotly graph objects!!



# Creating a Plot for understanding the Age Distribution over Covid19

fig.add_trace(go.Scatter(x=age_group['AgeGroup'],y=age_group['TotalCases'],line_shape='vhv',fill='tonextx',fillcolor ='gold')) 

fig.update_layout(title="Age wise Disturbution",yaxis_title="Total Number of cases",xaxis_title="Age Group")

fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=800,height=600)

fig.show()
px.histogram(ind_det, x='gender', color_discrete_sequence = ['palegoldenrod'], title='GenderWise  Distribution')
cp1=ind_det.copy()



cp1['current_status']=cp1['current_status'].replace(np.nan,'NaN')

cp2=cp1[cp1.gender == 'M'].groupby(['current_status']).count().reset_index()

cp3=cp1[cp1.gender == 'F'].groupby(['current_status']).count().reset_index()



cp2.rename(columns = {'id':'Male'}, inplace = True) 

cp3.rename(columns = {'id':'Female'}, inplace = True) 

cp4=pd.concat([cp2["current_status"],cp2["Male"] ,cp3["Female"]], axis=1).reset_index(drop=True, inplace=False)

cp4.style.background_gradient(cmap='terrain')
c4=cp4.set_index('current_status',inplace=True)  

#reseting the index and transpose the dataframe



c4=cp4.transpose().reset_index()
from plotly.offline import iplot



x = c4.index



trace1 = {

  'x': x,

  'y': c4.Deceased,

  'name': 'Deceased',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': c4.Hospitalized,

  'name': 'Hospitalized',

  'type': 'bar'

};







trace3 = {

  'x': x,

  'y': c4.Recovered,

  'name': 'Recovered',

  'type': 'bar'

};



data = [trace1, trace2,trace3];

layout = {

  'xaxis': {'title': 'Male-vs-Female'},

  'barmode': 'relative',

  'title': 'Gender-Wise-Disturbution of cases'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)

temp = cvd[["Date","Confirmed","Deaths","Cured"]]

temp['Date'] = temp['Date'].apply(pd.to_datetime, dayfirst=True)



date_wise_data = temp.groupby(["Date"]).sum().reset_index()
temp = date_wise_data.melt(id_vars="Date", value_vars=['Cured', 'Deaths', 'Confirmed'],var_name='Case', value_name='Count')

#temp.head()



cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801'

fig = px.area(temp, x="Date", y="Count", color='Case',title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()