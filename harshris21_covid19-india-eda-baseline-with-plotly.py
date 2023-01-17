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




path="/kaggle/input/covid19-in-india/"

def load_data(data):

    

    return pd.read_csv(path+data)



cvd=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

cvd.head()

age_group=load_data('AgeGroupDetails.csv')

ind_det=load_data('IndividualDetails.csv')  

#print(ind_det.head())

state_Test=load_data("StatewiseTestingDetails.csv")

ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')

#ICMR_details.head()

#IndividualDetails.csv

#cvd.head()

#cvd.columns

#cvd=cvd.rename(columns={'State/UnionTerritory':'STUT'}, inplace=True)

#print(cvd)

bed=load_data('HospitalBedsIndia.csv')

print(ind_det.current_status.unique())





ICMR_details.head()
state_details = pd.pivot_table(cvd, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')

state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],2)

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)

state_details = state_details.sort_values(by='Confirmed', ascending= False).reset_index(level=0)

state_details.style.background_gradient(cmap='plasma_r')




state_details.rename(columns={'State/UnionTerritory':'STUT'}, inplace=True)

state_details.head()







import plotly.express as px

df = px.data.tips()



fig = px.density_contour(state_details, x="Confirmed", y="Deaths")

fig.update_traces(contours_coloring="fill", contours_showlabels = True)

fig.show()



fig = px.density_contour(state_details, x="Confirmed", y="Cured")

fig.update_traces(contours_coloring="fill", contours_showlabels = True)

fig.show()
import plotly.graph_objs as go

#from chart_studio.plotly import iplot

import plotly



#import plotly.graph_objs as go

# Set notebook mode to work in offline

import plotly.offline as pyo

#from plotly.offline import iplot

# your code

pyo.init_notebook_mode()

x = state_details.STUT



trace1 = {

  'x': x,

  'y': state_details.Confirmed,

  'name': 'confirmed',

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

data = [trace1, trace2,trace3];

layout = {

  'xaxis': {'title': 'State-Data'},

  'barmode': 'relative',

  'title': 'Case Wise Disturbution'

};

fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig)

#plotly.offline.plot(fig, filename='gauge-meter-chart.html')




import plotly.express as px

df = px.data.wind()

fig = px.bar_polar(state_details, r=state_details.Confirmed, theta=state_details.Cured, color=state_details.STUT, template="plotly_dark",

            color_discrete_sequence= px.colors.sequential.Plasma_r)

fig.show()



locations = ind_det.groupby(['detected_state', 'detected_district', 'detected_city'])['government_id'].count().reset_index()

locations['country'] = 'India'

fig = px.treemap(locations, path=["country", "detected_state", "detected_district", "detected_city"], values="government_id", height=700,

           title='State ---> District --> City', color_discrete_sequence = px.colors.qualitative.Prism)



fig.data[0].textinfo = 'label+text+value+percent entry+percent root'

fig.show()
fig = go.Figure()

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



c4=cp4.set_index('current_status',inplace=True)  #reset the index and transpose the dataframe

c4=cp4.transpose().reset_index()
# prepare data frames

#df2014 = timesData[timesData.year == 2014].iloc[:3,:]

# import graph objects as "go"

import plotly.graph_objs as go

#from chart_studio.plotly import iplot

import plotly

from plotly.offline import iplot

# your code



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



temp = date_wise_data.melt(id_vars="Date", value_vars=['Cured', 'Deaths', 'Confirmed'],

               var_name='Case', value_name='Count')

#temp.head()

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801'

fig = px.area(temp, x="Date", y="Count", color='Case',title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')



fig = px.line(cvd, x='Date', y='Confirmed', title='Time Series with Rangeslider')



fig.update_xaxes(rangeslider_visible=True)

fig.show()
cvd.Confirmed.sum()
import plotly.graph_objects as go

import pandas as pd

fig.add_trace(go.Scatter(x=ICMR_details['DateTime'], y=ICMR_details['TotalSamplesTested'],

                    mode='lines',name='TotalSamplesTested'))



fig.add_trace(go.Scatter(x=ICMR_details['DateTime'], y=ICMR_details['TotalIndividualsTested'], 

                mode='lines',name='TotalIndividualsTested'))



fig.add_trace(go.Scatter(x=ICMR_details['DateTime'], y=ICMR_details['TotalPositiveCases'], 

                mode='lines',name='TotalPositiveCases'))



fig.update_layout(title_text='ICMR TEST conducted for COVID19',plot_bgcolor='rgb(225,230,255)')

fig.show()

fig.update_xaxes(rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')



fig = px.line(ICMR_details, x='DateTime', y='TotalSamplesTested', title='Time Series with Rangeslider')



fig.update_xaxes(rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')



fig = px.line(ICMR_details, x='DateTime', y='TotalIndividualsTested', title='Time Series with Rangeslider')



fig.update_xaxes(rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')



fig = px.line(ICMR_details, x='DateTime', y='TotalPositiveCases', title='Time Series with Rangeslider')



fig.update_xaxes(rangeslider_visible=True)

fig.show()


bed.rename(columns={'State/UT':'STUT'}, inplace=True)

bed.head()



bed["Total_bed"]=bed["NumPublicBeds_HMIS"]+bed["NumRuralBeds_NHP18"]+bed["NumUrbanBeds_NHP18"]

bed.tail()
bed1=bed[:-1]
import plotly.express as px

#data_canada = px.data.gapminder().query("country == 'Canada'")

fig = px.bar(bed[:-1], x='STUT', y='Total_bed', text='Total_bed')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',title="Disturbution of Beds in Hospital")

fig.show()
import plotly.graph_objs as go

#from chart_studio.plotly import iplot

import plotly

from plotly.offline import iplot

# your code



x = bed.STUT[:-1]



trace1 = {

  'x': x,

  'y': bed.NumPrimaryHealthCenters_HMIS,

  'name': 'Public Health Center',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': bed.NumCommunityHealthCenters_HMIS,

  'name': 'Community-hospitals',

  'type': 'bar'

};







trace3 = {

  'x': x,

  'y': bed.NumSubDistrictHospitals_HMIS,

  'name': 'sub-district',

  'type': 'bar'

};



trace4 = {

  'x': x,

  'y': bed.NumDistrictHospitals_HMIS,

  'name': 'District hospitals',

  'type': 'bar'

};





data = [trace1, trace2,trace3,trace4];

layout = {

  'xaxis': {'title': 'Statewise hospital Disturbution'},

  'barmode': 'relative',

  'title': 'Disturbution of Number of Hospitals'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
ind_det.head()


ind_det.rename(columns={'diagnosed_date':'date_announced'}, inplace=True)
days_to_status_change = ind_det[['date_announced', 'status_change_date', 'current_status']].dropna()



days_to_status_change['date_announced'] = days_to_status_change['date_announced'].apply(pd.to_datetime, dayfirst=True)

days_to_status_change['status_change_date'] = days_to_status_change['status_change_date'].apply(pd.to_datetime, dayfirst=True)





days_to_status_change = days_to_status_change[days_to_status_change['status_change_date'] != days_to_status_change['date_announced']]

days_to_status_change['days_to_status_change'] = days_to_status_change['status_change_date'] - days_to_status_change['date_announced']

days_to_status_change['days_to_status_change'] = days_to_status_change['days_to_status_change'].dt.days



days_to_recover = days_to_status_change[days_to_status_change['current_status']=='Recovered']

days_to_recover.head()
days_to_recover['days_to_status_change'].unique()
px.box(days_to_status_change, x="current_status", y="days_to_status_change", color='current_status')
import plotly.express as px



import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')



fig = px.line(df, x='Date', y='AAPL.High')

fig.show()