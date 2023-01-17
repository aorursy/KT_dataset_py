import numpy as np

import pandas as pd

import plotly

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.offline as py

import matplotlib.pyplot as plt

import folium

import warnings

py.init_notebook_mode()

warnings.filterwarnings("ignore")



from sklearn.metrics import r2_score

from scipy.optimize import curve_fit
#load data

patient_path = "../input/coronavirusdataset/PatientInfo.csv"

time_path = "../input/coronavirusdataset/Time.csv"

route_path = "../input/coronavirusdataset/PatientRoute.csv"



df_route = pd.read_csv(route_path)

df_all_cases = pd.read_csv(time_path)

df_patients = pd.read_csv(patient_path)






df_all_cases['date'] = pd.to_datetime(df_all_cases['date'])



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['test'], fill='tozeroy',name='total tests')) # fill down to xaxis

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['negative'], fill='tozeroy',name='negative test')) # fill down to xaxis

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['confirmed'], fill='tozeroy',name='positive test')) # fill down to xaxis

fig.update_layout(

    title = "Covid19 tests",

    #xaxis_range = [0,5.2],

    #yaxis_range = [0,3],

    yaxis_title="number of cases",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

)

py.iplot(fig)







fig = go.Figure()

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['released'], fill='tozeroy',name='released')) # fill down to xaxis

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['deceased'], fill='tozeroy',name='deceased')) # fill down to xaxis

fig.update_layout(

    title = "Released and deceased over time",

    #xaxis_range = [0,5.2],

    #yaxis_range = [0,3],

    yaxis_title="number of cases",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

)

py.iplot(fig)



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_all_cases['date'], y=np.round(100*df_all_cases['deceased']/df_all_cases['released'],2), fill='tozeroy',name='ratio')) # fill down to xaxis

fig.update_layout(

    title = "Ratio of deceased/released",

    #xaxis_range = [0,5.2],

    #yaxis_range = [0,3],

    yaxis_title="deceased/released %",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

)

py.iplot(fig)





















df_patients['Age'] = None

for i in range(df_patients.shape[0]):

    if df_patients.birth_year.index[i] in df_patients[df_patients.birth_year.notna()].index:

        if  df_patients.birth_year.iloc[i] !=' ':

            df_patients['Age'].iloc[i] = 2020 - float(df_patients.birth_year.iloc[i])



df_recovered = df_patients[df_patients['state']=='released']

df_deceased = df_patients[df_patients['state']=='deceased']
df_patients




fig = px.pie( values=df_patients.groupby(['infection_case']).size().values,names=df_patients.groupby(['infection_case']).size().index)

fig.update_layout(

    title = "Possible infection reason",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df_patients[df_patients.Age.notna()],x="Age",marginal="box",nbins=20)

fig.update_layout(

    title = "number of confirmed cases by age group",

    xaxis_title="Age",

    yaxis_title="number of cases",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)





fig = px.pie( values=df_patients.groupby(['sex']).size().values,names=df_patients.groupby(['sex']).size().index)

fig.update_layout(

    title = "Sex distribuition of confirmed cases",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)



df_patients_aux = df_patients[df_patients.Age.notna()]

df_patients_aux=df_patients_aux[df_patients_aux.sex.notna()]

#df_patients_aux=df_patients_aux.sex.notna()

fig = px.histogram(df_patients_aux,x="Age",color="sex",marginal="box",opacity=1,nbins=20)

fig.update_layout(

    title = "number of confirmed cases by age group and sex",

    xaxis_title="Age",

    yaxis_title="number of cases",

    barmode="group",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)







df_deceased_and_recovered = pd.concat([df_deceased,df_recovered])

fig = px.histogram(df_deceased_and_recovered,x="Age",color="state",marginal="box",nbins=10)

fig.update_layout(

    title = "Recovered and deceased patients by age group",

    xaxis_title="Age",

    yaxis_title="number of cases",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    bargap=0.2,

    barmode="group",

    xaxis_range = [0,100],

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)


df_deceased.drop(index=df_deceased[df_deceased.Age.isna()].index,inplace=True)

df_deceased.drop(index=df_deceased[df_deceased.sex.isna()].index,inplace=True)





df_recovered.drop(index=df_recovered[df_recovered.Age.isna()].index,inplace=True)

df_recovered.drop(index=df_recovered[df_recovered.sex.isna()].index,inplace=True)





fig = px.histogram(df_recovered,x="Age",color="sex",marginal="box",nbins=10)

fig.update_layout(

    title = "recovered patients by age and sex",

    xaxis_title="Age",

    yaxis_title="number of cases",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    bargap=0.2,

    barmode="group",

    xaxis_range = [0,100],

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)





fig = px.histogram(df_deceased,x="Age",color="sex",marginal="box",nbins=10)

fig.update_layout(

    title = "deceased patients by age and sex",

    xaxis_title="Age",

    yaxis_title="number of cases",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    bargap=0.2,

    barmode="group",

    xaxis_range = [0,100],

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)
df_deceased_and_recovered['time_lenght_to_recover_or_dead']= None

for i in range(df_deceased_and_recovered.shape[0]):

    if df_deceased_and_recovered['state'].iloc[i] == 'deceased':

        df_deceased_and_recovered['time_lenght_to_recover_or_dead'].iloc[i] = (pd.to_datetime(df_deceased_and_recovered['deceased_date'].iloc[i])- pd.to_datetime(df_deceased_and_recovered['confirmed_date'].iloc[i])).days

    if df_deceased_and_recovered['state'].iloc[i] == 'released':

        df_deceased_and_recovered['time_lenght_to_recover_or_dead'].iloc[i] = ( pd.to_datetime(df_deceased_and_recovered['released_date'].iloc[i]) - pd.to_datetime(df_deceased_and_recovered['confirmed_date'].iloc[i])).days

     





fig = px.histogram(df_deceased_and_recovered,x="time_lenght_to_recover_or_dead",color="state",marginal="box",nbins=10)

fig.update_layout(

    title = "Time do recovery/dead after confirmatition",

    xaxis_title="Days after confirmation",

    yaxis_title="number of cases",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = -10,

        dtick = 5),

    bargap=0.2,

    barmode="group",

    xaxis_range = [-5,40],

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)