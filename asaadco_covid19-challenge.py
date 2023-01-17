# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Case = pd.read_csv("../input/coronavirusdataset/Case.csv")

PatientInfo = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

PatientRoute = pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

Region = pd.read_csv("../input/coronavirusdataset/Region.csv")

SearchTrend = pd.read_csv("../input/coronavirusdataset/SearchTrend.csv")

Time = pd.read_csv("../input/coronavirusdataset/Time.csv")

TimeAge = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

TimeGender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")

TimeProvince = pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

confirmed = pd.read_excel("../input/covid19-coronavirus/time_series_covid19_confirmed.xlsx")

deaths = pd.read_excel("../input/covid19-coronavirus/time_series_covid19_deaths.xlsx")

recovered = pd.read_excel("../input/covid19-coronavirus/time_series_covid19_recovered.xlsx")
china_cases = confirmed.loc[confirmed["Country/Region"]=="China"].sum().reset_index()

china_cases = china_cases.iloc[4:]



saudiarabia_cases = confirmed.loc[confirmed["Country/Region"]=="Saudi Arabia"].sum().reset_index()

saudiarabia_cases = saudiarabia_cases.iloc[4:]





unitedKingdom_cases = confirmed.loc[confirmed["Country/Region"]=="United Kingdom"].sum().reset_index()

unitedKingdom_cases = unitedKingdom_cases.iloc[4:]



southkorea_cases = confirmed.loc[confirmed["Country/Region"]=="Korea, South"].sum().reset_index()

southkorea_cases = southkorea_cases.iloc[4:]



bahrain_cases = confirmed.loc[confirmed["Country/Region"]=="Bahrain"].sum().reset_index()

bahrain_cases = bahrain_cases.iloc[4:]



kuwait_cases = confirmed.loc[confirmed["Country/Region"]=="Kuwait"].sum().reset_index()

kuwait_cases = kuwait_cases.iloc[4:]



timeline = go.Figure()



timeline.add_trace(go.Scatter(x=china_cases["index"], y=china_cases[0], name="Cases in China",

                          line_color='red'))



timeline.add_trace(go.Scatter(x=bahrain_cases["index"], y=bahrain_cases[0], name="Cases in Bahrain",

                          line_color='deepskyblue'))



timeline.add_trace(go.Scatter(x=southkorea_cases["index"], y=southkorea_cases[0], name="Cases in South Korea",

                          line_color='purple'))



timeline.add_trace(go.Scatter(x=kuwait_cases["index"], y=kuwait_cases[0], name="Cases in Kuwait",

                          line_color='darkorange'))



timeline.add_trace(go.Scatter(x=saudiarabia_cases["index"], y=saudiarabia_cases[0], name="Cases in Saudi",

                          line_color='green'))



timeline.add_trace(go.Scatter(x=unitedKingdom_cases["index"], y=unitedKingdom_cases[0], name="Cases in UK",

                          line_color='blue'))



timeline.update_layout(title_text='Spread of Corona over a period of Time', yaxis_type = "log",

                  xaxis_rangeslider_visible=True)

timeline.show()
china_cases = deaths.loc[confirmed["Country/Region"]=="China"].sum().reset_index()

china_cases = china_cases.iloc[4:]



saudiarabia_cases = deaths.loc[confirmed["Country/Region"]=="Saudi Arabia"].sum().reset_index()

saudiarabia_cases = saudiarabia_cases.iloc[4:]





unitedKingdom_cases = deaths.loc[confirmed["Country/Region"]=="United Kingdom"].sum().reset_index()

unitedKingdom_cases = unitedKingdom_cases.iloc[4:]



southkorea_cases = deaths.loc[confirmed["Country/Region"]=="Korea, South"].sum().reset_index()

southkorea_cases = southkorea_cases.iloc[4:]



bahrain_cases = deaths.loc[confirmed["Country/Region"]=="Bahrain"].sum().reset_index()

bahrain_cases = bahrain_cases.iloc[4:]



kuwait_cases = deaths.loc[confirmed["Country/Region"]=="Kuwait"].sum().reset_index()

kuwait_cases = kuwait_cases.iloc[4:]



timeline = go.Figure()



timeline.add_trace(go.Scatter(x=china_cases["index"], y=china_cases[0], name="Cases in China",

                          line_color='red'))



timeline.add_trace(go.Scatter(x=bahrain_cases["index"], y=bahrain_cases[0], name="Cases in Bahrain",

                          line_color='deepskyblue'))



timeline.add_trace(go.Scatter(x=southkorea_cases["index"], y=southkorea_cases[0], name="Cases in South Korea",

                          line_color='purple'))



timeline.add_trace(go.Scatter(x=kuwait_cases["index"], y=kuwait_cases[0], name="Cases in Kuwait",

                          line_color='darkorange'))



timeline.add_trace(go.Scatter(x=saudiarabia_cases["index"], y=saudiarabia_cases[0], name="Cases in Saudi",

                          line_color='green'))



timeline.add_trace(go.Scatter(x=unitedKingdom_cases["index"], y=unitedKingdom_cases[0], name="Cases in UK",

                          line_color='blue'))



timeline.update_layout(title_text='Deaths over a period of Time', yaxis_type = "log",

                  xaxis_rangeslider_visible=True)

timeline.show()
china_cases = recovered.loc[confirmed["Country/Region"]=="China"].sum().reset_index()

china_cases = china_cases.iloc[4:]



saudiarabia_cases = recovered.loc[confirmed["Country/Region"]=="Saudi Arabia"].sum().reset_index()

saudiarabia_cases = saudiarabia_cases.iloc[4:]





unitedKingdom_cases = recovered.loc[confirmed["Country/Region"]=="United Kingdom"].sum().reset_index()

unitedKingdom_cases = unitedKingdom_cases.iloc[4:]



southkorea_cases = recovered.loc[confirmed["Country/Region"]=="Korea, South"].sum().reset_index()

southkorea_cases = southkorea_cases.iloc[4:]



bahrain_cases = recovered.loc[confirmed["Country/Region"]=="Bahrain"].sum().reset_index()

bahrain_cases = bahrain_cases.iloc[4:]



kuwait_cases = recovered.loc[confirmed["Country/Region"]=="Kuwait"].sum().reset_index()

kuwait_cases = kuwait_cases.iloc[4:]



timeline = go.Figure()



timeline.add_trace(go.Scatter(x=china_cases["index"], y=china_cases[0], name="Cases in China",

                          line_color='red'))



timeline.add_trace(go.Scatter(x=bahrain_cases["index"], y=bahrain_cases[0], name="Cases in Bahrain",

                          line_color='deepskyblue'))



timeline.add_trace(go.Scatter(x=southkorea_cases["index"], y=southkorea_cases[0], name="Cases in South Korea",

                          line_color='purple'))



timeline.add_trace(go.Scatter(x=kuwait_cases["index"], y=kuwait_cases[0], name="Cases in Kuwait",

                          line_color='darkorange'))



timeline.add_trace(go.Scatter(x=saudiarabia_cases["index"], y=saudiarabia_cases[0], name="Cases in Saudi",

                          line_color='green'))



timeline.add_trace(go.Scatter(x=unitedKingdom_cases["index"], y=unitedKingdom_cases[0], name="Cases in UK",

                          line_color='blue'))



timeline.update_layout(title_text='Recovery over a period of Time', yaxis_type = "log",

                  xaxis_rangeslider_visible=True)

timeline.show()
Time['date'] = pd.to_datetime(Time['date'])



fig1 = go.Figure() # Create a fig model

# Add a trace for each column you want presented. We are using Scatter

fig1.add_trace(go.Scatter(x=Time['date'], y=Time['confirmed'], fill='tozeroy', name='confirmed'))

fig1.add_trace(go.Scatter(x=Time['date'], y=Time['released'], fill='tozeroy', name='released'))

fig1.add_trace(go.Scatter(x=Time['date'], y=Time['deceased'], fill='tozeroy', name='deceased'))



# Make it look nice..

fig1.update_layout(

    title = "Confirmed, Released and Deceased over Time",

    yaxis_title = "Number of Cases",

    font = dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

)

py.iplot(fig1)
fig2 = go.Figure() # Create a fig model

# Add a trace for each column you want presented. We are using Scatter

fig2.add_trace(go.Scatter(x=Time['date'], y=Time['released']/Time['deceased'], fill='tozeroy', name='released'))



# Make it look nice..

fig2.update_layout(

    title = "Released/Deceased Ratio over Time",

    yaxis_title = "Number of Cases",

    font = dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

)

py.iplot(fig2)
PatientInfo['Age'] = 2020-PatientInfo['birth_year']

fig3 = px.histogram(PatientInfo[PatientInfo['state']=='deceased'],x="Age",marginal="box",nbins=20)

fig3.update_layout(

    title = "number of deceased by age ",

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

py.iplot(fig3)



fig4 = px.histogram(PatientInfo[PatientInfo['state']=='released'],x="Age",marginal="box",nbins=20)

fig4.update_layout(

    title = "number of released by age ",

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

py.iplot(fig4)
PatientInfo['Age'] = 2020-PatientInfo['birth_year']

fig5 = px.histogram(PatientInfo[PatientInfo['state']=='isolated'],x="Age",marginal="box",nbins=20)

fig5.update_layout(

    title = "age distibution of isolated cases ",

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

py.iplot(fig5)


SaudiCity = pd.read_csv("../input/saudicovid19/saudi_cityy.csv" )





temp = SaudiCity.sort_values('Total', ascending=True)

#state_order = SaudiCity['City']



fig = px.bar(temp,x="Total", y="city ", color='Total', title='Total Cases per City in March', orientation='h', width=800,color_discrete_sequence=px.colors.qualitative.Vivid)

fig.show()


