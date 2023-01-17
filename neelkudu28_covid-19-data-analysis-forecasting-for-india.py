import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from fbprophet import Prophet

!pip install plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

!pip install pyramid-arima

from pyramid.arima import auto_arima
covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head()
#Extracting India's data 

covid_india=covid[covid['Country/Region']=="India"]



#Extracting other countries for comparison of worst affected countries

covid_spain=covid[covid['Country/Region']=="Spain"]

covid_us=covid[covid['Country/Region']=="US"]

covid_italy=covid[covid['Country/Region']=="Italy"]

covid_iran=covid[covid['Country/Region']=="Iran"]

covid_france=covid[covid['Country/Region']=="France"]

covid_uk=covid[covid['Country/Region']=="UK"]

covid_br=covid[covid['Country/Region']=="Brazil"]

covid_russia=covid[covid['Country/Region']=="Russia"]



#Extracting data of neighbouring countries

covid_pak=covid[covid['Country/Region']=="Pakistan"]

covid_china=covid[covid['Country/Region']=="Mainland China"]

covid_afg=covid[covid['Country/Region']=="Afghanistan"]

covid_nepal=covid[covid['Country/Region']=="Nepal"]

covid_bhutan=covid[covid['Country/Region']=="Bhutan"]

covid_lanka=covid[covid["Country/Region"]=="Sri Lanka"]

covid_ban=covid[covid["Country/Region"]=="Bangladesh"]
#Converting the date into Datetime format

covid_india["ObservationDate"]=pd.to_datetime(covid_india["ObservationDate"])

covid_spain["ObservationDate"]=pd.to_datetime(covid_spain["ObservationDate"])

covid_us["ObservationDate"]=pd.to_datetime(covid_us["ObservationDate"])

covid_italy["ObservationDate"]=pd.to_datetime(covid_italy["ObservationDate"])

covid_iran["ObservationDate"]=pd.to_datetime(covid_iran["ObservationDate"])

covid_france["ObservationDate"]=pd.to_datetime(covid_france["ObservationDate"])

covid_uk["ObservationDate"]=pd.to_datetime(covid_uk["ObservationDate"])

covid_br["ObservationDate"]=pd.to_datetime(covid_br["ObservationDate"])

covid_russia["ObservationDate"]=pd.to_datetime(covid_russia["ObservationDate"])



covid_pak["ObservationDate"]=pd.to_datetime(covid_pak["ObservationDate"])

covid_china["ObservationDate"]=pd.to_datetime(covid_china["ObservationDate"])

covid_afg["ObservationDate"]=pd.to_datetime(covid_afg["ObservationDate"])

covid_nepal["ObservationDate"]=pd.to_datetime(covid_nepal["ObservationDate"])

covid_bhutan["ObservationDate"]=pd.to_datetime(covid_bhutan["ObservationDate"])

covid_lanka["ObservationDate"]=pd.to_datetime(covid_lanka["ObservationDate"])

covid_ban["ObservationDate"]=pd.to_datetime(covid_ban["ObservationDate"])
#Grouping the data based on the Date 

india_datewise=covid_india.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

spain_datewise=covid_spain.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

us_datewise=covid_us.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

italy_datewise=covid_italy.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

iran_datewise=covid_iran.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

france_datewise=covid_france.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

uk_datewise=covid_uk.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

brazil_datewise=covid_br.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

russia_datewise=covid_russia.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})



pak_datewise=covid_pak.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

china_datewise=covid_china.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

afg_datewise=covid_afg.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

nepal_datewise=covid_nepal.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

bhutan_datewise=covid_bhutan.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

lanka_datewise=covid_lanka.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

ban_datewise=covid_ban.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
#Adding week column to perfom weekly analysis further ahead

india_datewise["WeekofYear"]=india_datewise.index.weekofyear

spain_datewise["WeekofYear"]=spain_datewise.index.weekofyear

us_datewise["WeekofYear"]=us_datewise.index.weekofyear

italy_datewise["WeekofYear"]=italy_datewise.index.weekofyear

iran_datewise["WeekofYear"]=iran_datewise.index.weekofyear

france_datewise["WeekofYear"]=france_datewise.index.weekofyear

uk_datewise["WeekofYear"]=uk_datewise.index.weekofyear

brazil_datewise["WeekofYear"]=brazil_datewise.index.weekofyear

russia_datewise["WeekofYear"]=russia_datewise.index.weekofyear



pak_datewise["WeekofYear"]=pak_datewise.index.weekofyear

china_datewise["WeekofYear"]=china_datewise.index.weekofyear

afg_datewise["WeekofYear"]=afg_datewise.index.weekofyear

nepal_datewise["WeekofYear"]=nepal_datewise.index.weekofyear

bhutan_datewise["WeekofYear"]=bhutan_datewise.index.weekofyear

lanka_datewise["WeekofYear"]=lanka_datewise.index.weekofyear

ban_datewise["WeekofYear"]=ban_datewise.index.weekofyear
india_datewise["Days Since"]=(india_datewise.index-india_datewise.index[0])

india_datewise["Days Since"]=india_datewise["Days Since"].dt.days
No_Lockdown=covid_india[covid_india["ObservationDate"]<pd.to_datetime("2020-03-21")]

Lockdown_1=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-03-21"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-04-15"))]

Lockdown_2=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-04-15"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-05-04"))]

Lockdown_3=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-05-04"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-05-19"))]

Lockdown_4=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-05-19"))&(covid_india["ObservationDate"]<=pd.to_datetime("2020-05-31"))]

Unlock_1=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-06-01"))&(covid_india["ObservationDate"]<=pd.to_datetime("2020-06-30"))]

Unlock_2=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-07-01"))]



No_Lockdown_datewise=No_Lockdown.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_1_datewise=Lockdown_1.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_2_datewise=Lockdown_2.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_3_datewise=Lockdown_3.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Lockdown_4_datewise=Lockdown_4.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Unlock_1_datewise=Unlock_1.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

Unlock_2_datewise=Unlock_2.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])

grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]

grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])

grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
print("Number of Confirmed Cases",india_datewise["Confirmed"].iloc[-1])

print("Number of Recovered Cases",india_datewise["Recovered"].iloc[-1])

print("Number of Death Cases",india_datewise["Deaths"].iloc[-1])

print("Number of Active Cases",india_datewise["Confirmed"].iloc[-1]-india_datewise["Recovered"].iloc[-1]-india_datewise["Deaths"].iloc[-1])

print("Number of Closed Cases",india_datewise["Recovered"].iloc[-1]+india_datewise["Deaths"].iloc[-1])

print("Approximate Number of Confirmed Cases per day",round(india_datewise["Confirmed"].iloc[-1]/india_datewise.shape[0]))

print("Approximate Number of Recovered Cases per day",round(india_datewise["Recovered"].iloc[-1]/india_datewise.shape[0]))

print("Approximate Number of Death Cases per day",round(india_datewise["Deaths"].iloc[-1]/india_datewise.shape[0]))

print("Number of New Cofirmed Cases in last 24 hours are",india_datewise["Confirmed"].iloc[-1]-india_datewise["Confirmed"].iloc[-2])

print("Number of New Recoverd Cases in last 24 hours are",india_datewise["Recovered"].iloc[-1]-india_datewise["Recovered"].iloc[-2])

print("Number of New Death Cases in last 24 hours are",india_datewise["Deaths"].iloc[-1]-india_datewise["Deaths"].iloc[-2])
fig=px.bar(x=india_datewise.index,y=india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Active Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",)

fig.show()
fig=px.bar(x=india_datewise.index,y=india_datewise["Recovered"]+india_datewise["Deaths"])

fig.update_layout(title="Distribution of Number of Closed Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases")

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Growth of different types of cases in India",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
print('Mean Recovery Rate: ',((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).mean())

print('Mean Mortality Rate: ',((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).mean())

print('Median Recovery Rate: ',((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).median())

print('Median Mortality Rate: ',((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).median())



fig = make_subplots(rows=2, cols=1,

                   subplot_titles=("Recovery Rate", "Mortatlity Rate"))

fig.add_trace(

    go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]/india_datewise["Confirmed"])*100,

              name="Recovery Rate"),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(x=india_datewise.index, y=(india_datewise["Deaths"]/india_datewise["Confirmed"])*100,

              name="Mortality Rate"),

    row=2, col=1

)

fig.update_layout(height=1000,legend=dict(x=-0.1,y=1.2,traceorder="normal"))

fig.update_xaxes(title_text="Date", row=1, col=1)

fig.update_yaxes(title_text="Recovery Rate", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)

fig.update_yaxes(title_text="Mortality Rate", row=1, col=2)

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"]/india_datewise["Confirmed"].shift(),

                    mode='lines',

                    name='Growth Factor of Confirmed Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"]/india_datewise["Recovered"].shift(),

                    mode='lines',

                    name='Growth Factor of Recovered Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"]/india_datewise["Deaths"].shift(),

                    mode='lines',

                    name='Growth Factor of Death Cases'))

fig.update_layout(title="Datewise Growth Factor of different types of cases in India",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, 

                        y=(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"])/(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"]).shift(),

                    mode='lines',

                    name='Growth Factor of Active Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]+india_datewise["Deaths"])/(india_datewise["Recovered"]+india_datewise["Deaths"]).shift(),

                    mode='lines',

                    name='Growth Factor of Closed Cases'))

fig.update_layout(title="Datewise Growth Factor of Active and Closed cases in India",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().fillna(0),

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"].diff().fillna(0),

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().fillna(0),

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="Daily increase in different types of cases in India",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().rolling(window=7).mean(),

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"].diff().rolling(window=7).mean(),

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().rolling(window=7).mean().diff(),

                    mode='lines+markers',

                    name='Death Cases'))

fig.update_layout(title="7 Days Rolling mean of Confirmed, Recovered and Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"]).diff().rolling(window=7).mean(),

                    mode='lines+markers',

                    name='Active Cases'))

fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]+india_datewise["Deaths"]).diff().rolling(window=7).mean(),

                    mode='lines+markers',

                    name='Closed Cases'))

fig.update_layout(title="7 Days Rolling mean of Active and Closed Cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
week_num_india=[]

india_weekwise_confirmed=[]

india_weekwise_recovered=[]

india_weekwise_deaths=[]

w=1

for i in list(india_datewise["WeekofYear"].unique()):

    india_weekwise_confirmed.append(india_datewise[india_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    india_weekwise_recovered.append(india_datewise[india_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    india_weekwise_deaths.append(india_datewise[india_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_india.append(w)

    w=w+1
fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_confirmed,

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_recovered,

                    mode='lines+markers',

                    name='Weekly Growth of Recovered Cases'))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_deaths,

                    mode='lines+markers',

                    name='Weekly Growth of Death Cases'))

fig.update_layout(title="Weekly Growth of different types of Cases in India",

                 xaxis_title="Week Number",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
print("Average weekly increase in number of Confirmed Cases",round(pd.Series(india_weekwise_confirmed).diff().fillna(0).mean()))

print("Average weekly increase in number of Recovered Cases",round(pd.Series(india_weekwise_recovered).diff().fillna(0).mean()))

print("Average weekly increase in number of Death Cases",round(pd.Series(india_weekwise_deaths).diff().fillna(0).mean()))



fig = make_subplots(rows=1, cols=2)

fig.add_trace(

    go.Bar(x=week_num_india, y=pd.Series(india_weekwise_confirmed).diff().fillna(0),

          name="Weekly rise in number of Confirmed Cases"),

    row=1, col=1

)

fig.add_trace(

    go.Bar(x=week_num_india, y=pd.Series(india_weekwise_deaths).diff().fillna(0),

          name="Weekly rise in number of Death Cases"),

    row=1, col=2

)

fig.update_layout(title="India's Weekly increas in Number of Confirmed and Death Cases",

    font=dict(

        size=10,

    )

)

fig.update_layout(width=900,legend=dict(x=0,y=-0.5,traceorder="normal"))

fig.update_xaxes(title_text="Date", row=1, col=1)

fig.update_yaxes(title_text="Number of Cases", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)

fig.update_yaxes(title_text="Number of Cases", row=1, col=2)

fig.show()
cases=65

double_days=[]

C=[]

while(1):

    double_days.append(int(india_datewise[india_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<india_datewise["Confirmed"].max()):

        continue

    else:

        break

        

cases=65

tipling_days=[]

C1=[]

while(1):

    tipling_days.append(int(india_datewise[india_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C1.append(cases)

    cases=cases*3

    if(cases<india_datewise["Confirmed"].max()):

        continue

    else:

        break

        

india_doubling=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first case"])

india_doubling["Number of days required to Double the cases"]=india_doubling["Days since first case"].diff().fillna(india_doubling["Days since first case"].iloc[0])



india_tripling=pd.DataFrame(list(zip(C1,tipling_days)),columns=["No. of cases","Days since first case"])

india_tripling["Number of days required to Triple the cases"]=india_tripling["Days since first case"].diff().fillna(india_tripling["Days since first case"].iloc[0])



india_doubling.style.background_gradient(cmap='Reds')
india_tripling.style.background_gradient(cmap='Reds')
case_100k=5000

rise_100k=[]

C1=[]

while(1):

    rise_100k.append(int(india_datewise[india_datewise["Confirmed"]<=case_100k].iloc[[-1]]["Days Since"]))

    C1.append(case_100k)

    case_100k=case_100k+100000

    if(case_100k<india_datewise["Confirmed"].max()):

        continue

    else:

        break

rate_100k=pd.DataFrame(list(zip(C1,rise_100k)),columns=["No. of Cases","Days Since first Case"])

rate_100k["Days required for increase by 100K"]=rate_100k["Days Since first Case"].diff().fillna(rate_100k["Days Since first Case"].iloc[0])
fig=go.Figure()

fig.add_trace(go.Scatter(x=rate_100k["No. of Cases"], y=rate_100k["Days required for increase by 100K"],

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.update_layout(title="Number of Days required for increase in number of cases by 100K",

                 xaxis_title="Number of Cases",yaxis_title="Number of Days")

fig.show()
No_Lockdown_datewise["Active Cases"]=No_Lockdown_datewise["Confirmed"]-No_Lockdown_datewise["Recovered"]-No_Lockdown_datewise["Deaths"]

Lockdown_1_datewise["Active Cases"]=Lockdown_1_datewise["Confirmed"]-Lockdown_1_datewise["Recovered"]-Lockdown_1_datewise["Deaths"]

Lockdown_2_datewise["Active Cases"]=Lockdown_2_datewise["Confirmed"]-Lockdown_2_datewise["Recovered"]-Lockdown_2_datewise["Deaths"]

Lockdown_3_datewise["Active Cases"]=Lockdown_3_datewise["Confirmed"]-Lockdown_3_datewise["Recovered"]-Lockdown_3_datewise["Deaths"]

Lockdown_4_datewise["Active Cases"]=Lockdown_4_datewise["Confirmed"]-Lockdown_4_datewise["Recovered"]-Lockdown_4_datewise["Deaths"]

Unlock_1_datewise["Active Cases"]=Unlock_1_datewise["Confirmed"]-Unlock_1_datewise["Recovered"]-Unlock_1_datewise["Deaths"]

Unlock_2_datewise["Active Cases"]=Unlock_2_datewise["Confirmed"]-Unlock_2_datewise["Recovered"]-Unlock_2_datewise["Deaths"]





No_Lockdown_datewise["Days Since"]=(No_Lockdown_datewise.index-No_Lockdown_datewise.index.min()).days

Lockdown_1_datewise["Days Since"]=(Lockdown_1_datewise.index-Lockdown_1_datewise.index.min()).days

Lockdown_2_datewise["Days Since"]=(Lockdown_2_datewise.index-Lockdown_2_datewise.index.min()).days

Lockdown_3_datewise["Days Since"]=(Lockdown_3_datewise.index-Lockdown_3_datewise.index.min()).days

Lockdown_4_datewise["Days Since"]=(Lockdown_4_datewise.index-Lockdown_4_datewise.index.min()).days

Unlock_1_datewise["Days Since"]=(Unlock_1_datewise.index-Unlock_1_datewise.index.min()).days

Unlock_2_datewise["Days Since"]=(Unlock_2_datewise.index-Unlock_2_datewise.index.min()).days





cases=1

NL_doubling=[]

C=[]

while(1):

    NL_doubling.append(int(No_Lockdown_datewise[No_Lockdown_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<No_Lockdown_datewise["Confirmed"].max()):

        continue

    else:

        break

NL_Double_rate=pd.DataFrame(list(zip(C,NL_doubling)),columns=["No. of Cases","Days Since First Case"])

NL_Double_rate["Days required for Doubling"]=NL_Double_rate["Days Since First Case"].diff().fillna(NL_Double_rate["Days Since First Case"].iloc[0])



cases=Lockdown_1_datewise["Confirmed"].min()

L1_doubling=[]

C=[]

while(1):

    L1_doubling.append(int(Lockdown_1_datewise[Lockdown_1_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<Lockdown_1_datewise["Confirmed"].max()):

        continue

    else:

        break

L1_Double_rate=pd.DataFrame(list(zip(C,L1_doubling)),columns=["No. of Cases","Days Since Lockdown 1.0"])

L1_Double_rate["Days required for Doubling"]=L1_Double_rate["Days Since Lockdown 1.0"].diff().fillna(L1_Double_rate["Days Since Lockdown 1.0"].iloc[0])



cases=Lockdown_2_datewise["Confirmed"].min()

L2_doubling=[]

C=[]

while(1):

    L2_doubling.append(int(Lockdown_2_datewise[Lockdown_2_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<Lockdown_2_datewise["Confirmed"].max()):

        continue

    else:

        break

L2_Double_rate=pd.DataFrame(list(zip(C,L2_doubling)),columns=["No. of Cases","Days Since Lockdown 2.0"])

L2_Double_rate["Days required for Doubling"]=L2_Double_rate["Days Since Lockdown 2.0"].diff().fillna(L2_Double_rate["Days Since Lockdown 2.0"].iloc[0])



cases=Lockdown_3_datewise["Confirmed"].min()

L3_doubling=[]

C=[]

while(1):

    L3_doubling.append(int(Lockdown_3_datewise[Lockdown_3_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<Lockdown_3_datewise["Confirmed"].max()):

        continue

    else:

        break

L3_Double_rate=pd.DataFrame(list(zip(C,L3_doubling)),columns=["No. of Cases","Days Since Lockdown 3.0"])

L3_Double_rate["Days required for Doubling"]=L3_Double_rate["Days Since Lockdown 3.0"].diff().fillna(L3_Double_rate["Days Since Lockdown 3.0"].iloc[0])



cases=Lockdown_4_datewise["Confirmed"].min()

L4_doubling=[]

C=[]

while(1):

    L4_doubling.append(int(Lockdown_4_datewise[Lockdown_4_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<Lockdown_4_datewise["Confirmed"].max()):

        continue

    else:

        break

L4_Double_rate=pd.DataFrame(list(zip(C,L4_doubling)),columns=["No. of Cases","Days Since Lockdown 4.0"])

L4_Double_rate["Days required for Doubling"]=L4_Double_rate["Days Since Lockdown 4.0"].diff().fillna(L4_Double_rate["Days Since Lockdown 4.0"].iloc[0])



cases=Unlock_1_datewise["Confirmed"].min()

UL1_doubling=[]

C=[]

while(1):

    UL1_doubling.append(int(Unlock_1_datewise[Unlock_1_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<Unlock_1_datewise["Confirmed"].max()):

        continue

    else:

        break

UL1_Double_rate=pd.DataFrame(list(zip(C,UL1_doubling)),columns=["No. of Cases","Days Since Lockdown 4.0"])

UL1_Double_rate["Days required for Doubling"]=UL1_Double_rate["Days Since Lockdown 4.0"].diff().fillna(UL1_Double_rate["Days Since Lockdown 4.0"].iloc[0])
print("Average Active Cases growth rate in Lockdown 1.0: ",(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()).mean())

print("Median Active Cases growth rate in Lockdown 1.0: ",(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()).median())

print("Average Active Cases growth rate in Lockdown 2.0: ",(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()).mean())

print("Median Active Cases growth rate in Lockdown 2.0: ",(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()).median())

print("Average Active Cases growth rate in Lockdown 3.0: ",(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()).mean())

print("Median Active Cases growth rate in Lockdown 3.0: ",(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()).median())

print("Average Active Cases growth rate in Lockdown 4.0: ",(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()).mean())

print("Median Active Cases growth rate in Lockdown 4.0: ",(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()).median())

print("Average Active Cases growth rate in Unlock 1.0: ",(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()).mean())

print("Median Active Cases growth rate in Unlock 1.0: ",(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()).median())





fig=go.Figure()

fig.add_trace(go.Scatter(y=list(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()),

                    mode='lines+markers',

                    name='Growth Factor of Lockdown 1.0 Active Cases'))

fig.add_trace(go.Scatter(y=list(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()),

                    mode='lines+markers',

                    name='Growth Factor of Lockdown 2.0 Active Cases'))

fig.add_trace(go.Scatter(y=list(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()),

                    mode='lines+markers',

                    name='Growth Factor of Lockdown 3.0 Active Cases'))

fig.add_trace(go.Scatter(y=list(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()),

                    mode='lines+markers',

                    name='Growth Factor of Lockdown 4.0 Active Cases'))

fig.add_trace(go.Scatter(y=list(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()),

                    mode='lines+markers',

                    name='Growth Factor of Unlock 1.0 Active Cases'))

# fig.add_trace(go.Scatter(y=list(Unlock_2_datewise["Active Cases"]/Unlock_2_datewise["Active Cases"].shift()),

#                     mode='lines+markers',

#                     name='Growth Factor of Unlock 2.0 Active Cases'))

fig.update_layout(title="Lockdownwise Growth Factor of Active Cases in India",

                 xaxis_title="Date",yaxis_title="Growth Factor",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
NL_Double_rate.style.background_gradient(cmap='Reds')
L1_Double_rate.style.background_gradient(cmap='Reds')
L2_Double_rate.style.background_gradient(cmap='Reds')
L3_Double_rate.style.background_gradient(cmap='Reds')
n_countries=["Pakistan","Mainland China","Afghanistan","Nepal","Bhutan","Sri Lanka","Bangladesh","India"]

comp_data=pd.concat([pak_datewise.iloc[[-1]],china_datewise.iloc[[-1]],afg_datewise.iloc[[-1]],nepal_datewise.iloc[[-1]],

          bhutan_datewise.iloc[[-1]],lanka_datewise.iloc[[-1]],ban_datewise.iloc[[-1]],india_datewise.iloc[[-1]]])

comp_data.drop(["Days Since","WeekofYear"],1,inplace=True)

comp_data.index=n_countries

comp_data["Mortality"]=(comp_data["Deaths"]/comp_data["Confirmed"])*100

comp_data["Recovery"]=(comp_data["Recovered"]/comp_data["Confirmed"])*100

comp_data["Survival Probability"]=(1-(comp_data["Deaths"]/comp_data["Confirmed"]))*100

comp_data.sort_values(["Confirmed"],ascending=False)

comp_data.style.background_gradient(cmap='Reds').format("{:.2f}")
print("Pakistan reported it's first confirm case on: ",pak_datewise.index[0].date())

print("China reported it's first confirm case on: ",china_datewise.index[0].date())

print("Afghanistan reported it's first confirm case on: ",afg_datewise.index[0].date())

print("Nepal reported it's first confirm case on: ",nepal_datewise.index[0].date())

print("Bhutan reported it's first confirm case on: ",bhutan_datewise.index[0].date())

print("Sri Lanka reported it's first confirm case on: ",lanka_datewise.index[0].date())

print("Bangladesh reported it's first confirm case on: ",ban_datewise.index[0].date())

print("India reported it's first confirm case on: ",india_datewise.index[0].date())
print("Pakistan reported it's first death case on: ",pak_datewise[pak_datewise["Deaths"]>0].index[0].date())

print("China reported it's first death case on: ",china_datewise[china_datewise["Deaths"]>0].index[0].date())

print("Afghanistan reported it's first death case on: ",afg_datewise[afg_datewise["Deaths"]>0].index[0].date())

print("Nepal reported it's first death case on: ",nepal_datewise[nepal_datewise["Deaths"]>0].index[0].date())

print("Sri Lanka reported it's first death case on: ",lanka_datewise[lanka_datewise["Deaths"]>0].index[0].date())

print("Bangladesh reported it's first death case on: ",lanka_datewise[lanka_datewise["Deaths"]>0].index[0].date())

print("India reported it's first death case on: ",india_datewise[india_datewise["Deaths"]>0].index[0].date())
fig=go.Figure()

fig.add_trace(go.Scatter(x=pak_datewise.index, y=np.log(pak_datewise["Confirmed"]),

                    mode='lines',name="Pakistan"))

fig.add_trace(go.Scatter(x=china_datewise.index, y=np.log(china_datewise["Confirmed"]),

                    mode='lines',name="China"))

fig.add_trace(go.Scatter(x=afg_datewise.index, y=np.log(afg_datewise["Confirmed"]),

                    mode='lines',name="Afghanistan"))

fig.add_trace(go.Scatter(x=nepal_datewise.index, y=np.log(nepal_datewise["Confirmed"]),

                    mode='lines',name="Nepal"))

fig.add_trace(go.Scatter(x=bhutan_datewise.index, y=np.log(bhutan_datewise["Confirmed"]),

                    mode='lines',name="Bhutan"))

fig.add_trace(go.Scatter(x=lanka_datewise.index, y=np.log(lanka_datewise["Confirmed"]),

                    mode='lines',name="Sri-Lanka"))

fig.add_trace(go.Scatter(x=ban_datewise.index, y=np.log(ban_datewise["Confirmed"]),

                    mode='lines',name="Bangladesh"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=np.log(india_datewise["Confirmed"]),

                    mode='lines',name="India"))

fig.update_layout(title="Confirmed Cases plot for Neighbouring Countries of India (Logarithmic Scale)",

                  xaxis_title="Date",yaxis_title="Number of Cases (Log scale)",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
mean_mortality=[((pak_datewise["Deaths"]/pak_datewise["Confirmed"])*100).mean(),((china_datewise["Deaths"]/china_datewise["Confirmed"])*100).mean(),

               ((afg_datewise["Deaths"]/afg_datewise["Confirmed"])*100).mean(),((nepal_datewise["Deaths"]/nepal_datewise["Confirmed"])*100).mean(),

               ((bhutan_datewise["Deaths"]/bhutan_datewise["Confirmed"])*100).mean(),((lanka_datewise["Deaths"]/lanka_datewise["Confirmed"])*100).mean(),

               ((ban_datewise["Deaths"]/ban_datewise["Confirmed"])*100).mean(),((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).mean()]

mean_recovery=[((pak_datewise["Recovered"]/pak_datewise["Confirmed"])*100).mean(),((china_datewise["Recovered"]/china_datewise["Confirmed"])*100).mean(),

               ((afg_datewise["Recovered"]/afg_datewise["Confirmed"])*100).mean(),((nepal_datewise["Recovered"]/nepal_datewise["Confirmed"])*100).mean(),

               ((bhutan_datewise["Recovered"]/bhutan_datewise["Confirmed"])*100).mean(),((lanka_datewise["Recovered"]/lanka_datewise["Confirmed"])*100).mean(),

               ((ban_datewise["Recovered"]/ban_datewise["Confirmed"])*100).mean(),((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).mean()]



comp_data["Mean Mortality Rate"]=mean_mortality

comp_data["Mean Recovery Rate"]=mean_recovery
fig = make_subplots(rows=2, cols=1)

fig.add_trace(

    go.Bar(y=comp_data.index, x=comp_data["Mean Mortality Rate"],orientation='h'),

    row=1, col=1

)

fig.add_trace(

    go.Bar(y=comp_data.index, x=comp_data["Mean Recovery Rate"],orientation='h'),

    row=2, col=1

)

fig.update_layout(title="Mean Mortality and Recovery Rate of Neighbouring countries",

    font=dict(

        size=10,

    )

)

fig.update_layout(height=800)

fig.update_yaxes(title_text="Country Name", row=1, col=1)

fig.update_xaxes(title_text="Mortality Rate", row=1, col=1)

fig.update_yaxes(title_text="Country Name", row=2, col=1)

fig.update_xaxes(title_text="Recovery Rate", row=2, col=1)

fig.show()
print("Mean Mortality Rate of all Neighbouring Countries: ",comp_data["Mortality"].drop(comp_data.index[1],0).mean())

print("Median Mortality Rate of all Neighbouring Countries: ",comp_data["Mortality"].drop(comp_data.index[1],0).median())

print("Mortality Rate in India: ",comp_data.ix[1]["Mortality"])

print("Mean Mortality Rate in India: ",(india_datewise["Deaths"]/india_datewise["Confirmed"]).mean()*100)

print("Median Mortality Rate in India: ",(india_datewise["Deaths"]/india_datewise["Confirmed"]).median()*100)



fig = make_subplots(rows=3, cols=1)

fig.add_trace(

    go.Bar(y=comp_data.index, x=comp_data["Mortality"],orientation='h'),

    row=1, col=1

)

fig.add_trace(

    go.Bar(y=comp_data.index, x=comp_data["Recovery"],orientation='h'),

    row=2, col=1

)

fig.add_trace(

    go.Bar(y=comp_data.index, x=comp_data["Survival Probability"],orientation='h'),

    row=3, col=1

)

fig.update_layout(title="Mortality, Recovery and Survival Probability of Neighbouring countries",

    font=dict(

        size=10,

    )

)

fig.update_layout(height=900)

fig.update_yaxes(title_text="Country Name", row=1, col=1)

fig.update_xaxes(title_text="Mortality", row=1, col=1)

fig.update_yaxes(title_text="Country Name", row=2, col=1)

fig.update_xaxes(title_text="Recovery", row=2, col=1)

fig.update_yaxes(title_text="Country Name", row=3, col=1)

fig.update_xaxes(title_text="Survival Probability", row=3, col=1)

fig.show()
n_median_age=[23.5,38.7,18.6,25,28.6,34.1,27.5, 28.2]

n_tourist=[907000,59270000,0,753000,210000,2051000,303000,14570000]

n_gdp=[0.38,15.12,0.02,0.03,0.00,0.11,0.31,3.28]

area=[907132,9596961,652230,147181,38394,65610,147570,3287263]

population_density=[286.5,148,59.63,204.430,21.188,341.5,1265.036,450.419]

avg_weight=[58.976,60.555,56.935,50.476,51.142,50.421,49.591,52.943]

comp_data["Median Age"]=n_median_age

comp_data["Tourists"]=n_tourist

comp_data["GDP"]=n_gdp

comp_data["Area (square km)"]=area

comp_data["Population Density (per sq km)"]=population_density

comp_data["Average Weight"]=avg_weight

comp_data.style.background_gradient(cmap='Reds').format("{:.2f}")
req=comp_data[["Confirmed","Deaths","Recovered","Median Age","Tourists","GDP",

               "Area (square km)","Population Density (per sq km)","Average Weight"]]

plt.figure(figsize=(12,6))

mask = np.triu(np.ones_like(req.corr(), dtype=np.bool))

sns.heatmap(req.corr(),annot=True, mask=mask)
fig=go.Figure()

for country in n_countries:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country]["log_confirmed"], y=grouped_country.ix[country]["log_active"],

                    mode='lines',name=country))

fig.update_layout(height=600,title="COVID-19 Journey of India's Neighbouring countries",

                 xaxis_title="Confirmed Cases (Logrithmic Scale)",yaxis_title="Active Cases (Logarithmic Scale)",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in n_countries:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Confirmed"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(title="7 Days Rolling Average of Daily increase of Confirmed Cases for Neighbouring Countries",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in n_countries:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Deaths"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(title="7 Days Rolling Average of Daily increase of Death Cases for Neighbouring Countries",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig = px.pie(comp_data, values='Confirmed', names=comp_data.index, 

             title='Proportion of Confirmed Cases in India and among Neighbouring countries ')

fig.show()
fig = px.pie(comp_data, values='Recovered', names=comp_data.index, 

             title='Proportion of Recovered Cases in India and among Neighbouring countries ')

fig.show()
fig = px.pie(comp_data, values='Deaths', names=comp_data.index, 

             title='Proportion of Death Cases in India and among Neighbouring countries ')

fig.show()
pd.set_option('float_format', '{:f}'.format)

country_names=["Spain","US","Italy","Iran","France","UK","Brazil","Russia","India"]

country_data=pd.concat([spain_datewise.iloc[[-1]],us_datewise.iloc[[-1]],italy_datewise.iloc[[-1]],iran_datewise.iloc[[-1]],

                        france_datewise.iloc[[-1]],uk_datewise.iloc[[-1]],brazil_datewise.iloc[[-1]],russia_datewise.iloc[[-1]],

                        india_datewise.iloc[[-1]]])

country_data=country_data.drop(["Days Since","WeekofYear"],1)

country_data["Mortality"]=(country_data["Deaths"]/country_data["Confirmed"])*100

country_data["Recovery"]=(country_data["Recovered"]/country_data["Confirmed"])*100

country_data.index=country_names

country_data.style.background_gradient(cmap='Blues').format("{:.2f}")
max_confirm_india=india_datewise["Confirmed"].iloc[-1]

print("It took",spain_datewise[(spain_datewise["Confirmed"]>0)&(spain_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Spain to reach number of Confirmed Cases equivalent to India")

print("It took",us_datewise[(us_datewise["Confirmed"]>0)&(us_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in USA to reach number of Confirmed Cases equivalent to India")

print("It took",italy_datewise[(italy_datewise["Confirmed"]>0)&(italy_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Italy to reach number of Confirmed Cases equivalent to India")

print("It took",iran_datewise[(iran_datewise["Confirmed"]>0)&(iran_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Iran to reach number of Confirmed Cases equivalent to India")

print("It took",france_datewise[(france_datewise["Confirmed"]>0)&(france_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in France to reach number of Confirmed Cases equivalent to India")

print("It took",uk_datewise[(uk_datewise["Confirmed"]>0)&(uk_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in United Kingdom to reach number of Confirmed Cases equivalent to India")

print("It took",brazil_datewise[(brazil_datewise["Confirmed"]>0)&(brazil_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Brazil to reach number of Confirmed Cases equivalent to India")

print("It took",russia_datewise[(russia_datewise["Confirmed"]>0)&(russia_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Russia to reach number of Confirmed Cases equivalent to India")

print("It took",india_datewise[india_datewise["Confirmed"]>0].shape[0],"days in India to reach",max_confirm_india,"Confirmed Cases")



fig=go.Figure()

fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india].index, y=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Confirmed"]<=max_confirm_india].index, y=us_datewise[us_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india].index, y=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india].index, y=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="Iran"))

fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Confirmed"]<=max_confirm_india].index, y=france_datewise[france_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="France"))

fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india].index, y=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="United Kingdom"))

fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india].index, y=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india].index, y=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],

                    mode='lines',name="Russia"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],

                    mode='lines',name="India"))

fig.update_layout(title="Growth of Confirmed Cases with respect to India",

                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
max_deaths_india=india_datewise["Deaths"].iloc[-1]

print("It took",spain_datewise[(spain_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Spain to reach number of Deaths Cases equivalent to India")

print("It took",us_datewise[(us_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in USA to reach number of Deaths Cases equivalent to India")

print("It took",italy_datewise[(italy_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Italy to reach number of Deaths Cases equivalent to India")

print("It took",iran_datewise[(iran_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Iran to reach number of Deaths Cases equivalent to India")

print("It took",france_datewise[(france_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in France to reach number of Deaths Cases equivalent to India")

print("It took",uk_datewise[(uk_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in UK to reach number of Deaths Cases equivalent to India")

print("It took",brazil_datewise[(brazil_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Brazil to reach number of Deaths Cases equivalent to India")

print("It took",russia_datewise[(russia_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Russia to reach number of Deaths Cases equivalent to India")

print("It took",india_datewise.shape[0],"days in India to reach",max_deaths_india,"Deaths Cases")



fig=go.Figure()

fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india].index, y=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Deaths"]<=max_deaths_india].index, y=us_datewise[us_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india].index, y=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india].index, y=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="Iran"))

fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Deaths"]<=max_deaths_india].index, y=france_datewise[france_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="France"))

fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india].index, y=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="United Kingdom"))

fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india].index, y=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india].index, y=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india]["Deaths"],

                    mode='lines',name="Russia"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],

                    mode='lines',name="India"))

fig.update_layout(title="Growth of Death Cases with respect to India",

                 xaxis_title="Date",yaxis_title="Number of Death Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
max_recovered_india=india_datewise["Recovered"].iloc[-1]

print("It took",spain_datewise[(spain_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Spain to reach number of Recovered Cases equivalent to India")

print("It took",us_datewise[(us_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in USA to reach number of Recovered Cases equivalent to India")

print("It took",italy_datewise[(italy_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Italy to reach number of Recovered Cases equivalent to India")

print("It took",iran_datewise[(iran_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Iran to reach number of Recovered Cases equivalent to India")

print("It took",france_datewise[(france_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in France to reach number of Recovered Cases equivalent to India")

print("It took",uk_datewise[(uk_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in UK to reach number of Recovered Cases equivalent to India")

print("It took",brazil_datewise[(brazil_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Brazil to reach number of Recovered Cases equivalent to India")

print("It took",russia_datewise[(russia_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Russia to reach number of Recovered Cases equivalent to India")

print("It took",india_datewise.shape[0],"days in India to reach",max_recovered_india,"Recovered Cases")



fig=go.Figure()

fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Recovered"]<=max_recovered_india].index, y=spain_datewise[spain_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Recovered"]<=max_recovered_india].index, y=us_datewise[us_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Recovered"]<=max_recovered_india].index, y=italy_datewise[italy_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Recovered"]<=max_recovered_india].index, y=iran_datewise[iran_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="Iran"))

fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Recovered"]<=max_recovered_india].index, y=france_datewise[france_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="France"))

fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Recovered"]<=max_recovered_india].index, y=uk_datewise[uk_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="United Kingdom"))

fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Recovered"]<=max_recovered_india].index, y=brazil_datewise[brazil_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Recovered"]<=max_recovered_india].index, y=russia_datewise[russia_datewise["Recovered"]<=max_recovered_india]["Recovered"],

                    mode='lines',name="Russia"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"],

                    mode='lines',name="India"))

fig.update_layout(title="Growth of Recovered Cases with respect to India",

                 xaxis_title="Date",yaxis_title="Number of Recovered Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india].index, y=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Confirmed"]<=max_confirm_india].index, y=us_datewise[us_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india].index, y=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india].index, y=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Iran"))

fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Confirmed"]<=max_confirm_india].index, y=france_datewise[france_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="France"))

fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india].index, y=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="United Kingdom"))

fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india].index, y=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india].index, y=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),

                    mode='lines',name="Russia"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().fillna(0),

                    mode='lines',name="India"))

fig.update_layout(title="Daily Increase in Number of Confirmed Cases",

                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india].index, y=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="Spain"))

fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Deaths"]<=max_deaths_india].index, y=us_datewise[us_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="USA"))

fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india].index, y=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="Italy"))

fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india].index, y=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="Iran"))

fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Deaths"]<=max_deaths_india].index, y=france_datewise[france_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="France"))

fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india].index, y=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="United Kingdom"))

fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india].index, y=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="Brazil"))

fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india].index, y=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),

                    mode='lines',name="Russia"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().fillna(0),

                    mode='lines',name="India"))

fig.update_layout(title="Daily Increase in Number of Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Death Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
week_num_spain=[]

spain_weekwise_confirmed=[]

spain_weekwise_recovered=[]

spain_weekwise_deaths=[]

w=1

for i in list(spain_datewise["WeekofYear"].unique()):

    spain_weekwise_confirmed.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    spain_weekwise_recovered.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    spain_weekwise_deaths.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_spain.append(w)

    w=w+1



week_num_us=[]

us_weekwise_confirmed=[]

us_weekwise_recovered=[]

us_weekwise_deaths=[]

w=1

for i in list(us_datewise["WeekofYear"].unique()):

    us_weekwise_confirmed.append(us_datewise[us_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    us_weekwise_recovered.append(us_datewise[us_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    us_weekwise_deaths.append(us_datewise[us_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_us.append(w)

    w=w+1



week_num_italy=[]

italy_weekwise_confirmed=[]

italy_weekwise_recovered=[]

italy_weekwise_deaths=[]

w=1

for i in list(italy_datewise["WeekofYear"].unique()):

    italy_weekwise_confirmed.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    italy_weekwise_recovered.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    italy_weekwise_deaths.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_italy.append(w)

    w=w+1

    

week_num_iran=[]

iran_weekwise_confirmed=[]

iran_weekwise_recovered=[]

iran_weekwise_deaths=[]

w=1

for i in list(iran_datewise["WeekofYear"].unique()):

    iran_weekwise_confirmed.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    iran_weekwise_recovered.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    iran_weekwise_deaths.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_iran.append(w)

    w=w+1

    

week_num_france=[]

france_weekwise_confirmed=[]

france_weekwise_recovered=[]

france_weekwise_deaths=[]

w=1

for i in list(france_datewise["WeekofYear"].unique()):

    france_weekwise_confirmed.append(france_datewise[france_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    france_weekwise_recovered.append(france_datewise[france_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    france_weekwise_deaths.append(france_datewise[france_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_france.append(w)

    w=w+1

    

week_num_uk=[]

uk_weekwise_confirmed=[]

uk_weekwise_recovered=[]

uk_weekwise_deaths=[]

w=1

for i in list(uk_datewise["WeekofYear"].unique()):

    uk_weekwise_confirmed.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    uk_weekwise_recovered.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    uk_weekwise_deaths.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_uk.append(w)

    w=w+1

    

week_num_br=[]

br_weekwise_confirmed=[]

br_weekwise_recovered=[]

br_weekwise_deaths=[]

w=1

for i in list(brazil_datewise["WeekofYear"].unique()):

    br_weekwise_confirmed.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    br_weekwise_recovered.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    br_weekwise_deaths.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_br.append(w)

    w=w+1

    

week_num_rus=[]

rus_weekwise_confirmed=[]

rus_weekwise_recovered=[]

rus_weekwise_deaths=[]

w=1

for i in list(russia_datewise["WeekofYear"].unique()):

    rus_weekwise_confirmed.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    rus_weekwise_recovered.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    rus_weekwise_deaths.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_rus.append(w)

    w=w+1
fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_confirmed,

                    mode='lines+markers',name="Spain"))

fig.add_trace(go.Scatter(x=week_num_us, y=us_weekwise_confirmed,

                    mode='lines+markers',name="USA"))

fig.add_trace(go.Scatter(x=week_num_italy, y=italy_weekwise_confirmed,

                    mode='lines+markers',name="Italy"))

fig.add_trace(go.Scatter(x=week_num_iran, y=iran_weekwise_confirmed,

                    mode='lines+markers',name="Iran"))

fig.add_trace(go.Scatter(x=week_num_france, y=france_weekwise_confirmed,

                    mode='lines+markers',name="France"))

fig.add_trace(go.Scatter(x=week_num_uk, y=uk_weekwise_confirmed,

                    mode='lines+markers',name="United Kingdom"))

fig.add_trace(go.Scatter(x=week_num_br, y=br_weekwise_confirmed,

                    mode='lines+markers',name="Brazil"))

fig.add_trace(go.Scatter(x=week_num_rus, y=rus_weekwise_confirmed,

                    mode='lines+markers',name="Russia"))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_confirmed,

                    mode='lines+markers',name="India"))

fig.update_layout(title="Weekly Growth of Confirmed Cases",

                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_deaths,

                    mode='lines+markers',name="Spain"))

fig.add_trace(go.Scatter(x=week_num_us, y=us_weekwise_deaths,

                    mode='lines+markers',name="USA"))

fig.add_trace(go.Scatter(x=week_num_italy, y=italy_weekwise_deaths,

                    mode='lines+markers',name="Italy"))

fig.add_trace(go.Scatter(x=week_num_iran, y=iran_weekwise_deaths,

                    mode='lines+markers',name="Iran"))

fig.add_trace(go.Scatter(x=week_num_france, y=france_weekwise_deaths,

                    mode='lines+markers',name="France"))

fig.add_trace(go.Scatter(x=week_num_uk, y=uk_weekwise_deaths,

                    mode='lines+markers',name="United Kingdom"))

fig.add_trace(go.Scatter(x=week_num_br, y=br_weekwise_deaths,

                    mode='lines+markers',name="Brazil"))

fig.add_trace(go.Scatter(x=week_num_rus, y=rus_weekwise_deaths,

                    mode='lines+markers',name="Russia"))

fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_deaths,

                    mode='lines+markers',name="India"))

fig.update_layout(title="Weekly Growth of Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Death Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_spain, y=pd.Series(spain_weekwise_confirmed).diff().fillna(0),

                    mode='lines+markers',name="Spain"))

fig.add_trace(go.Scatter(x=week_num_us, y=pd.Series(us_weekwise_confirmed).diff().fillna(0),

                     mode='lines+markers',name="USA"))

fig.add_trace(go.Scatter(x=week_num_italy, y=pd.Series(italy_weekwise_confirmed).diff().fillna(0),

                    mode='lines+markers',name="Italy"))

fig.add_trace(go.Scatter(x=week_num_iran, y=pd.Series(iran_weekwise_confirmed).diff().fillna(0),

                    mode='lines+markers',name="Iran"))

fig.add_trace(go.Scatter(x=week_num_france, y=pd.Series(france_weekwise_confirmed).diff().fillna(0),

                    mode='lines+markers',name="France"))

fig.add_trace(go.Scatter(x=week_num_uk, y=pd.Series(uk_weekwise_confirmed).diff().fillna(0),

                     mode='lines+markers',name="United Kingdom"))

fig.add_trace(go.Scatter(x=week_num_br, y=pd.Series(br_weekwise_confirmed).diff().fillna(0),

                     mode='lines+markers',name="Brazil"))

fig.add_trace(go.Scatter(x=week_num_rus, y=pd.Series(rus_weekwise_confirmed).diff().fillna(0),

                     mode='lines+markers',name="Russia"))

fig.add_trace(go.Scatter(x=week_num_india, y=pd.Series(india_weekwise_confirmed).diff().fillna(0),

                     mode='lines+markers',name="India"))

fig.update_layout(title="Weekly Growth of Death Cases",

                 xaxis_title="Date",yaxis_title="Number of Death Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ac_median_age=[45.5,38.3,47.9,32.4,42,40.8,33.5,39.6,28.2]

ac_tourists=[75315000,76407000,52372000,4942000,82700000,35814000,6547000,24571000,14570000]

ac_weight=[70.556,81.928,69.205,67.608,66.782,75.795,66.093,71.418,52.943]

ac_gdp=[1.62,24.08,2.40,0.56,3.19,3.26,2.54,1.95,3.28]

ac_area=[505992,9833517,301339,1648195,640679,242495,8515767,17098246,3287263]

ac_pd=[93,34,200,51,123,280,25.43,8.58,414]

country_data["Median Age"]=ac_median_age

country_data["Tourists"]=ac_tourists

country_data["GDP"]=ac_gdp

country_data["Area (square km)"]=ac_area

country_data["Average Weight"]=ac_weight

country_data["Population Density (per sq km)"]=ac_pd

country_data.sort_values(["Confirmed"],ascending=False)

country_data.style.background_gradient(cmap='Blues').format("{:.2f}")
new_req=country_data[["Confirmed","Deaths","Recovered","Median Age","Tourists","Average Weight",

                     "GDP","Area (square km)","Population Density (per sq km)"]]

plt.figure(figsize=(10,5))

mask = np.triu(np.ones_like(new_req.corr(), dtype=np.bool))

sns.heatmap(new_req.corr(),annot=True, mask=mask)
fig=go.Figure()

for country in country_names:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country]["log_confirmed"], y=grouped_country.ix[country]["log_active"],

                    mode='lines',name=country))

fig.update_layout(height=600,title="COVID-19 Journey of some worst affected countries and India",

                 xaxis_title="Confirmed Cases (Logrithmic Scale)",yaxis_title="Active Cases (Logarithmic Scale)",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig=go.Figure()

for country in country_names:

    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Confirmed"].rolling(window=7).mean().diff(),

                    mode='lines',name=country))

fig.update_layout(height=600,title="7 Days Rolling Average of Daily increase of Confirmed Cases for Worst affected countries and India",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
fig = px.pie(country_data, values='Confirmed', names=country_data.index, 

             title='Proportion of Confirmed Cases in India and among Worst affected countries')

fig.show()
fig = px.pie(country_data, values='Recovered', names=country_data.index, 

             title='Proportion of Recovered Cases in India and among Worst affected countries')

fig.show()
fig = px.pie(country_data, values='Deaths', names=country_data.index, 

             title='Proportion of Death Cases in India and among Worst affected countries')

fig.show()
model_data=comp_data.drop(["Survival Probability","Mean Mortality Rate","Mean Recovery Rate"],1)

model_data=pd.concat([model_data,country_data])
X=model_data.drop(["Confirmed","Recovered","Deaths","Recovery","Mortality"],1)

y1=model_data["Confirmed"]

y2=model_data["Recovered"]

y3=model_data["Deaths"]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

k_best_confirmed=SelectKBest(score_func=f_regression,k='all')

k_best_confirmed.fit(X,y1)

k_best_recovered=SelectKBest(score_func=f_regression,k='all')

k_best_recovered.fit(X,y2)

k_best_deaths=SelectKBest(score_func=f_regression,k='all')

k_best_deaths.fit(X,y3)
fig = go.Figure(data=[go.Bar(name='Feature Importance for Confirmed Cases', x=k_best_confirmed.scores_, y=pd.Series(list(X)),orientation='h'),

    go.Bar(name='Feature Importance for Recovered Cases', x=k_best_recovered.scores_, y=pd.Series(list(X)),orientation='h'),

    go.Bar(name='Feature Importance for Death Cases', x=k_best_deaths.scores_, y=pd.Series(list(X)),orientation='h')])

fig.update_layout(barmode='group',width=900,legend=dict(x=0,y=-0.5,traceorder="normal"),

                 title="Feature Importance using Select K-Best")

fig.show()
train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

model_scores=[]
poly = PolynomialFeatures(degree = 6) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))

valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))

y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)

linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)

rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))

model_scores.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)                          
comp_data=poly.fit_transform(np.array(india_datewise["Days Since"]).reshape(-1,1))

plt.figure(figsize=(11,6))

predictions_poly=linreg.predict(comp_data)

fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=predictions_poly,

                    mode='lines',name="Polynomial Regression Best Fit",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_date=[]

new_prediction_poly=[]

for i in range(1,18):

    new_date.append(india_datewise.index[-1]+timedelta(days=i))

    new_date_poly=poly.fit_transform(np.array(india_datewise["Days Since"].max()+i).reshape(-1,1))

    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
model_predictions=pd.DataFrame(zip(new_date,new_prediction_poly),columns=["Date","Polynomial Regression Prediction"])

model_predictions.head()
train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
svm=SVR(C=0.01,degree=7,kernel='poly')
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),train_ml["Confirmed"])
prediction_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

rmse_svm=np.sqrt(mean_squared_error(prediction_svm,valid_ml["Confirmed"]))

model_scores.append(rmse_svm)

print("Root Mean Square Error for SVR Model: ",rmse_svm)
plt.figure(figsize=(11,6))

predictions=svm.predict(np.array(india_datewise["Days Since"]).reshape(-1,1))

fig=go.Figure()

fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=india_datewise.index, y=predictions,

                    mode='lines',name="Support Vector Machine Best fit Kernel",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
new_date=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(india_datewise.index[-1]+timedelta(days=i))

    new_prediction_svm.append(svm.predict(np.array(india_datewise["Days Since"].max()+i).reshape(-1,1))[0])
model_predictions["SVM Prediction"]=new_prediction_svm

model_predictions.head()
model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

y_pred=valid.copy()
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=0.3, smoothing_slope=1.2)
y_pred["Holt"]=holt.forecast(len(valid))

rmse_holt_linear=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"]))

model_scores.append(rmse_holt_linear)

print("Root Mean Square Error Holt's Linear Model: ",rmse_holt_linear)
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
holt_new_prediction=[]

for i in range(1,18):

    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])



model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction

model_predictions.head()
model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

y_pred=valid.copy()
es=ExponentialSmoothing(np.asarray(model_train['Confirmed']),seasonal_periods=15, trend='mul', seasonal='mul').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))

rmse_holt_winter=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"]))

model_scores.append(rmse_holt_winter)

print("Root Mean Square Error for Holt's Winter Model: ",rmse_holt_winter)
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt\'s Winter Model"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases Holt's Winter Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
holt_winter_new_prediction=[]

for i in range(1,18):

    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])

model_predictions["Holt's Winter Model Prediction"]=holt_winter_new_prediction

model_predictions.head()
model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_ar= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=3,max_q=0,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ar.fit(model_train["Confirmed"])
prediction_ar=model_ar.predict(len(valid))

y_pred["AR Model Prediction"]=prediction_ar
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))

print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["AR Model Prediction"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases AR Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
AR_model_new_prediction=[]

for i in range(1,18):

    AR_model_new_prediction.append(model_ar.predict(len(valid)+i)[-1])

model_predictions["AR Model Prediction"]=AR_model_new_prediction

model_predictions.head()
model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_ma= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=5,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ma.fit(model_train["Confirmed"])
prediction_ma=model_ma.predict(len(valid))

y_pred["MA Model Prediction"]=prediction_ma
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["MA Model Prediction"])))

print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["MA Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["MA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases MA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
MA_model_new_prediction=[]

for i in range(1,18):

    MA_model_new_prediction.append(model_ma.predict(len(valid)+i)[-1])

model_predictions["MA Model Prediction"]=MA_model_new_prediction

model_predictions.head()
model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]

valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]

y_pred=valid.copy()
model_arima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima.fit(model_train["Confirmed"])
prediction_arima=model_arima.predict(len(valid))

y_pred["ARIMA Model Prediction"]=prediction_arima
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["ARIMA Model Prediction"])))

print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["ARIMA Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ARIMA_model_new_prediction=[]

for i in range(1,18):

    ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+i)[-1])

model_predictions["ARIMA Model Prediction"]=ARIMA_model_new_prediction

model_predictions.head()
prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_confirmed=pd.DataFrame(zip(list(india_datewise.index),list(india_datewise["Confirmed"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)
forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_confirmed=forecast_c.copy()
confirmed_forecast=prophet_c.predict(forecast_c)
rmse_prophet=np.sqrt(mean_squared_error(india_datewise["Confirmed"],confirmed_forecast['yhat'].head(india_datewise.shape[0])))

model_scores.append(rmse_prophet)

print("Root Mean Squared Error for Prophet Model: ",rmse_prophet)
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))

model_predictions["Prophet's Upper Bound"]=list(confirmed_forecast["yhat_upper"].tail(17))

model_predictions.head()
models=["Polynomial Regression","Support Vector Machine Regresssor","Holt's Linear Model",

       "Holt's Winter Model","Auto Regressive Model (AR)", "Moving Average Model (MA)","ARIMA Model","Facebook's Prophet Model"]
model_evaluation=pd.DataFrame(list(zip(models,model_scores)),columns=["Model Name","Root Mean Squared Error"])

model_evaluation=model_evaluation.sort_values(["Root Mean Squared Error"])

model_evaluation.style.background_gradient(cmap='Reds')
model_predictions["Average of Predictions Models"]=model_predictions.mean(axis=1)

show_predictions=model_predictions.head()

show_predictions