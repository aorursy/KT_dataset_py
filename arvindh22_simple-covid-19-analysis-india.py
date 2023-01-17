import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import datetime as dt

from datetime import timedelta



from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score, make_scorer

from sklearn.preprocessing import PolynomialFeatures



import scipy.cluster.hierarchy as sch



import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

from plotly.subplots import make_subplots

py.init_notebook_mode(connected= True)





import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/covid19-indian-statewise-data/COVID-19 State wise Cases.csv")
data.head()
data['Date']= pd.to_datetime(data['Date'])

data.head(5)
print("Dataset Description")

print("Earliest Entry: ",data['Date'].min())

print("Last Entry:    ",data['Date'].max())

print("Total Days:    ",(data['Date'].max() - data['Date'].min()))
#statewise cases 12-03-2020 to 10-10-2020

statewise= data.groupby(['Region','Date']).agg({"Confirmed Cases":'sum',"Cured/Discharged":'sum',"Death":'sum', 'Active Cases':'sum'})

statewise
statecases=data.groupby('Region')['Confirmed Cases','Active Cases','Death','Cured/Discharged'].max().reset_index()

statecases.sort_values('Confirmed Cases', ascending= False, inplace =True)

statecases
activecases=statecases['Active Cases'].sum()

cured=statecases['Cured/Discharged'].sum()

death=statecases['Death'].sum()



labels=['Active Cases','Cured/Discharged','Death']

values=[activecases,cured,death]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(title=" Pie chart of Different types of cases")

fig.show() 
#datawise cases from 12-03-2020 to 10-10-2020

datewise= data.groupby(['Date']).agg({"Confirmed Cases":'sum',"Cured/Discharged":'sum',"Death":'sum',"Active Cases":'sum'})

datewise.tail()
print("No. of States/UT suffering: ",len(data['Region'].unique()))

print("Total Confirmed Cases in India:  ",datewise['Confirmed Cases'].iloc[-1])

print("Total Recovered Cases in India:  ",datewise['Cured/Discharged'].iloc[-1])

print("Total Death Cases in India:  ",datewise['Death'].iloc[-1])
fig= go.Figure()



fig.add_trace(go.Scatter(x=datewise.index, y=datewise['Confirmed Cases'], mode='lines+markers', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise['Cured/Discharged'], mode='lines+markers', name='Cured Cases'))

fig.add_trace(go.Scatter(x=datewise.index, y=datewise['Death'], mode='lines+markers', name='Death Cases'))



fig.update_layout(title=" Growth of Different types of cases", xaxis_title="Date", yaxis_title="Number of Cases", legend= dict(x=0, y=1, traceorder="normal"))

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Maharashtra","Andra Pradesh","Karnataka",

"Tamil Nadu"])



#Maharashtra

Maharashtra_active = statecases['Active Cases'].iloc[0]

Maharashtra_cured = statecases['Cured/Discharged'].iloc[0]

Maharashtra_death = statecases['Death'].iloc[0]



labels=['Active Cases','Cured/Discharged','Death']

values_maharashtra=[Maharashtra_active,Maharashtra_cured,Maharashtra_death]

fig.add_trace(go.Pie(labels=labels, values=values_maharashtra), 1, 1)



#Andrapradesh

ap_active = statecases['Active Cases'].iloc[1]

ap_cured = statecases['Cured/Discharged'].iloc[1]

ap_death = statecases['Death'].iloc[1]



labels=['Active Cases','Cured/Discharged','Death']

values_ap=[ap_active,ap_cured,ap_death]

fig.add_trace(go.Pie(labels=labels, values=values_ap), 1, 2)



#Karnataka

karnataka_active = statecases['Active Cases'].iloc[2]

karnataka_cured = statecases['Cured/Discharged'].iloc[2]

karnataka_death = statecases['Death'].iloc[2]



labels=['Active Cases','Cured/Discharged','Death']

values_karnataka=[karnataka_active,karnataka_cured,karnataka_death]

fig.add_trace(go.Pie(labels=labels, values=values_karnataka), 2, 1)



#Tamilnadu

tn_active = statecases['Active Cases'].iloc[3]

tn_cured = statecases['Cured/Discharged'].iloc[3]

tn_death = statecases['Death'].iloc[3]



labels=['Active Cases','Cured/Discharged','Death']

values_tn=[tn_active,tn_cured,tn_death]

fig.add_trace(go.Pie(labels=labels, values=values_tn), 2, 2)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Uttar Pradesh","Delhi","West bengal","Kerela"])



#Uttarpradesh

up_active = statecases['Active Cases'].iloc[4]

up_cured = statecases['Cured/Discharged'].iloc[4]

up_death = statecases['Death'].iloc[4]



labels=['Active Cases','Cured/Discharged','Death']

values_up=[up_active,up_cured,up_death]

fig.add_trace(go.Pie(labels=labels, values=values_up), 1, 1)



#Delhi

delhi_active = statecases['Active Cases'].iloc[5]

delhi_cured = statecases['Cured/Discharged'].iloc[5]

delhi_death = statecases['Death'].iloc[5]



labels=['Active Cases','Cured/Discharged','Death']

values_delhi=[delhi_active,delhi_cured,delhi_death]

fig.add_trace(go.Pie(labels=labels, values=values_delhi), 1, 2)



#west bengal

wb_active = statecases['Active Cases'].iloc[6]

wb_cured = statecases['Cured/Discharged'].iloc[6]

wb_death = statecases['Death'].iloc[6]



labels=['Active Cases','Cured/Discharged','Death']

values_wb=[wb_active,wb_cured,wb_death]

fig.add_trace(go.Pie(labels=labels, values=values_wb), 2, 1)



#kerela

kerela_active = statecases['Active Cases'].iloc[7]

kerela_cured = statecases['Cured/Discharged'].iloc[7]

kerela_death = statecases['Death'].iloc[7]



labels=['Active Cases','Cured/Discharged','Death']

values_kerela=[kerela_active,kerela_cured,kerela_death]

fig.add_trace(go.Pie(labels=labels, values=values_kerela), 2, 2)





fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Odisha","Telangana","Bihar","Assam"])



#odisha

odisha_active = statecases['Active Cases'].iloc[8]

odisha_cured = statecases['Cured/Discharged'].iloc[8]

odisha_death = statecases['Death'].iloc[8]



labels=['Active Cases','Cured/Discharged','Death']

values_odisha=[odisha_active,odisha_cured,odisha_death]

fig.add_trace(go.Pie(labels=labels, values=values_odisha), 1, 1)



#Telangana

telangana_active = statecases['Active Cases'].iloc[9]

telangana_cured = statecases['Cured/Discharged'].iloc[9]

telangana_death = statecases['Death'].iloc[9]



labels=['Active Cases','Cured/Discharged','Death']

values_telangana=[telangana_active,telangana_cured,telangana_death]

fig.add_trace(go.Pie(labels=labels, values=values_telangana), 1, 2)



#Bihar

bihar_active = statecases['Active Cases'].iloc[10]

bihar_cured = statecases['Cured/Discharged'].iloc[10]

bihar_death = statecases['Death'].iloc[10]



labels=['Active Cases','Cured/Discharged','Death']

values_bihar=[bihar_active,bihar_cured,bihar_death]

fig.add_trace(go.Pie(labels=labels, values=values_bihar), 2, 1)



#Assam

assam_active = statecases['Active Cases'].iloc[11]

assam_cured = statecases['Cured/Discharged'].iloc[11]

assam_death = statecases['Death'].iloc[11]



labels=['Active Cases','Cured/Discharged','Death']

values_assam=[assam_active,assam_cured,assam_death]

fig.add_trace(go.Pie(labels=labels, values=values_assam), 2, 2)





fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Rajasthan","Gujarat","Madhya Pradesh","Haryana"])



#Rajasthan

rj_active = statecases['Active Cases'].iloc[12]

rj_cured = statecases['Cured/Discharged'].iloc[12]

rj_death = statecases['Death'].iloc[12]



labels=['Active Cases','Cured/Discharged','Death']

values_rj=[rj_active,rj_cured,rj_death]

fig.add_trace(go.Pie(labels=labels, values=values_rj), 1, 1)



#Gujarat

guj_active = statecases['Active Cases'].iloc[13]

guj_cured = statecases['Cured/Discharged'].iloc[13]

guj_death = statecases['Death'].iloc[13]



labels=['Active Cases','Cured/Discharged','Death']

values_guj=[guj_active,guj_cured,guj_death]

fig.add_trace(go.Pie(labels=labels, values=values_guj), 1, 2)



#Madhya Pradesh

mp_active = statecases['Active Cases'].iloc[14]

mp_cured = statecases['Cured/Discharged'].iloc[14]

mp_death = statecases['Death'].iloc[14]



labels=['Active Cases','Cured/Discharged','Death']

values_mp=[mp_active,mp_cured,mp_death]

fig.add_trace(go.Pie(labels=labels, values=values_mp), 2, 1)



#Haryana

hy_active = statecases['Active Cases'].iloc[15]

hy_cured = statecases['Cured/Discharged'].iloc[15]

hy_death = statecases['Death'].iloc[15]



labels=['Active Cases','Cured/Discharged','Death']

values_hy=[hy_active,hy_cured,hy_death]

fig.add_trace(go.Pie(labels=labels, values=values_hy), 2, 2)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Chhattisgarh","Punjab","Jammu and Kashmir",

"Uttarakhand"])



#Chhattisgarh

cht_active = statecases['Active Cases'].iloc[16]

cht_cured = statecases['Cured/Discharged'].iloc[16]

cht_death = statecases['Death'].iloc[16]



labels=['Active Cases','Cured/Discharged','Death']

values_cht=[cht_active,cht_cured,cht_death]

fig.add_trace(go.Pie(labels=labels, values=values_cht), 1, 1)



#Punjab

pun_active = statecases['Active Cases'].iloc[17]

pun_cured = statecases['Cured/Discharged'].iloc[17]

pun_death = statecases['Death'].iloc[17]



labels=['Active Cases','Cured/Discharged','Death']

values_pun=[pun_active,pun_cured,pun_death]

fig.add_trace(go.Pie(labels=labels, values=values_pun), 1, 2)



#Jammu and Kashmir

jk_active = statecases['Active Cases'].iloc[18]

jk_cured = statecases['Cured/Discharged'].iloc[18]

jk_death = statecases['Death'].iloc[18]



labels=['Active Cases','Cured/Discharged','Death']

values_jk=[jk_active,jk_cured,jk_death]

fig.add_trace(go.Pie(labels=labels, values=values_jk), 2, 1)



#Uttarakhand

ut_active = statecases['Active Cases'].iloc[19]

ut_cured = statecases['Cured/Discharged'].iloc[19]

ut_death = statecases['Death'].iloc[19]



labels=['Active Cases','Cured/Discharged','Death']

values_ut=[ut_active,ut_cured,ut_death]

fig.add_trace(go.Pie(labels=labels, values=values_ut), 2, 2)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles=["Himachal Pradesh","Goa","Puducherry","Tripura"])



#Goa

goa_active = statecases['Active Cases'].iloc[20]

goa_cured = statecases['Cured/Discharged'].iloc[20]

goa_death = statecases['Death'].iloc[20]



labels=['Active Cases','Cured/Discharged','Death']

values_goa=[goa_active,goa_cured,goa_death]

fig.add_trace(go.Pie(labels=labels, values=values_goa), 1, 1)



#Puducherry

pd_active = statecases['Active Cases'].iloc[21]

pd_cured = statecases['Cured/Discharged'].iloc[21]

pd_death = statecases['Death'].iloc[21]



labels=['Active Cases','Cured/Discharged','Death']

values_pd=[pd_active,pd_cured,pd_death]

fig.add_trace(go.Pie(labels=labels, values=values_pd), 1, 2)



#Tripura

tp_active = statecases['Active Cases'].iloc[22]

tp_cured = statecases['Cured/Discharged'].iloc[22]

tp_death = statecases['Death'].iloc[22]



labels=['Active Cases','Cured/Discharged','Death']

values_tp=[tp_active,tp_cured,tp_death]

fig.add_trace(go.Pie(labels=labels, values=values_tp), 2, 1)



#Himachal Pradesh

hp_active = statecases['Active Cases'].iloc[23]

hp_cured = statecases['Cured/Discharged'].iloc[23]

hp_death = statecases['Death'].iloc[23]



labels=['Active Cases','Cured/Discharged','Death']

values_hp=[hp_active,hp_cured,hp_death]

fig.add_trace(go.Pie(labels=labels, values=values_hp), 2, 2)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}],

         [{'type':'domain'},{'type':'domain'}]]

fig = make_subplots(rows=3, cols=2, specs=specs, subplot_titles=["Chandigarh","Manipur",

                                                                 "Arunachal Pradesh","Meghalaya","Nagaland"])



#Chandigarh

chg_active = statecases['Active Cases'].iloc[24]

chg_cured = statecases['Cured/Discharged'].iloc[24]

chg_death = statecases['Death'].iloc[24]



labels=['Active Cases','Cured/Discharged','Death']

values_chg=[chg_active,chg_cured,chg_death]

fig.add_trace(go.Pie(labels=labels, values=values_chg), 1, 1)



#Manipur

manipur_active = statecases['Active Cases'].iloc[25]

manipur_cured = statecases['Cured/Discharged'].iloc[25]

manipur_death = statecases['Death'].iloc[25]



labels=['Active Cases','Cured/Discharged','Death']

values_manipur=[manipur_active,manipur_cured,manipur_death]

fig.add_trace(go.Pie(labels=labels, values=values_manipur), 1, 2)



#Arunachal Pradesh

ap_active = statecases['Active Cases'].iloc[26]

ap_cured = statecases['Cured/Discharged'].iloc[26]

ap_death = statecases['Death'].iloc[26]



labels=['Active Cases','Cured/Discharged','Death']

values_ap=[ap_active,ap_cured,ap_death]

fig.add_trace(go.Pie(labels=labels, values=values_ap), 2, 1)



#Meghalaya

meg_active = statecases['Active Cases'].iloc[-8]

meg_cured = statecases['Cured/Discharged'].iloc[-8]

meg_death = statecases['Death'].iloc[-8]



labels=['Active Cases','Cured/Discharged','Death']

values_meg=[meg_active,meg_cured,meg_death]

fig.add_trace(go.Pie(labels=labels, values=values_meg), 2, 2)



#Nagaland

nag_active = statecases['Active Cases'].iloc[-7]

nag_cured = statecases['Cured/Discharged'].iloc[-7]

nag_death = statecases['Death'].iloc[-7]



labels=['Active Cases','Cured/Discharged','Death']

values_nag=[nag_active,nag_cured,nag_death]

fig.add_trace(go.Pie(labels=labels, values=values_nag), 3, 1)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
specs = [[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}],

         [{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=3, cols=2, specs=specs, subplot_titles=["Ladakh","Andaman and Nicobar Islands",

                                                        "Sikkim","Dadra and Nagar Haveli and Daman and Diu","Mizoram"])



#Ladakh

lad_active = statecases['Active Cases'].iloc[-6]

lad_cured = statecases['Cured/Discharged'].iloc[-6]

lad_death = statecases['Death'].iloc[-6]



labels=['Active Cases','Cured/Discharged','Death']

values_lad=[lad_active,lad_cured,lad_death]

fig.add_trace(go.Pie(labels=labels, values=values_lad), 1, 1)



#Andaman and Nicobar Islands

ani_active = statecases['Active Cases'].iloc[-5]

ani_cured = statecases['Cured/Discharged'].iloc[-5]

ani_death = statecases['Death'].iloc[-5]



labels=['Active Cases','Cured/Discharged','Death']

values_ani=[ani_active,ani_cured,ani_death]

fig.add_trace(go.Pie(labels=labels, values=values_ani), 1, 2)



#Sikkim

sikkim_active = statecases['Active Cases'].iloc[-4]

sikkim_cured = statecases['Cured/Discharged'].iloc[-4]

sikkim_death = statecases['Death'].iloc[-4]



labels=['Active Cases','Cured/Discharged','Death']

values_sikkim=[sikkim_active,sikkim_cured,sikkim_death]

fig.add_trace(go.Pie(labels=labels, values=values_sikkim), 2, 1)



#Dadra and Nagar Haveli and Daman and Diu

dndu_active = statecases['Active Cases'].iloc[-3]

dndu_cured = statecases['Cured/Discharged'].iloc[-3]

dndu_death = statecases['Death'].iloc[-3]



labels=['Active Cases','Cured/Discharged','Death']

values_dndu=[dndu_active,dndu_cured,dndu_death]

fig.add_trace(go.Pie(labels=labels, values=values_dndu), 2, 2)



#Mizoram

miz_active = statecases['Active Cases'].iloc[-2]

miz_cured = statecases['Cured/Discharged'].iloc[-2]

miz_death = statecases['Death'].iloc[-2]



labels=['Active Cases','Cured/Discharged','Death']

values_miz=[miz_active,miz_cured,miz_death]

fig.add_trace(go.Pie(labels=labels, values=values_miz), 3, 1)



fig.update(layout_title_text='State wise pie chart of different types of cases',layout_showlegend=False)

fig = go.Figure(fig)

fig.show()
fig = px.pie(statecases, values='Cured/Discharged', names='Region',title = "Pie Chart of Cured Cases")

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(statecases, values='Active Cases', names='Region',title = "Pie Chart of Active Cases")

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(statecases, values='Death', names='Region',title = "Pie Chart of Deceased Cases")

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()