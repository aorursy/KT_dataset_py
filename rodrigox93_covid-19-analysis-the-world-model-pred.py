import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_error

import plotly.graph_objs as go

import datetime

import plotly.express as px

import folium

import warnings

import folium 

from folium import plugins



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *

from plotly.subplots import make_subplots





import warnings



warnings.filterwarnings('ignore')



%matplotlib inline



#Funciones:



def rmsle_cv(model,x_test,y_test):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_test)

    rmse= np.sqrt(-cross_val_score(model, x_test, y_test, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def grafico_lt(dates,cl_cases,ar_cases,br_cases,pe_cases,co_cases,bo_cases,ec_cases,title):



    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=dates, y=cl_cases, name='Chile'))

    fig1.add_trace(go.Scatter(x=dates, y=ar_cases, name='Argentina'))

    fig1.add_trace(go.Scatter(x=dates, y=br_cases, name='Brazil'))

    fig1.add_trace(go.Scatter(x=dates, y=pe_cases, name='Peru'))

    fig1.add_trace(go.Scatter(x=dates, y=co_cases, name='Colombia'))

    fig1.add_trace(go.Scatter(x=dates, y=bo_cases, name='Bolivia'))

    fig1.add_trace(go.Scatter(x=dates, y=ec_cases, name='Ecuador'))





    fig1.layout.update(title_text=title,xaxis_showgrid=False, yaxis_showgrid=False, width=800,

            height=600,font=dict(

            size=15,

            color="Black"    

        ))

    fig1.layout.plot_bgcolor = 'White'

    fig1.layout.paper_bgcolor = 'White'

    fig1.show()

    

!pip install folium

# installing external lib opencage

!pip install opencage



from opencage.geocoder import OpenCageGeocode



%matplotlib inline
data_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

ultima_fecha = data_confirmed.columns

ultima_fecha = ultima_fecha[-1]
data_confirmed.head()
deaths_data.head()
recoveries_df
import folium

world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)

for i in range(0,len(data_confirmed)):

    folium.Circle(

        location=[data_confirmed.iloc[i]['Lat'], data_confirmed.iloc[i]['Long']],

        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+data_confirmed.iloc[i]['Country/Region']+"</h5>"+

                    "<div style='text-align:center;'>"+"</div>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #555;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed "+str(data_confirmed.iloc[i,-1])+"</li>"+

        "</ul>"

        ,

        radius=(int((np.log(data_confirmed.iloc[i,-1]+1.00001)))+0.2)*50000,

        color='#ff6600',

        fill_color='#ff8533',

        fill=True).add_to(world_map)



world_map
confirmed = data_confirmed.loc[:, '1/22/20': ultima_fecha]

dates = confirmed.keys()

days = np.array([i for i in range(len(dates))]).reshape(-1, 1)

title = 'Number of cases in Latin America'

cl_cases = []

ar_cases = []

br_cases = []

pe_cases = []

co_cases = []

bo_cases = []

ec_cases = []



world_cases = []



for i in dates:

    confirmed_sum = confirmed[i].sum()

    

    world_cases.append(confirmed_sum)

   

    cl_cases.append(data_confirmed[data_confirmed['Country/Region']=='Chile'][i].sum())

    ar_cases.append(data_confirmed[data_confirmed['Country/Region']=='Argentina'][i].sum())

    br_cases.append(data_confirmed[data_confirmed['Country/Region']=='Brazil'][i].sum())

    pe_cases.append(data_confirmed[data_confirmed['Country/Region']=='Peru'][i].sum())

    co_cases.append(data_confirmed[data_confirmed['Country/Region']=='Colombia'][i].sum())

    bo_cases.append(data_confirmed[data_confirmed['Country/Region']=='Bolivia'][i].sum())

    ec_cases.append(data_confirmed[data_confirmed['Country/Region']=='Ecuador'][i].sum())



recuperados_cl = []

recuperados_ec = []

recuperados_br = []



for i in dates:

    recuperados_cl.append(recoveries_df[recoveries_df['Country/Region']=='Chile'][i].sum())

    recuperados_ec.append(recoveries_df[recoveries_df['Country/Region']=='Ecuador'][i].sum())

    recuperados_br.append(recoveries_df[recoveries_df['Country/Region']=='Brazil'][i].sum())

    

    

confirmed_death = deaths_data.loc[:, '1/22/20': ultima_fecha]

dates_d = confirmed_death.keys()

days_d = np.array([i for i in range(len(dates_d))]).reshape(-1, 1)

title2='Number of deaths in Latin America'

cl_cases_d = []

ar_cases_d = []

br_cases_d = []

pe_cases_d = []

co_cases_d = []

bo_cases_d = []

ec_cases_d = []



world_cases_d = []



for i in dates_d:

    confirmed_sum_d = confirmed_death[i].sum()

    

    world_cases_d.append(confirmed_sum_d)

   

    cl_cases_d.append(deaths_data[deaths_data['Country/Region']=='Chile'][i].sum())

    ar_cases_d.append(deaths_data[deaths_data['Country/Region']=='Argentina'][i].sum())

    br_cases_d.append(deaths_data[deaths_data['Country/Region']=='Brazil'][i].sum())

    pe_cases_d.append(deaths_data[deaths_data['Country/Region']=='Peru'][i].sum())

    co_cases_d.append(deaths_data[deaths_data['Country/Region']=='Colombia'][i].sum())

    bo_cases_d.append(deaths_data[deaths_data['Country/Region']=='Bolivia'][i].sum())

    ec_cases_d.append(deaths_data[deaths_data['Country/Region']=='Ecuador'][i].sum())

    

    

confirmed_rec= recoveries_df.loc[:, '1/22/20': ultima_fecha]

dates_r = confirmed_death.keys()

days_r = np.array([i for i in range(len(dates_d))]).reshape(-1, 1)

title3='Number of Recovered in Latin America'



world_cases_r = []



for i in dates_r:

    confirmed_sum_r = confirmed_rec[i].sum()

    

    world_cases_r.append(confirmed_sum_r)



    

datos_world_rdca = pd.DataFrame({'Date':ultima_fecha,'Deaths':[world_cases_d[-1]],'Confirmed':[world_cases[-1]],'Recovered':[world_cases_r[-1]]})

# Active Case = confirmed - deaths - recovered

datos_world_rdca['Active'] = datos_world_rdca['Confirmed'] - datos_world_rdca['Deaths'] - datos_world_rdca['Recovered']

temp = datos_world_rdca

temp.style.background_gradient(cmap='Pastel1')
confirmed = '#393e46' 

death = '#ff2e63' 

recovered = '#21bf73' 

active = '#fe9801' 



tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,

                 color_discrete_sequence=[recovered, active, death])

fig.show()
datos_world_rdca_fecha = pd.DataFrame({'Date':dates,'Deaths':world_cases_d,'Confirmed':world_cases,'Recovered':world_cases_r})



#https://www.kaggle.com/gatunnopvp/covid-19-in-brazil-prediction-updated-04-20-20

by_date = datos_world_rdca_fecha[['Date','Confirmed','Deaths']]



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Cases and Deaths by Day"

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=by_date['Date']

           , y=by_date['Confirmed']),

    

    go.Bar(name='Death'

           , x=by_date['Date']

           , y=by_date['Deaths']

           , text=by_date['Deaths']

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
fig1 = go.Figure()

datos_world_rdca_edit2  = datos_world_rdca_fecha.drop(['Date'],axis=1)



grupos = datos_world_rdca_edit2.columns



colores = ['red','blue','green']

for i in range(0,len(grupos)):

    fig1.add_trace(go.Scatter(x=datos_world_rdca_fecha['Date'], y=datos_world_rdca_fecha[grupos[i]],line_color=colores[i], name=grupos[i]))





fig1.layout.update(title_text='Total number of deaths by age group',xaxis_showgrid=False, yaxis_showgrid=False, width=800,

            height=600,font=dict(

            size=15,

            color="Black"    

        ))

fig1.layout.plot_bgcolor = 'White'

fig1.layout.paper_bgcolor = 'White'

fig1.show()
grafico_lt(dates,cl_cases,ar_cases,br_cases,pe_cases,co_cases,bo_cases,ec_cases,title)
confirmed_death = deaths_data.loc[:, '1/22/20': ultima_fecha]

dates_d = confirmed_death.keys()

days_d = np.array([i for i in range(len(dates_d))]).reshape(-1, 1)

title2='Number of deaths in Latin America'

cl_cases_d = []

ar_cases_d = []

br_cases_d = []

pe_cases_d = []

co_cases_d = []

bo_cases_d = []

ec_cases_d = []



world_cases_d = []



for i in dates_d:

    confirmed_sum_d = confirmed_death[i].sum()

    

    world_cases_d.append(confirmed_sum)

   

    cl_cases_d.append(deaths_data[deaths_data['Country/Region']=='Chile'][i].sum())

    ar_cases_d.append(deaths_data[deaths_data['Country/Region']=='Argentina'][i].sum())

    br_cases_d.append(deaths_data[deaths_data['Country/Region']=='Brazil'][i].sum())

    pe_cases_d.append(deaths_data[deaths_data['Country/Region']=='Peru'][i].sum())

    co_cases_d.append(deaths_data[deaths_data['Country/Region']=='Colombia'][i].sum())

    bo_cases_d.append(deaths_data[deaths_data['Country/Region']=='Bolivia'][i].sum())

    ec_cases_d.append(deaths_data[deaths_data['Country/Region']=='Ecuador'][i].sum())

grafico_lt(dates,cl_cases_d,ar_cases_d,br_cases_d,pe_cases_d,co_cases_d,bo_cases_d,ec_cases_d,title2)
confirmed_death = deaths_data.loc[:, '1/22/20': ultima_fecha]

dates_d = confirmed_death.keys()

days_d = np.array([i for i in range(len(dates_d))]).reshape(-1, 1)



us_cases_d = []

china_cases_d = []



us_cases = []

china_cases = []

it_cases = []



sp_cases_d = []

fr_cases_d = []

it_cases_d = []



sp_cases = []

fr_cases = []





recuperados_china = []

recuperados_us = []

recuperados_fr = []

recuperados_sp = []

recuperados_it = []



for i in dates_d:



   

    china_cases_d.append(deaths_data[deaths_data['Country/Region']=='China'][i].sum())

    us_cases_d.append(deaths_data[deaths_data['Country/Region']=='US'][i].sum())

    fr_cases_d.append(deaths_data[deaths_data['Country/Region']=='France'][i].sum())

    sp_cases_d.append(deaths_data[deaths_data['Country/Region']=='Spain'][i].sum())

    it_cases_d.append(deaths_data[deaths_data['Country/Region']=='Italy'][i].sum())





    



for i in dates:



   

    china_cases.append(data_confirmed[data_confirmed['Country/Region']=='China'][i].sum())

    us_cases.append(data_confirmed[data_confirmed['Country/Region']=='US'][i].sum())

    fr_cases.append(data_confirmed[data_confirmed['Country/Region']=='France'][i].sum())

    sp_cases.append(data_confirmed[data_confirmed['Country/Region']=='Spain'][i].sum())

    it_cases.append(data_confirmed[data_confirmed['Country/Region']=='Italy'][i].sum())











for i in dates:

    recuperados_china.append(recoveries_df[recoveries_df['Country/Region']=='China'][i].sum())

    recuperados_us.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())

    recuperados_fr.append(recoveries_df[recoveries_df['Country/Region']=='France'][i].sum())

    recuperados_sp.append(recoveries_df[recoveries_df['Country/Region']=='Spain'][i].sum())

    recuperados_it.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())





gris = '#393e46' 

rojo = '#ff2e63' 

verde = '#21bf73' 

amarrillo = '#fe9801' 



data_total_us = pd.DataFrame({'Country': ('China'),'Date': pd.to_datetime(dates),'Active': china_cases,'Deaths': china_cases_d,'Recovered':recuperados_china})

data_total_china = pd.DataFrame({'Country': ('United States'),'Date': pd.to_datetime(dates),'Active': us_cases,'Deaths': us_cases_d,'Recovered':recuperados_us})

data_total_fr = pd.DataFrame({'Country': ('France'),'Date': pd.to_datetime(dates),'Active': fr_cases,'Deaths': fr_cases_d,'Recovered':recuperados_fr})

data_total_sp = pd.DataFrame({'Country': ('Spain'),'Date': pd.to_datetime(dates),'Active': sp_cases,'Deaths': sp_cases_d,'Recovered':recuperados_sp})

data_total_it = pd.DataFrame({'Country': ('Italy'),'Date': pd.to_datetime(dates),'Active': it_cases,'Deaths': it_cases_d,'Recovered':recuperados_it})





# Apilar los __DataFrames__ uno encima del otro

china_us= pd.concat([data_total_us, data_total_china], axis=0)

china_us_fr= pd.concat([china_us, data_total_fr], axis=0)

china_us_fr_sp= pd.concat([china_us_fr, data_total_sp], axis=0)



china_us_fr_sp_it= pd.concat([china_us_fr_sp, data_total_it], axis=0)





def location(row):

    if row['Country']=='China':

            return 'China'

    elif row['Country']=='United States':

            return 'United States'

        

    elif row['Country']=='France':

            return 'France'

        

    elif row['Country']=='Italy':

            return 'Italy'

    else:

        return 'Spain'

        



temp = china_us_fr_sp_it.copy()

temp['Region'] = temp.apply(location, axis=1)

temp['Date'] = temp['Date'].dt.strftime('%Y-%m-%d')

temp = temp.groupby(['Country', 'Date'])['Active', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp.melt(id_vars=['Country', 'Date'], value_vars=['Active', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count')



fig = px.bar(temp, y='Country', x='Count', color='Case', barmode='group', orientation='h',

             text='Count', title='Casos COVID-19 China-US-France-Spain', animation_frame='Date',

             color_discrete_sequence= [gris,rojo, verde], range_x=[0, 900000])

fig.update_traces(textposition='outside')



fig.show()
print("total confirmed cases:",confirmed_sum)
X = days

y= np.array(world_cases).reshape(-1, 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=42) 





parameters = {

        'alpha':[0.000001,0.0001,0.001,0.1,0.0002,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.4,1.5,1.6,2,2.1,2.2,2.3,2.4,2.5,3,4,5,6,7,8,9,10],

        'kernel': ['polynomial'],

        'degree': [0.0001,0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,1.6,2,2.1,2.2,2.3,2.4,3,4,4.2,4.3,4.5,5,6,7,7.5,8,9],

        'coef0': [0.0001,0.001, 0.1,0.0002,0.2,0.25,1,1.2,1.5,2,2.1,2.2,2.5,3,3.2,3.5,4,4.1,4.2,4.3,4.5,5,6,7,8,9]

    }

clf =KernelRidge()

clf1 = GridSearchCV(clf, parameters,scoring='neg_mean_squared_error', n_jobs=-1, cv=5)

clf1.fit(X_train, y_train)



best_params = clf1.best_params_

beast_score =clf1.best_score_



print("Mejor puntuacion:",beast_score)

print("Mejores Parametros;",best_params)
model_kr = KernelRidge(**best_params)

model_kr.fit(X, y)



score = rmsle_cv(model_kr,X,y)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
dataframe=pd.DataFrame(X_test, columns=['Days'])



pred_rg=model_kr.predict(np.array(X_test).reshape(-1,1))

#xgb_pred = np.expm1(model_xgb.predict(test))

#pred_rg = np.expm1(model_kr.predict(np.array(X_test).reshape(-1,1)))



print("Root Mean Square Value:",np.sqrt(mean_squared_error( y_test,pred_rg)))

print('MAE:', mean_absolute_error(pred_rg,  y_test))

print('MSE:',mean_squared_error(pred_rg,  y_test))



plt.figure(figsize=(11,6))

plt.plot( y_test,label="Actual Confirmed Cases")

plt.plot(dataframe.index,pred_rg, linestyle='--',label="Predicted Confirmed Cases using Kernel Ridge",color='black')

plt.xlabel('Days')

plt.ylabel('Confirmed Cases')

plt.xticks(rotation=90)

plt.legend()
dataframe=pd.DataFrame(X, columns=['Days'])



pred_rg=model_kr.predict(np.array(X).reshape(-1,1))

#xgb_pred = np.expm1(model_xgb.predict(test))

#pred_rg = np.expm1(model_kr.predict(np.array(X).reshape(-1,1)))



print("Root Mean Square Value:",np.sqrt(mean_squared_error( y,pred_rg)))

print('MAE:', mean_absolute_error(pred_rg, y))

print('MSE:',mean_squared_error(pred_rg,  y))



plt.figure(figsize=(11,6))

plt.plot( y,label="Actual Confirmed Cases")

plt.plot(dataframe.index,pred_rg, linestyle='--',label="Predicted Confirmed Cases using Kernel Ridge",color='black')

plt.xlabel('Days')

plt.ylabel('Confirmed Cases')

plt.xticks(rotation=90)

plt.legend()
days_in_future = 20

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-days_in_future]



start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


kr_pred = model_kr.predict(future_forcast)



Predict_df= pd.DataFrame()

Predict_df["Date"] = list(future_forcast_dates[-days_in_future:])

Predict_df["N° Cases"] =np.round(kr_pred[-days_in_future:])

Predict_df.head()
trace1 = go.Scatter(

                x=future_forcast_dates,

                y=world_cases,

                name="Confirmed cases",

                mode='lines+markers',

                line_color='green')





trace2 = go.Scatter(

                x=Predict_df["Date"],

                y=Predict_df["N° Cases"],

                name="Predictions",

                mode='lines+markers',

                line_color='blue')





layout = go.Layout(template="ggplot2", width=850, height=600, title_text = '<b>Prediction of the next '+str(days_in_future)+' days in the World</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=Predict_df["Date"], y=Predict_df["N° Cases"], name='world'))

 

fig1.layout.update(title_text='Prediction number of cases in World',xaxis_showgrid=False, yaxis_showgrid=False, width=800,

            height=600,font=dict(

            size=10,

            color="Black"    

        ))

fig1.layout.plot_bgcolor = 'White'

fig1.layout.paper_bgcolor = 'White'

fig1.show()
gris = '#393e46' 

rojo = '#ff2e63' 

verde = '#21bf73' 

amarrillo = '#fe9801' 



data_total_cl_3 = pd.DataFrame({'Country': ('Chile'),'Date': pd.to_datetime(dates),'Cases': cl_cases,'Deaths': cl_cases_d,'Recovered':recuperados_cl})

data_total_ec_2 = pd.DataFrame({'Country': ('Ecuador'),'Date': pd.to_datetime(dates),'Cases': ec_cases,'Deaths': ec_cases_d,'Recovered':recuperados_ec})

data_total_br_2 = pd.DataFrame({'Country': ('Brazil'),'Date': pd.to_datetime(dates),'Cases': br_cases,'Deaths': br_cases_d,'Recovered':recuperados_br})





# Apilar los __DataFrames__ uno encima del otro

chile_ecuador= pd.concat([data_total_cl_3, data_total_ec_2], axis=0)

chile_ecuador_brazil= pd.concat([chile_ecuador, data_total_br_2], axis=0)



chile_ecuador_brazil = chile_ecuador_brazil.drop(chile_ecuador_brazil[chile_ecuador_brazil['Cases']==0].index).reset_index()

chile_ecuador_brazil = chile_ecuador_brazil.drop(['index'],axis=1)





def location(row):

    if row['Country']=='Chile':

            return 'Chile'

        

    elif row['Country']=='Ecuador':

            return 'Ecuador'

    else:

        return 'Brazil'

        



temp = chile_ecuador_brazil.copy()

temp['Region'] = temp.apply(location, axis=1)

temp['Date'] = temp['Date'].dt.strftime('%Y-%m-%d')

temp = temp.groupby(['Country', 'Date'])['Cases', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp.melt(id_vars=['Country', 'Date'], value_vars=['Cases', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count')



temp.head()



fig = px.bar(temp, y='Country', x='Count', color='Case', barmode='group', orientation='h',

             text='Count', title='Chile - Ecuador - Brazil', animation_frame='Date',

             color_discrete_sequence= [gris,rojo, verde], range_x=[0, 400000])

fig.update_traces(textposition='outside')



fig.show()