# Data Visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.express as px
import plotly.graph_objs as go
import folium

import warnings
warnings.filterwarnings("ignore")
import datetime

# Pipeline Regression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

#ARIMA libraries
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from random import random
# for live kernels to be able to run plotly we use these statements.
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
data=pd.read_csv('/kaggle/input/covid19/train(1).csv')
train=pd.read_csv('/kaggle/input/covid19/train(1).csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
d1=pd.read_csv('/kaggle/input/covid19/countries.csv')
d2=pd.read_csv('/kaggle/input/covid19/population.csv')
d3=pd.read_csv('/kaggle/input/covid19/training_data_with_weather_info_week_2.csv')
data.describe()
d3.describe()
print('Number of Countries  '+str(len(set(data.Country_Region))))
print(set(data.Country_Region))
data_cases = pd.DataFrame(data.groupby(['Date'])['ConfirmedCases'].sum())
data_cases['Deaths'] = pd.DataFrame(data.groupby(['Date'])['Fatalities'].sum())
data_cases['Mortality_Rate']=(data_cases['Deaths']/data_cases['ConfirmedCases'])*100
data_cases.head()
datas1=go.Bar(y=data_cases.ConfirmedCases, x=data_cases.index, name= 'Cases',xaxis='x1',yaxis='y1')
datas2=go.Bar(y=data_cases.Deaths, x=data_cases.index, name= 'Deaths',xaxis='x2',yaxis='y2')

fig = go.Figure(
    data=[datas1,datas2],
    layout=go.Layout(
        xaxis=dict(showgrid=False,domain=[0,0.45]),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False,domain=[0.55,1]),
        yaxis2=dict(showgrid=False,domain=[0, 1],anchor='x2'),
        title_text="Number of Confirmed Cases and Deaths WorldWide.",
        
    )
)
fig.show()
datas1=go.Scatter(y=data_cases.ConfirmedCases, x=data_cases.index, name= 'Cases',xaxis='x1',yaxis='y1')
datas2=go.Scatter(y=data_cases.Deaths, x=data_cases.index, name= 'Deaths',xaxis='x2',yaxis='y2')

fig = go.Figure(
    data=[datas1,datas2],
    layout=go.Layout(
        xaxis=dict(showgrid=False,domain=[0,0.45]),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False,domain=[0.55,1]),
        yaxis2=dict(showgrid=False,domain=[0, 1],anchor='x2'),
        title_text="Number of Confirmed Cases and Deaths Worldwide.",
        
    )
)
fig.show()
case_country = data.groupby(['Country_Region'], as_index=False)['ConfirmedCases'].max()
# Sorting by number of cases
case_country.sort_values('ConfirmedCases', ascending=False, inplace = True)
case_country
death_country = data.groupby(['Country_Region'], as_index=False)['Fatalities'].max()
# Sorting by number of deaths
death_country.sort_values('Fatalities', ascending=False, inplace = True)
death_country
first_date = data[data['ConfirmedCases']>0]
first_date = first_date.groupby('Country_Region')['Date'].agg(['min']).reset_index()
ddd = datetime.datetime(2020, 3, 26)
first_date['last_date']=ddd
first_date['min']=pd.to_datetime(first_date['min'])
first_date=first_date.sort_values('min',ascending=True)
first_date['Days']=first_date['last_date']-first_date['min']
first_date.columns=['Task', 'Start', 'Finish', 'Days']
first_date
# plot showing the arrival of the disease
import random
clr = ["#"+''.join([random.choice('0123456789ABC') for j in range(6)]) for i in range(len(first_date))]
import plotly.figure_factory as ff
fig = ff.create_gantt(first_date, index_col='Task',colors=clr, show_colorbar=False, 
                      bar_width=0.2, showgrid_x=True, showgrid_y=True, height=2500)
fig.show()
# Sum countries with states.
train_agg= train[['Country_Region','Date','ConfirmedCases','Fatalities']].groupby(['Country_Region','Date'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})

# change to datetime format
train_agg['Date'] = pd.to_datetime(train_agg['Date'])
fig = px.line(train_agg, x='Date', y='ConfirmedCases', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Confirmed Cases Over Time for Each Country')
fig.show()
# Interactive time series plot of fatalities
fig = px.line(train_agg, x='Date', y='Fatalities', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Fatalities Over Time for Each Country')
fig.show()
country=case_country.merge(death_country,how='inner',on='Country_Region')
d2.rename(columns={'con ':'Country_Region'},inplace=True)
country=country.merge(d2,how='inner',on='Country_Region')
d1.rename(columns={'name':'Country_Region'},inplace=True)
country=country.merge(d1,how='inner',on='Country_Region')
country=country.drop('S.No.',1)
country=country.drop('Country ',1)
country['mortality_rate']=(country['Fatalities']/country['ConfirmedCases'])*100
country
w=d3.groupby(['Country_Region'])[['temp','stp','prcp']].mean()
country=country.merge(w,how='inner',on='Country_Region')
(country['UrbanPop %'])=(country['UrbanPop %']).astype(str).astype(float)
(country['Med.Age'])=(country['Med.Age']).astype(str).astype(float)

temp_f = country.sort_values(by='ConfirmedCases', ascending=False)
temp_f = temp_f[['Country_Region', 'ConfirmedCases', 'Population(2020)', 'Fatalities','temp', 'UrbanPop %','Med.Age','mortality_rate','Density(P/Km²)','prcp']]
temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap="Blues", subset=['ConfirmedCases', 'prcp'])\
            .background_gradient(cmap="Greens", subset=['Population(2020)','Med.Age'])\
            .background_gradient(cmap="Reds", subset=['Fatalities','mortality_rate'])\
            .background_gradient(cmap="Purples", subset=['Density(P/Km²)','UrbanPop %'])\
            .background_gradient(cmap="Oranges", subset=['temp'])\

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(country)):
    folium.Circle(
        location=[country.iloc[i]['latitude'], country.iloc[i]['longitude']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(country.iloc[i]['Country_Region'])+
                    '<li><bold>Confirmed : '+str(country.iloc[i]['ConfirmedCases'])+
                    '<li><bold>Deaths : '+str(country.iloc[i]['Fatalities']),
        radius=int(country.iloc[i]['ConfirmedCases'])**1.1).add_to(m)
m
fig = px.choropleth(country, locations="Country_Region", 
                    locationmode='country names', color=np.log(country["ConfirmedCases"]), 
                    hover_name="Country_Region", hover_data=['ConfirmedCases','Fatalities'],
                    color_continuous_scale="Sunsetdark", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()
formated_gdf = data.groupby(['Date', 'Country_Region'])['ConfirmedCases', 'Fatalities'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

fig = px.choropleth(formated_gdf, locations="Country_Region", 
                    locationmode='country names', color=formated_gdf['ConfirmedCases'], 
                    hover_name="Country_Region", hover_data=['ConfirmedCases','Fatalities'],
                    color_continuous_scale="Sunsetdark", 
                    animation_frame="Date",  title='Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.scatter(temp_f.sort_values('Fatalities', ascending=False).iloc[:15, :], 
                 x='Med.Age', y='Fatalities', color='Country_Region', size='ConfirmedCases', height=800,
                 text='Country_Region', log_x=True, log_y=True, title='Deaths vs Median Age of Country')
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_rangeslider_visible=True,xaxis=dict(showgrid=False))
fig.show()
fig = px.scatter(temp_f.sort_values('Fatalities', ascending=False).iloc[:15, :], 
                 x='Density(P/Km²)', y='ConfirmedCases', color='Country_Region', size='ConfirmedCases', height=800,
                 text='Country_Region', log_x=True, log_y=True, title='Confirmed Cases vs Density of Country')
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_rangeslider_visible=True,xaxis=dict(showgrid=False))
fig.show()
fig = px.scatter(temp_f.sort_values('Fatalities', ascending=False).iloc[:15, :], 
                 x='UrbanPop %', y='ConfirmedCases', color='Country_Region', size='ConfirmedCases', height=800,
                 text='Country_Region', log_x=True, log_y=True, title='Confirmed Cases vs UrbanPop % of Country')
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_rangeslider_visible=True,xaxis=dict(showgrid=False))
fig.show()
country
top_cases=case_country.iloc[:16,:]
top_deaths=death_country.iloc[:16,:]
datas1=go.Bar(y=top_deaths.Fatalities, x=top_deaths.Country_Region, name= 'Deaths',xaxis='x1',yaxis='y1')
datas2=go.Bar(y=top_cases.ConfirmedCases, x=top_cases.Country_Region, name= 'Cases',xaxis='x2',yaxis='y2')

fig = go.Figure(
    data=[datas1,datas2],
    layout=go.Layout(
        xaxis=dict(showgrid=False,domain=[0,0.45]),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False,domain=[0.55,1]),
        yaxis2=dict(showgrid=False,domain=[0, 1],anchor='x2'),
        title_text="Countrires with maximum number of Confirmed Cases and Deaths.",
        
    )
)
fig.show()
corr=country.corr(method ='pearson')
corr
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns).set_title('Correlation')

train['Date_datetime'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
train.head()
# using pipeline model for prediction of ConfirmedCases
for country in train['Country_Region'].unique():
    print ('training model for country ==>'+str(country))
    country_pd_train = train[train['Country_Region']==country]
    country_pd_test = test[test['Country_Region']==country]
    if country_pd_train['Province_State'].isna().unique().any()==True:
        x = np.array(range(len(country_pd_train))).reshape((-1,1))
        y = country_pd_train['ConfirmedCases']
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                         ('linear', LinearRegression(fit_intercept=False))])
        model = model.fit(x, y)
        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))
        test.loc[test['Country_Region']==country,'ConfirmedCases'] = model.predict(predict_x)
    else:
        for state in country_pd_train['Province_State'].unique():
            state_pd = country_pd_train[country_pd_train['Province_State']==state] 
            state_pd_test = country_pd_test[country_pd_test['Province_State']==state] 
            x = np.array(range(len(state_pd))).reshape((-1,1))
            y = state_pd['ConfirmedCases']
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                         ('linear', LinearRegression(fit_intercept=False))])
            model = model.fit(x, y)
            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))
            test.loc[(test['Country_Region']==country)&(test['Province_State']==state),'ConfirmedCases'] = model.predict(predict_x)
# using pipeline model for prediction of Fatalities
for country in train['Country_Region'].unique():
    print ('training model for country ==>'+str(country))
    country_pd_train = train[train['Country_Region']==country]
    country_pd_test = test[test['Country_Region']==country]
    if country_pd_train.loc[:,'Province_State'].isna().unique().any()==True:
        x = np.array(range(len(country_pd_train))).reshape((-1,1))
        y = country_pd_train['Fatalities']
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                         ('linear', LinearRegression(fit_intercept=False))])
        model = model.fit(x, y)
        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))
        test.loc[test['Country_Region']==country,'Fatalities'] = model.predict(predict_x)
    else:
        for state in country_pd_train['Province_State'].unique():
            state_pd = country_pd_train[country_pd_train['Province_State']==state] 
            state_pd_test = country_pd_test[country_pd_test['Province_State']==state] 
            x = np.array(range(len(state_pd))).reshape((-1,1))
            y = state_pd['Fatalities']
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                         ('linear', LinearRegression(fit_intercept=False))])
            model = model.fit(x, y)
            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))
            test.loc[(test['Country_Region']==country)&(test['Province_State']==state),'Fatalities'] = model.predict(predict_x)
test['Fatalities']=test['Fatalities'].astype('str')
test['ConfirmedCases']=test['ConfirmedCases'].astype('str')
test['Fatalities']=test['Fatalities'].str.replace('-','')
test['ConfirmedCases']=test['ConfirmedCases'].str.replace('-','')
test['Fatalities']=test['Fatalities'].astype('float')
test['ConfirmedCases']=test['ConfirmedCases'].astype('float')
# result of Linear regressive pipeline model
test.head()
submits = pd.DataFrame()
submits['ConfirmedCases'] = test['ConfirmedCases'].astype('int')
submits['Fatalities'] = test['Fatalities'].astype('int')
submits.describe()
# regions=unique(countries+province)
for i in range(len(train)):
    if(pd.isna(train.loc[i,'Province_State'])==True):
        train.loc[i,'Lat']=train.loc[i,'Country_Region']
    else:
        train.loc[i,'Lat']=train.loc[i,'Country_Region']+str(train.loc[i,'Province_State'])
countries_list=train.Lat.unique()
df1=[]
for i in countries_list:
    df1.append(train[train['Lat']==i])
print("we have "+ str(len(df1))+" regions in our dataset")
# ARIMA MODEL PREDICTION.
submit_confirmed=[]
submit_fatal=[]
for i in df1:
    # contrived dataset
    data = i.ConfirmedCases.astype('int32').tolist()
    # fit model
    try:
        model = SARIMAX(data, order=(2,1,0), seasonal_order=(1,1,0,12))#seasonal_order=(1, 1, 1, 1))
        #model = ARIMA(data, order=(3,1,2))
        model_fit = model.fit(disp=False)
        # make prediction
        predicted = model_fit.predict(len(data), len(data)+34)
        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)
        submit_confirmed.extend(list(new[-43:]))
    except:
        submit_confirmed.extend(list(data[-10:-1]))
        for j in range(34):
            submit_confirmed.append(data[-1]*2)
    
    # contrived dataset
    data = i.Fatalities.astype('int32').tolist()
    # fit model
    try:
        model = SARIMAX(data, order=(2,1,0), seasonal_order=(1,1,0,12))#seasonal_order=(1, 1, 1, 1))
        #model = ARIMA(data, order=(3,1,2))
        model_fit = model.fit(disp=False)
        # make prediction
        predicted = model_fit.predict(len(data), len(data)+34)
        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)
        submit_fatal.extend(list(new[-43:]))
    except:
        submit_fatal.extend(list(data[-10:-1]))
        for j in range(34):
            submit_fatal.append(data[-1]*2)

df_submit=pd.concat([pd.Series(np.arange(1,1+len(submit_confirmed))),pd.Series(submit_confirmed),pd.Series(submit_fatal)],axis=1)
df_submit=df_submit.fillna(method='pad').astype(int)
s=df_submit
df_submit.index=test.Date
df_submit.columns=['Id','ConfirmedCases','Fatalities']
data_casess = pd.DataFrame(test.groupby(['Date'])['ConfirmedCases'].sum())
data_casess['Deaths'] = pd.DataFrame(test.groupby(['Date'])['Fatalities'].sum())
data_casess['Mortality_Rate']=(data_cases['Deaths']/data_cases['ConfirmedCases'])*100
data_casess.head()
datas1=go.Bar(y=data_casess.ConfirmedCases, x=data_casess.index, name= 'Cases',xaxis='x1',yaxis='y1')
datas2=go.Bar(y=data_casess.Deaths, x=data_casess.index, name= 'Deaths',xaxis='x2',yaxis='y2')

fig = go.Figure(
    data=[datas1,datas2],
    layout=go.Layout(
        xaxis=dict(showgrid=False,domain=[0,0.45]),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False,domain=[0.55,1]),
        yaxis2=dict(showgrid=False,domain=[0, 1],anchor='x2'),
        title_text="PIPELINE REGREESION Forecast of Number of Confirmed Cases and Deaths WorldWide.",
        
    )
)
fig.show()
datas1=go.Bar(y=df_submit.ConfirmedCases, x=data_casess.index, name= 'Cases',xaxis='x1',yaxis='y1')
datas2=go.Bar(y=df_submit.Fatalities, x=data_casess.index, name= 'Deaths',xaxis='x2',yaxis='y2')

fig = go.Figure(
    data=[datas1,datas2],
    layout=go.Layout(
        xaxis=dict(showgrid=False,domain=[0,0.45]),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False,domain=[0.55,1]),
        yaxis2=dict(showgrid=False,domain=[0, 1],anchor='x2'),
        title_text="ARIMA Forecast of Number of Confirmed Cases and Deaths WorldWide.",
        
    )
)
fig.show()
