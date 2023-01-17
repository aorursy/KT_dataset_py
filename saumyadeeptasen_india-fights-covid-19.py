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
import pandas as pd

import numpy as np

import seaborn as sns

from scipy.integrate import odeint



import math

import bokeh

import plotly.express as px

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from urllib.request import urlopen

import json

from dateutil import parser

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.layouts import row, column

from bokeh.resources import INLINE

from bokeh.io import output_notebook

from bokeh.models import Span

import warnings

from datetime import date

import scipy

warnings.filterwarnings("ignore")

output_notebook(resources=INLINE)



from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import PolynomialFeatures
covid_India_cases = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

covid_India_cases.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered', 'Confirmed': 'Confirmed'}, inplace=True)

statewise_cases = pd.DataFrame(covid_India_cases.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())

statewise_cases["Country"] = "India"

fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',

                  color='Confirmed', hover_data=['State'],

                  color_continuous_scale='Rainbow')



fig.show()
positions = pd.read_csv('../input/utm-of-india/UTM ZONES of INDIA.csv')

ind_grp=statewise_cases.merge(positions , left_on='State', right_on='State / Union Territory')
import folium

map = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='Stamen Toner')

for lat, lon,state,Confirmed,Recovered,Deaths in zip(ind_grp['Latitude'], ind_grp['Longitude'],ind_grp['State'],ind_grp['Confirmed'],ind_grp['Recovered'],ind_grp['Deaths']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =(

                    'State: ' + str(state) + '<br>'

                    'Confirmed: ' + str(Confirmed) + '<br>'

                      'Recovered: ' + str(Recovered) + '<br>'

                      'Deaths: ' + str(Deaths) + '<br>'),



                        fill_color='red',

                        fill_opacity=0.7 ).add_to(map)

map
ind_map = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

pos = pd.read_csv('../input/utm-of-india/UTM ZONES of INDIA.csv')

ind_map = ind_map.merge(pos , left_on='State/UnionTerritory', right_on='State / Union Territory')

ind_map  = ind_map.groupby(['Date', 'State/UnionTerritory','Latitude','Longitude'])['Confirmed'].sum().reset_index()

#ind_map.head()
ind_map['size'] = ind_map['Confirmed']*100000000



fig = px.scatter_mapbox(ind_map, lat="Latitude", lon="Longitude",

                     color="Confirmed", size='size',hover_data=['State/UnionTerritory'],

                     color_continuous_scale='burgyl', animation_frame="Date", 

                     title='Spread total cases over time in India')

fig.update(layout_coloraxis_showscale=True)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, mapbox_center = {"lat":20.5937,"lon":78.9629})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
train1 = pd.read_csv('../input/countrydatafile/coordinates.csv',parse_dates=['Date'])

train1['Province/State'] = train1['Province/State'].fillna('')

temp = train1[[col for col in train1.columns if col != 'Province/State']]



latest = temp[temp['Date'] == max(temp['Date'])].reset_index()

latest_grouped = latest.groupby('Country/Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()
indiansubcon = list(['India','Pakistan','Bangladesh','Nepal','Sri Lanka','Bhutan','Maldives'])

insubc_latest_grouped = latest_grouped[latest_grouped['Country/Region'].isin(indiansubcon)]
fig = px.choropleth(insubc_latest_grouped, locations="Country/Region", 

                    locationmode='country names', color="ConfirmedCases", 

                    hover_name="Country/Region", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='Indian Subcontinent Countries with Confirmed Cases', scope='asia', height=800)

fig.show()
age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

india_covid_19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')
fig = px.pie(age_details, values="TotalCases", names="AgeGroup",title='Confirmed cases of India')

fig.show()
#individual_details.head()

labels = ['Missing', 'Male', 'Female']

sizes = []

sizes.append(individual_details['gender'].isnull().sum())

sizes.append(list(individual_details['gender'].value_counts())[0])

sizes.append(list(individual_details['gender'].value_counts())[1])





explode = (0, 0.1, 0)





plt.figure(figsize= (15,10))

plt.title('Percentage of Gender',fontsize = 20)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'])

state_testing['Date'] = pd.to_datetime(state_testing['Date'])



spread=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

#spread.head()



spread['Date'] = spread['Date'].apply(pd.to_datetime)

spread = spread[spread['Date'] > pd.Timestamp(date(2020,1,20))]

india_spread = spread[spread['Country/Region']=='India'].reset_index(drop=True)

india_spread['Active'] = india_spread['Confirmed'] - (india_spread['Deaths'] + india_spread['Recovered'])





# Daily cases

india_spread['Confirmed_diff'] = india_spread['Confirmed'].diff()

india_spread['Death_diff'] = india_spread['Deaths'].diff()

india_spread['Recovered_diff'] = india_spread['Recovered'].diff()



# growth rate of confirmed cases

india_spread['Confirmed_gr'] = np.round(india_spread['Confirmed_diff'].pct_change(), 2)
date_india_spread = india_spread.groupby('Date')['Confirmed','Deaths','Recovered', 'Active'].sum().reset_index()



trace1 = go.Scatter(

                x=date_india_spread['Date'],

                y=date_india_spread['Confirmed'],

                name="Confirmed",

                mode='lines+markers',

                line_color='orange')

trace2 = go.Scatter(

                x=date_india_spread['Date'],

                y=date_india_spread['Deaths'],

                name="Deaths",

                mode='lines+markers',

                line_color='red')



trace3 = go.Scatter(

                x=date_india_spread['Date'],

                y=date_india_spread['Recovered'],

                name="Recovered",

                mode='lines+markers',

                line_color='green')

trace4 = go.Scatter(

                x=date_india_spread['Date'],

                y=date_india_spread['Active'],

                name="Active",

                mode='lines+markers',

                line_color='blue')



layout = go.Layout(template="ggplot2", width=700, height=500, title_text = '<b>Spread of the Coronavirus In India Over Time </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2,trace3, trace4], layout = layout)

fig.show()



# plot daily cases

colors = ['#FFA500']*85

colors[-5] = 'crimson'

fig = px.bar(india_spread, 

             x="Date", y="Confirmed_diff", 

             title='<b>New Confirm Cases Per Day In India</b>', 

             orientation='v', 

             width=700, height=600)

fig.update_traces(marker_color=colors, opacity=0.8)



fig.add_annotation( # add a text callout with arrow

    text="First Lockdown", x='2020-03-24', y=1400, arrowhead=1, showarrow=True

)



fig.add_annotation( # add a text callout with arrow

    text="Extended Lockdown", x='2020-04-14', y=1400, arrowhead=1, showarrow=True

)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-03-24',

            y0=0,

            x1='2020-03-24',

            y1=1800,

            line=dict(

                color="RoyalBlue",

                width=1,

                dash="dashdot"

            )))



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-04-14',

            y0=0,

            x1='2020-04-14',

            y1=1800,

            line=dict(

                color="RoyalBlue",

                width=1,

                dash="dashdot"

            )))





fig.update_layout(template = 'plotly_white',font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()







# plot of growth rate of confirmed cases

fig1 = px.scatter(india_spread, 

                 x='Date', 

                  y="Confirmed_gr", 

                  text='Confirmed_gr',

                  range_x=['2020-03-05','2020-04-22'])

fig1.update_traces(marker=dict(size=3,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                  marker_color='#4169e1',

                  mode='text+lines+markers',textposition='top center', )



fig1.update_layout(template = 'plotly_white', width=700, height=700, title_text = '<b>Growth percent in number of total<br>COVID-19 cases in India on each day<br>compared to the previous day</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig1.show()
state_details = pd.pivot_table(india_covid_19, values=['Confirmed','Deaths','Cured'], index='State/UnionTerritory', aggfunc='max')

state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],2)

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)

state_details = state_details.sort_values(by='Confirmed', ascending= False)

state_details.style.background_gradient(cmap='Reds')
#state_testing.head()

testing=state_testing.groupby('State').sum().reset_index()

#testing.head()

fig = px.bar(testing, 

             x="TotalSamples",

             y="State", 

             orientation='h',

             height=800,

             title='Testing statewise insight')

fig.show()
fig = px.treemap(ICMR_labs, path=['state','city'],

                  color='city', hover_data=['lab','address'],

                  color_continuous_scale='reds')

fig.show()
values = list(ICMR_labs['state'].value_counts())

names = list(ICMR_labs['state'].value_counts().index)

df = pd.DataFrame(list(zip(values, names)), 

               columns =['values', 'names'])

fig = px.bar(df , x="values",y="names",orientation='h',height=1000, title="ICMR Testing Centers in each State")

fig.show()
bed = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

bed['Total beds'] = bed.NumPublicBeds_HMIS + bed.NumRuralBeds_NHP18 + bed.NumUrbanBeds_NHP18
plt.figure(figsize=(20,100))



plt.subplot(5,1,1)

bed=bed.sort_values('NumUrbanHospitals_NHP18', ascending= False)

sns.barplot(data=bed,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('RdBu')[0])

plt.title('Urban Hospitals per states')

plt.xlabel('Count')

plt.ylabel('States')

for i in range(bed.shape[0]):

    count = bed.iloc[i]['NumUrbanHospitals_NHP18']

    plt.text(count+10,i,count,ha='center',va='center')





plt.subplot(5,1,2)

beds=bed.sort_values('NumRuralHospitals_NHP18', ascending= False)

sns.barplot(data=bed,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('RdBu')[1])

plt.title('Rural Hospitals per states')

plt.xlabel('Count')

plt.ylabel('States')

for i in range(bed.shape[0]):

    count = bed.iloc[i]['NumRuralHospitals_NHP18']

    plt.text(count+100,i,count,ha='center',va='center')

    

    

plt.subplot(5,1,3)

Beds=bed.sort_values('Total beds', ascending= False)

sns.barplot(data=Beds,y='State/UT',x='Total beds',color=sns.color_palette('RdBu')[5])

plt.title('Total Beds per states')

plt.xlabel('Count')

plt.ylabel('States')

for i in range(Beds.shape[0]):

    count = Beds.iloc[i]['Total beds']

    plt.text(count+1500,i,count,ha='center',va='center')

    

plt.subplot(5,1,4)

hospitalBeds=bed.sort_values('NumUrbanBeds_NHP18', ascending= False)

sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('RdBu')[2])

plt.title('Rural Beds per states')

plt.xlabel('Count')

plt.ylabel('States')

for i in range(hospitalBeds.shape[0]):

    count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']

    plt.text(count+2000,i,count,ha='center',va='center')

    

    

plt.subplot(5,1,5)

hospitalBeds=bed.sort_values('NumRuralBeds_NHP18', ascending= False)

sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('RdBu')[3])

plt.title('Rural Beds per states')

plt.xlabel('Count')

plt.ylabel('States')

for i in range(hospitalBeds.shape[0]):

    count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']

    plt.text(count+2500,i,count,ha='center',va='center')



plt.show()

plt.tight_layout()
train=pd.read_csv('../input/covid19all1/covid-19-all.csv')

train.head()
india_df = train[train['Country/Region']=='India'].groupby('Date')['Confirmed','Deaths'].sum().reset_index()

india_df
india_df['day_count'] = list(range(1,len(india_df)+1))





#india_df.Confirmed

#india_df.Confirmed.shift(1)



india_df['rate']=(india_df.Confirmed-india_df.Confirmed.shift(1))/(india_df.Confirmed)

india_df['increase']=(india_df.Confirmed-india_df.Confirmed.shift(1))



ydata = india_df.Confirmed

xdata = india_df.day_count



plt.plot(xdata, ydata, 'o')

plt.title("India")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()
in_df = train[train['Country/Region']=='India'].groupby('Date')['Confirmed','Deaths','Recovered'].sum().reset_index(False)

in_df['Active']=in_df['Confirmed']-in_df['Deaths']-in_df['Recovered']

in_df = in_df[in_df.Active>=100]
from scipy.optimize import curve_fit

import pylab

from datetime import timedelta



in_df['day_count'] = list(range(1,len(in_df)+1))

in_df['increase'] = (in_df.Active-in_df.Active.shift(1))

in_df['rate'] = (in_df.Active-in_df.Active.shift(1))/in_df.Active
def sigmoid(x,c,a,b):

     y = c*1 / (1 + np.exp(-a*(x-b)))

     return y



xdata = np.array(list(in_df.day_count)[::2])

ydata = np.array(list(in_df.Active)[::2])



population=1.332*10**9

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6,100.]))

print(popt)
est_a = 24641

est_b = 0.18

est_c = 32

x = np.linspace(-1, in_df.day_count.max()+50, 50)

y = sigmoid(x,est_a,est_b,est_c)

pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit',alpha = 0.5)

pylab.ylim(-0.05, est_a*1.05)

pylab.xlim(-0.05, est_c*2.05)

pylab.legend(loc='best')

plt.xlabel('days from day 1')

plt.ylabel('confirmed cases')

plt.title('India')

pylab.show()





print('model start date:',in_df[in_df.day_count==1].index[0])

print('model fitted max Active at:',int(est_a))

print('model sigmoidal coefficient is:',round(est_b,3))

print('model curve stop steepening, start flattening by day:',int(est_c))

print('model curve flattens by day:',int(est_c)*2)

display(in_df)
def logistic(x, L, k, x0):

    return L / (1 + np.exp(-k * (x - x0))) + 1

p0 = (0,0,0)



def plot_logistic_fit_data(date_india_spread, title, p0=p0):

    date_india_spread['x'] = np.arange(len(date_india_spread)) + 1

    date_india_spread['y'] = date_india_spread['Confirmed']

    

    

    x = date_india_spread['x']

    y = date_india_spread['y']



    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )

    

    popt, pcov = c2

    

    

    x = range(1,date_india_spread.shape[0] + int(popt[2]))

    y_fit = logistic(x, *popt)

    

    p_df = pd.DataFrame()

    p_df['x'] = x

    p_df['y'] = y_fit.astype(int)

    

    print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))

    print("Predicted k (growth rate): " + str(float(popt[1])))

    print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")

    

    x0 = int(popt[2])

    

    

    traceC = go.Scatter(

        x=date_india_spread['x'], y=date_india_spread['y'],

        name="Confirmed",

        marker=dict(color="#FF4500"),

        mode = "markers+lines",

        text=date_india_spread['Confirmed'],

    )



    traceP = go.Scatter(

        x=p_df['x'], y=p_df['y'],

        name="Predicted",

        marker=dict(color="blue"),

        mode = "lines",

        text=p_df['y'],

    )

    

    trace_x0 = go.Scatter(

        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],

        name = "X0 - Inflexion point",

        marker=dict(color="black"),

        mode = "lines",

        text = "X0 - Inflexion point"

    )



    data = [traceC, traceP, trace_x0]

    

    

    layout = go.Layout(template = 'plotly_white',width=700, height=500, title = 'Cumulative Conformed cases and logistic curve projection',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'),

                  xaxis = dict(title = 'Day since first case', showticklabels=True), 

                  yaxis = dict(title = 'Number of cases'),

                  hovermode = 'closest'

         )

    

    

        

    fig = go.Figure(data = data, layout = layout)

    fig.show()

    

    



L = 3308643

k = 0.25

x0 = 100

p0 = (L, k, x0)

plot_logistic_fit_data(date_india_spread, 'India')
data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')



data["Date"]=pd.to_datetime(data["Date"])

datewise=data.groupby(["Date"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise["Days Since"]=datewise.index-datewise.index.min()

datewise["Days Since"]=datewise.index-datewise.index[0]

datewise["Days Since"]=datewise["Days Since"].dt.days

datewise.head()
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]

valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]

model_scores=[]
lin_reg=LinearRegression(normalize=True)

lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
plt.figure(figsize=(19,8))

prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Confirmed"],label="Actual Confirmed Cases")

plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Linear Regression Prediction")

plt.legend()
poly = PolynomialFeatures(degree = 2)
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))

valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))

y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)

linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)

rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))

model_scores.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))

plt.figure(figsize=(19,8))

predictions_poly=linreg.predict(comp_data)

plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)

plt.plot(datewise.index,predictions_poly, linestyle='--',label="Best Fit for Polynomial Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Polynomial Regression Prediction")

plt.legend()
new_prediction_poly=[]

for i in range(1,18):

    new_date_poly=poly.fit_transform(np.array(datewise["Days Since"].max()+i).reshape(-1,1))

    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))

print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
plt.figure(figsize=(20,10))

prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)

plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Support Vector Machine Regressor Prediction")

plt.legend()
forest_reg = RandomForestRegressor(n_estimators=5, random_state=42)

forest_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_forestreg=forest_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_forestreg)))

print("Root Mean Square Error for Forest Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_forestreg)))
plt.figure(figsize=(20,10))

prediction_forestreg=forest_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))

plt.plot(datewise["Confirmed"],label="Actual Confirmed Cases")

plt.plot(datewise.index,prediction_forestreg, linestyle='--',label="Predicted Confirmed Cases using Forest Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.title("Confirmed Cases Forest Regression Prediction")

plt.legend()
new_date=[]

new_prediction_lr=[]

new_prediction_svm=[]

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('display.float_format', lambda x: '%.6f' % x)

model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_poly,new_prediction_svm),

                               columns=["Dates","Linear Regression Prediction","Polynonmial Regression Prediction","SVM Prediction"])

model_predictions