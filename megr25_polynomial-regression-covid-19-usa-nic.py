#Basic Libraries

import pandas as pd

import numpy as np 

import math 

import re 



#Visualization 

import seaborn as sns

import matplotlib.pyplot as plt 

import cufflinks as cf

%matplotlib inline 

sns.set_style("whitegrid")

import folium

cf.go_offline()



from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots
#Reading Data

df = pd.read_csv('../input/covid_19_clean_complete.csv')

df1 = pd.read_csv('../input/tests.csv')

df2 =  pd.read_csv('../input/usa_county_wise.csv')
df.tail(5) 
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (20,5))

sns.heatmap(data = df.isna() ,yticklabels=False , cmap ="plasma", ax= ax1)

sns.heatmap(data = df1.isna() ,yticklabels=False ,cmap ="plasma", ax= ax2)

sns.heatmap(data = df2.isna() ,yticklabels=False ,cmap ="plasma", ax= ax3)

print("Missing Data")
wd = df   # wd = World

wd.info()
wd['Date_c'] = pd.to_datetime(wd['Date'])  #wd World 

UptoDate =wd[wd['Date_c'] == max(wd['Date_c'])] # max because it will filter and show the latest day so It would be updated



#Create canvas map 

World = folium.Map(location = [0,0], tiles='OpenStreetMap', #tite = typed of map https://python-visualization.github.io/folium/quickstart.html

               min_zoom=2, max_zoom=5, zoom_start=2)



#Adding points and Circle 

for date in range (0 , len(UptoDate)):# the len od UptoDate is 265

    folium.Circle(

        radius=int(UptoDate.iloc[date]['Confirmed'])*0.5, # Go to the most updated case and them bring the Confirmed case 

        location=[UptoDate.iloc[date]['Lat'],UptoDate.iloc[date]['Long']], # provide the lat and long of the most updated value

        popup='The Waterfront',color='crimson',

        tooltip=    # her Will show the Legend of all Data 

        '<li><bold>Country : ' + str(UptoDate.iloc[date]['Country/Region'])+  # Podes ponerlos en etiqueta HTML 

        '<li><bold>Confirmed : ' + str(UptoDate.iloc[date]['Confirmed'])+

        '<li><bold>Deaths : ' + str(UptoDate.iloc[date]['Deaths'])+

        '<li><bold>Recovered : ' + str(UptoDate.iloc[date]['Recovered']) + 

         str(" Country :" + str(UptoDate.iloc[date]['Country/Region'])),  # str in python 

        fill=True,fill_color='#3186cc').add_to(World)
World
# Create a New Columns to Show Active cases arround the world 

wd['Active Cases'] = wd['Confirmed'] - wd['Recovered'] - wd['Deaths']



#Analazing Total Confirmed , Recovered, Active cases and Deaths (CRAD)

CRAD = wd[wd['Date_c'] == max(wd['Date_c'])][['Confirmed','Deaths','Recovered','Active Cases']].sum()



#plotting the CRAD

CRAD.iplot(kind= 'barh', color= "turquoise", title = 'COVID-19 Arround the World by Million Cases')
#Tracking the Spread Per day (SPD)



SPD = wd[['Date_c','Confirmed','Deaths','Recovered','Active Cases']].groupby('Date_c').sum()

SPD.iplot(kind= 'scatter', colors=['blue','Darkred','green','Purple'], title = 'COVID-19 Arround the World by Million Cases', xTitle = 'Date' , yTitle = 'People per Million')
import plotly.express as px

fig = px.choropleth(df, locations="Country/Region", locationmode='country names', color=np.log(df["Confirmed"]), 

                    hover_name="Country/Region", animation_frame=df["Date_c"].dt.strftime('%Y-%m-%d'),

                    title='Cases over time', color_continuous_scale=px.colors.sequential.Viridis)

fig.update(layout_coloraxis_showscale=False)

fig.show()
#Recovery Rate UpToDate = (Confirmed - (Infected + Deaths) / Confimed ) * 100

wd['RecRate']= round((((wd['Confirmed'] - (wd['Deaths']+wd['Active Cases'])) / (wd['Confirmed']))*100).fillna(0),2)

wd.tail(5)
NRR = wd[(wd['Date_c'] == max(wd['Date_c'])) & (wd['RecRate'] >= 0 )][['Country/Region','RecRate']].groupby('Country/Region').max().sort_values('RecRate')

NRR.iplot(kind= 'barh', xTitle = ' Recovery Rate 0- 100 %' , yTitle = 'Countries')
#Top Countries Confirmed cases

Cases_confirmed = wd.groupby('Country/Region',as_index= False)[['Confirmed']].max().sort_values('Confirmed')

CC = px.bar(Cases_confirmed.tail(7), x="Confirmed", y="Country/Region",  text='Confirmed', orientation='h', color_discrete_sequence = ['Green'])

#Top Countries Deaths reported

Deaths = wd.groupby('Country/Region',as_index= False)[['Deaths']].max().sort_values('Deaths')

D = px.bar(Deaths.tail(7), x="Deaths", y="Country/Region",  text='Deaths', orientation='h', color_discrete_sequence = ['Darkred'])

#Top Countries Recovered

Recovered = wd.groupby('Country/Region',as_index= False)[['Recovered']].max().sort_values('Recovered')

R = px.bar(Recovered.tail(7), x="Recovered", y="Country/Region",  text='Recovered', orientation='h', color_discrete_sequence = ['Orange'])

#Top Countries Active Cases

Active_cases = wd.groupby('Country/Region',as_index= False)[['Active Cases']].max().sort_values('Active Cases')

AC = px.bar(Active_cases.tail(7), x="Active Cases", y="Country/Region",  text='Active Cases', orientation='h', color_discrete_sequence = ['Skyblue'])





fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Top Countries Confirmed cases', 'Top Countries Deaths reported' , 'Top Countries Recovered','Top Countries Active Cases'  ))



fig.add_trace(CC['data'][0], row=1, col=1) #Remember hacerlo a un lista para que lo pueda leer



fig.add_trace(D['data'][0], row=1, col=2)

fig.add_trace(R['data'][0], row=2, col=1)

fig.add_trace(AC['data'][0], row=2, col=2)





fig.update_layout(height=700)
#Extracting Nic Information

nic = df[df["Country/Region"] == "Nicaragua"].reset_index()

nic.head(2)
#Cleaning Data

df_nic= nic.drop(['index','Province/State'], axis = 1) 

df_nic.head()
df_nic.tail(5)
# Spread COVD- 19 per Day 

NICSPREAD = df_nic[['Date_c','Confirmed','Deaths','Recovered','Active Cases']].groupby('Date_c').sum()

NICSPREAD.iplot(kind='scatter', title = 'Cases in Nicaragua', xTitle = 'Date' , yTitle = 'Verified Cases')
#CRAD_NIC (Confirmed, Recovered, Actived, Death)

CRAD_NIC = df_nic[df_nic['Date_c'] == max(df_nic['Date_c'])][['Confirmed','Deaths','Recovered','Active Cases']].sum() # max or Sum will give the same value

CRAD_NIC.iplot(kind= 'barh', color= "Green", title = 'COVID-19 in Nicaragua')
#COvid 19 - Spread Distribution 

fig,(ax1) = plt.subplots(1,1 , figsize = (35,15))

n = sns.scatterplot(x= 'Date' , y = 'Confirmed', data = df_nic, ax=ax1)

n.set_xticklabels(n.get_xticklabels(), rotation=45)

n.set_title('CoronaVirus in Nic')
# Adding Long and long

nic_dep = pd.read_excel('../input/lat_long_nic.xlsx')

#nic_dep = Nicaragua + Departments

nic_dep['Lat']= nic_dep['Lat'].apply(lambda x : round(x,4))

nic_dep['Long']= nic_dep['Long'].apply(lambda x : round(x,4))

nic_dep['Date'] = df_nic ['Date_c']  # Date Extracted From the Orginal Data

nic_dep.tail(2)
# Nicaragua map

NIC = folium.Map(location = [12.8654, -85.2072], tiles='OpenStreetMap', 

               min_zoom=7, max_zoom=8, zoom_start=7)

  

for (index,row) in nic_dep.iterrows():

    folium.Circle(

        radius=int(row.loc['Cases'])**7, # Go to the Row case and them bring the Confirmed case 

        location=[row.loc['Lat'], row.loc['Long']], # provide the lat and long according to index

        popup='The Waterfront',color='crimson',

        tooltip=    # this Will show the Legend of the markets 

        '<li><bold>Deparment : ' + str(row.loc['Deparments'])+  # HTML  <li><> is a bullet point

        '<li><bold>Confirmed : ' + str(row.loc['Cases']),   

        fill=True,fill_color='#ccfa00').add_to(NIC)

NIC
usa = df2.drop(['UID','iso2','iso3','Admin2','code3','FIPS','Combined_Key'], axis = 1)

# converting date into time format 

usa['Date'] =pd.to_datetime(usa['Date'])



# Grouping by State #The United States of America has 50 states, 1 Federal District, 5 Territories.

#District of Columbia(D.C) is a Federal District, not a state

SBD = usa[['Date','Confirmed','Deaths']].groupby('Date').sum()  #SD = Spread by Day



#Spread by day 

SBD.iplot(kind = 'spread', title = 'COVID-19 in USA')
CD = usa.groupby('Province_State')[['Confirmed','Deaths']].max()

CD.iplot(kind = 'barh', title = 'Confirmed / Death by State')
#Top 10 State Confirmed cases

USA_CONFIRMED = usa.groupby('Province_State',as_index= False)[['Confirmed']].max().sort_values('Confirmed') # USACC = USA CASE CONFIRMED

USA_CC = px.bar(USA_CONFIRMED.tail(15), x="Confirmed", y="Province_State",  text='Confirmed', orientation='h', color_discrete_sequence = ['Green'])



#Top 10 State Death 

USA_DEATH = usa.groupby('Province_State',as_index= False)[['Deaths']].max().sort_values('Deaths') # USACC = USA CASE CONFIRMED

USA_D = px.bar(USA_DEATH.tail(15), x="Deaths", y="Province_State",  text='Deaths', orientation='h', color_discrete_sequence = ['Blue'])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('USA COVID - 19 Confirmed Cases ', 'USA COVID - 19 Deaths'))



fig.add_trace(USA_CC['data'][0], row=1, col=1) #Remember hacerlo a un lista para que lo pueda leer

fig.add_trace(USA_D['data'][0], row=1, col=2)

fig.update_layout(height=700)
import datetime as dt

usa['Date_ordinal'] = usa['Date'].map(dt.datetime.toordinal) #changing Datetime to ordinal (I am not sure if it is the best way, but i couldnt find any other solution)

data = usa.drop(['Province_State','Lat','Long_','Date','Deaths','Country_Region'],axis = 1)

polydata = data.groupby('Date_ordinal',as_index = False).sum()

polydata.tail(5)
# importing Data Set  

y = polydata.iloc[:,1].values

X = polydata.iloc[:,0:1].values

#plotting for better understanding 

plt.scatter(X, y)

print('Cases per day')
#Splitting Data   

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



#Transforming to degree 2 

poly_reg = PolynomialFeatures(degree = 2)   #because polynomial 2nd degree

X_train_poly = poly_reg.fit_transform(X_train)  # transform the data according to the degree selected

X_test_poly = poly_reg.fit_transform(X_test)



#Predicting 

PL =LinearRegression()

PL.fit(X_train_poly,y_train)

y_pred = PL.predict(X_test_poly)

y_pred
print("r2=" ,r2_score(y_test, y_pred))

print("Coef=", PL.coef_)
X1 = np.arange (737570 , 737587) # arrange of numbers to pass to datetime

X1_pol = poly_reg.fit_transform(X1.reshape(-1, 1))

Predictions =PL.predict(X1_pol)





Till_Jun = pd.DataFrame(data = X1 , columns = ['Day'])

Prediction_jun = pd.DataFrame(Predictions , columns = ['Deaths'])

Jun = pd.concat([Till_Jun,Prediction_jun], axis = 1)

Jun ['Date'] = Jun['Day'].map(dt.datetime.fromordinal)

Jun.tail(3)
Jun.iplot(kind = 'bar' , x = 'Date' , y ='Deaths', title = 'Cases till Jun 10th')