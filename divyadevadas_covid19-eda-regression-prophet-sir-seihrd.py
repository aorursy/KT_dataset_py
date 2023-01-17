!pip install lmfit 

# Required Libraries

import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import time
import datetime
from datetime import datetime, date,timedelta
from scipy import integrate, optimize
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

import os

# plotly library
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

#matplot lib
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import ticker

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing, svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,explained_variance_score
import sklearn 
import matplotlib.dates as dates

import mpld3
mpld3.enable_notebook()
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
matplotlib.style.use('ggplot')
import lmfit
from lmfit.lineshapes import gaussian, lorentzian


# Data

TodaysData_Country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
cleaned_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv')
owid_covid_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')


TodaysData_Country.head()
cleaned_data.head()
owid_covid_data.head()
# Data Cleaning
ColumnToClean = ['Confirmed', 'Deaths', 'Recovered', 'Active']
# filling missing values 
TodaysData_Country[['Country_Region']] = TodaysData_Country[['Country_Region']].fillna('')
TodaysData_Country[ColumnToClean] = TodaysData_Country[ColumnToClean].fillna(0)
TodaysData_Country.loc[TodaysData_Country['Country_Region'] == "United Kingdom", "Country_Region"] = "UK"
TodaysData_Country.head()
# Top 10 Countries with highest Death cases
Top10_Countries_death = TodaysData_Country.drop(['Last_Update', 'Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','Mortality_Rate','UID','ISO3'], axis=1) 
Top10_Countries_death = Top10_Countries_death.nlargest(10, 'Deaths')
Top10_Countries_death.head(10)
# Top 10 Countries with highest number of Confirmed cases

Top10_Countries_Confirmed = TodaysData_Country.drop(['Last_Update', 'Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','Mortality_Rate','UID','ISO3'], axis=1) 
Top10_Countries_Confirmed = Top10_Countries_Confirmed.nlargest(10, 'Confirmed')
Top10_Countries_Confirmed.head(10)
# Top 10 Countries with highest number of Recoverd cases

Top10_Countries_Recovered = TodaysData_Country.drop(['Last_Update', 'Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','Mortality_Rate','UID','ISO3'], axis=1) 
Top10_Countries_Recovered = Top10_Countries_Recovered.nlargest(10, 'Recovered')
Top10_Countries_Recovered.head(10)
# Top 10 Countries with highest number of Recoverd cases
Top10_Countries_Active = TodaysData_Country.drop(['Last_Update', 'Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','Mortality_Rate','UID','ISO3'], axis=1) 
Top10_Countries_Active = Top10_Countries_Active.nlargest(10, 'Active')
Top10_Countries_Active.head(10)
# Top 10 countries with Highest Number of Death Rates
fig = px.bar(Top10_Countries_death.sort_values('Deaths',ascending=False)[:20][::-1],x='Deaths',y='Country_Region',title='Top 10 Countries with highest number of Death Cases',text='Deaths', height=900, orientation='h')

#image_bytes = fig.to_image(format='png', , width=1200, height=700, scale=1) # you can use other formats as well (like 'svg','jpeg','pdf')
#img_bytes = fig.to_image(format="png")
#instead of using fig.show()
##from IPython.display import Image
#Image(img_bytes)
#fig.show()
plotly.offline.iplot(fig)

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("Top 10 Countries with Confirmed Cases","Top 10 Countries with Death Cases", "Top 10 Countries with Recovered Cases", "Top 10 Countries with Active Cases")
)

fig.add_trace(go.Bar(name='Confirmed',text='Confirmed', x=Top10_Countries_Confirmed['Country_Region'], y=Top10_Countries_Confirmed['Confirmed']),
              row=1, col=1)


fig.add_trace(go.Bar(name='Deaths',text='Deaths', x=Top10_Countries_death['Country_Region'], y=Top10_Countries_death['Deaths']),
              row=1, col=2)

fig.add_trace(go.Bar(name='Recovered', text='Recovered',x=Top10_Countries_Active['Country_Region'], y=Top10_Countries_Active['Recovered']),
              row=2, col=1)

fig.add_trace(go.Bar(name='Active',text='Active', x=Top10_Countries_Recovered['Country_Region'], y=Top10_Countries_Recovered['Active']),
              row=2, col=2)

fig.update_layout(height=700,title_text="World top 10 countries with Covid-19 Cases", showlegend=False)

#fig.show()
plotly.offline.iplot(fig)

fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=Top10_Countries_death['Country_Region'], y=Top10_Countries_death['Confirmed']),
     go.Bar(name='Deaths', x=Top10_Countries_death['Country_Region'], y=Top10_Countries_death['Deaths']),
     go.Bar(name='Recovered', x=Top10_Countries_death['Country_Region'], y=Top10_Countries_death['Recovered']),
])
# Change the bar mode
fig.update_layout(barmode='group')
#fig.show()
plotly.offline.iplot(fig)

formated_gdf = cleaned_data.groupby(['Report_Date_String', 'Country_Region'])['Confirmed'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Report_Date_String'] = pd.to_datetime(formated_gdf['Report_Date_String'])
formated_gdf['Report_Date_String'] = formated_gdf['Report_Date_String'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country_Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country_Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="Report_Date_String", 
                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
#fig.show()
plotly.offline.iplot(fig)

owid_covid_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
owid_covid_data_Selected = owid_covid_data[['date','new_cases','new_deaths']]
#Set 0 for NAN
#ColumnNan = ['new_cases','new_deaths']
#owid_covid_data_Selected[ColumnNan] = owid_covid_data_Selected[ColumnNan].fillna(0)
owid_covid_data_Selected.tail()
#owid_covid_data.head()
# W weekly frequency
owid_covid_newcases = owid_covid_data_Selected.groupby(['date'])['new_cases'].sum()
owid_covid_newcases = owid_covid_newcases.reset_index()
owid_covid_newcases['date'] = pd.to_datetime(owid_covid_newcases['date'])
owid_covid_newcases['date'] = owid_covid_newcases['date'].dt.strftime('%m/%d/%Y')
owid_covid_newcases.head()

owid_covid_newdeaths = owid_covid_data_Selected.groupby(['date'])['new_deaths'].sum()
owid_covid_newdeaths = owid_covid_newdeaths.reset_index()
owid_covid_newdeaths['date'] = pd.to_datetime(owid_covid_newdeaths['date'])
owid_covid_newdeaths['date'] = owid_covid_newdeaths['date'].dt.strftime('%m/%d/%Y')
owid_covid_newdeaths.head()
fig = go.Figure(data=[
    go.Line(x=owid_covid_newcases['date'], y=owid_covid_newcases['new_cases'],mode='lines',name='New Case'),
     go.Line(x=owid_covid_newdeaths['date'], y=owid_covid_newdeaths['new_deaths'],mode='lines',name='Death'),
])

fig.update_layout(
    title="TrendLine - World Death and new cases over the time",
    yaxis_title="New or Death Cases",
    xaxis_title="Date",
    showlegend=True
)
#fig.show()
plotly.offline.iplot(fig)

Top10_Countries = Top10_Countries_death[['Country_Region']]
Top10_Countries.loc[Top10_Countries['Country_Region'] == "UK", "Country_Region"] = "United Kingdom"
Top10_Countries.loc[Top10_Countries['Country_Region'] == "US", "Country_Region"] = "United States"
filter_list = Top10_Countries['Country_Region']. values. tolist()
owid_Top10_Countries = owid_covid_data.loc[owid_covid_data['location'].isin(filter_list)]

Top10_gdp_per_capita =  owid_Top10_Countries.groupby(['location'])['gdp_per_capita'].max().to_frame(name = 'gdp_per_capita').reset_index()
Top10_cvd_death_rate =  owid_Top10_Countries.groupby(['location'])['total_deaths'].max().to_frame(name = 'total_deaths').reset_index()
Top10_diabetes_prevalence =  owid_Top10_Countries.groupby(['location'])['diabetes_prevalence'].max().to_frame(name = 'diabetes_prevalence').reset_index()
Top10_female_smokers =  owid_Top10_Countries.groupby(['location'])['female_smokers'].max().to_frame(name = 'female_smokers').reset_index()
Top10_male_smokers =  owid_Top10_Countries.groupby(['location'])['male_smokers'].max().to_frame(name = 'male_smokers').reset_index()
Top10_hospital_beds_per_100k =  owid_Top10_Countries.groupby(['location'])['hospital_beds_per_thousand'].max().to_frame(name = 'hospital_beds_per_100k').reset_index()

Top10_cvd_death_rate.head()
Bottom10_Countries_death = TodaysData_Country.drop(['Last_Update', 'Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized','Mortality_Rate','UID','ISO3'], axis=1) 
Bottom10_Countries_death = Bottom10_Countries_death.loc[(Bottom10_Countries_death['Deaths'] >= 1)].nsmallest(10, 'Deaths')
Bottom10_Countries_death.head(10)
Bottom10_Countries = Bottom10_Countries_death[['Country_Region']]
Bottom10_Countries.loc[Bottom10_Countries['Country_Region'] == "UK", "Country_Region"] = "United Kingdom"
Bottom10_Countries.loc[Bottom10_Countries['Country_Region'] == "US", "Country_Region"] = "United States"
filter_list = Bottom10_Countries['Country_Region']. values. tolist()
owid_Bottom10_Countries = owid_covid_data.loc[owid_covid_data['location'].isin(filter_list)]

Bottom10_gdp_per_capita =  owid_Bottom10_Countries.groupby(['location'])['gdp_per_capita'].max().to_frame(name = 'gdp_per_capita').reset_index()
Bottom10_cvd_death_rate =  owid_Bottom10_Countries.groupby(['location'])['total_deaths'].max().to_frame(name = 'total_deaths').reset_index()
Bottom10_diabetes_prevalence =  owid_Bottom10_Countries.groupby(['location'])['diabetes_prevalence'].max().to_frame(name = 'diabetes_prevalence').reset_index()
Bottom10_female_smokers =  owid_Bottom10_Countries.groupby(['location'])['female_smokers'].max().to_frame(name = 'female_smokers').reset_index()
Bottom10_male_smokers =  owid_Bottom10_Countries.groupby(['location'])['male_smokers'].max().to_frame(name = 'male_smokers').reset_index()
Bottom10_hospital_beds_per_100k =  owid_Bottom10_Countries.groupby(['location'])['hospital_beds_per_thousand'].max().to_frame(name = 'hospital_beds_per_100k').reset_index()
Bottom10_gdp_per_capita.head(10)
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("Top 10 Countries with Death Cases","Bottom 10 Countries with Death Cases")
)

fig.add_trace(go.Bar(name='Deaths',text='Deaths', x=Top10_Countries_death['Country_Region'], y=Top10_Countries_death['Deaths']),
              row=1, col=1)

fig.add_trace(go.Bar(name='Deaths',text='Deaths', x=Bottom10_Countries_death['Country_Region'], y=Bottom10_Countries_death['Deaths']),
              row=1, col=2)
fig.update_layout(height=900,title_text="Comaprison - COVID Death Cases", showlegend=False)

#fig.show()
plotly.offline.iplot(fig)

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("GDP Comaprison Top 10 Countries with Death Cases VS Bottom 10","Diabetes Prevalence Comaprison Top 10 Countries with Death Cases VS Bottom 10", "Female smokers Comaprison Top 10 Countries with Death Cases VS Bottom 10", "Male smokers Comaprison Top 10 Countries with Death Cases VS Bottom 10","Hospital beds per 100k Comaprison Top 10 Countries with Death Cases VS Bottom 10", "Covid death rate Comaprison Top 10 Countries with Death Cases VS Bottom 10")
)

fig.add_trace(go.Bar(name='gdp_per_capita',text='gdp_per_capita', x=Top10_gdp_per_capita['location'], y=Top10_gdp_per_capita['gdp_per_capita']),
              row=1, col=1)

fig.add_trace(go.Bar(name='gdp_per_capita',text='gdp_per_capita', x=Bottom10_gdp_per_capita['location'], y=Bottom10_gdp_per_capita['gdp_per_capita']),
              row=1, col=1)

fig.add_trace(go.Bar(name='diabetes_prevalence',text='diabetes_prevalence', x=Top10_diabetes_prevalence['location'], y=Top10_diabetes_prevalence['diabetes_prevalence']),
              row=1, col=2)

fig.add_trace(go.Bar(name='diabetes_prevalence', text='diabetes_prevalence',x=Bottom10_diabetes_prevalence['location'], y=Bottom10_diabetes_prevalence['diabetes_prevalence']),
              row=1, col=2)




fig.add_trace(go.Bar(name='female_smokers',text='female_smokers', x=Top10_female_smokers['location'], y=Top10_female_smokers['female_smokers']),
              row=2, col=1)

fig.add_trace(go.Bar(name='female_smokers',text='female_smokers', x=Bottom10_female_smokers['location'], y=Bottom10_female_smokers['female_smokers']),
              row=2, col=1)

fig.add_trace(go.Bar(name='male_smokers',text='male_smokers', x=Top10_male_smokers['location'], y=Top10_male_smokers['male_smokers']),
              row=2, col=2)

fig.add_trace(go.Bar(name='male_smokers', text='male_smokers',x=Bottom10_male_smokers['location'], y=Bottom10_male_smokers['male_smokers']),
              row=2, col=2)




fig.add_trace(go.Bar(name='hospital_beds_per_100k',text='hospital_beds_per_100k', x=Top10_hospital_beds_per_100k['location'], y=Top10_hospital_beds_per_100k['hospital_beds_per_100k']),
              row=3, col=1)

fig.add_trace(go.Bar(name='hospital_beds_per_100k',text='hospital_beds_per_100k', x=Bottom10_hospital_beds_per_100k['location'], y=Bottom10_hospital_beds_per_100k['hospital_beds_per_100k']),
              row=3, col=1)

fig.add_trace(go.Bar(name='total_deaths',text='total_deaths', x=Top10_cvd_death_rate['location'], y=Top10_cvd_death_rate['total_deaths']),
              row=3, col=2)

fig.add_trace(go.Bar(name='total_deaths', text='total_deaths',x=Bottom10_cvd_death_rate['location'], y=Bottom10_cvd_death_rate['total_deaths']),
              row=3, col=2)


fig.update_layout(height=900,title_text="Comaprison - Top 10 Countries with Death Cases VS Bottom 10", showlegend=False)

#fig.show()
plotly.offline.iplot(fig)

#Select Coloumn to clean
ColumnToClean = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand']
#Replace the nan with emty string
owid_covid_data[['location']] = owid_covid_data[['location']].fillna('')
#Replace the Nan with 0
owid_covid_data[ColumnToClean] = owid_covid_data[ColumnToClean].fillna(0)
#Filter the data so we will get only overall world data
owid_covid_data = owid_covid_data.query('location=="World"' )
Data_For_Regression = pd.DataFrame(columns=['date','total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand'], data=owid_covid_data[['date','total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand']].values)
Data_For_Regression.head()
#set the index as date
Data_For_Regression['date'] = pd.to_datetime(Data_For_Regression['date'])
Data_For_Regression = Data_For_Regression.set_index('date')
Data_For_Regression.head()
#Plot the graph
Data_For_Regression['total_cases'].plot(figsize=(12,5), color="green")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()

Data_For_Regression['total_deaths'].plot(figsize=(12,5), color="red")
plt.xlabel('Date')
plt.ylabel('Death')
plt.show()

Data_For_Regression['new_cases'].plot(figsize=(12,5), color="blue")
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.show()

# pick total death as forecast column
forecast_col = 'total_deaths'

# Chosing 30 days as number of forecast days
forecast_out = int(30)
print('length =',len(Data_For_Regression), "and forecast_out =", forecast_out)
# Creating label by shifting 'total_deaths' according to 'forecast_out'
Data_For_Regression['temp'] = Data_For_Regression[forecast_col].shift(-forecast_out)
print(Data_For_Regression.head(2))
print('\n')
# verify rows with NAN in Label column 
print(Data_For_Regression.tail(2))
# Define features Matrix X by excluding the label column which we just created 
X = np.array(Data_For_Regression.drop(['temp'], 1))

# Using a feature in sklearn, preposessing to scale features
X = preprocessing.scale(X)
print(X[1,:])
# X contains last 'n= forecast_out' rows for which we don't have label data
# Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

X_forecast_out = X[-forecast_out:]
X = X[:-forecast_out]
print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X))
# Define vector y for the data we have prediction for
# make sure length of X and y are identical
y = np.array(Data_For_Regression['temp'])
y = y[:-forecast_out]
print('Length of y: ',len(y))
# (split into test and train data)
# test_size = 0.2 ==> 20% data is test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('length of X_train and x_test: ', len(X_train), len(X_test))
# Create linear regression object
lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)
# Test
accuracy = lr.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)
# Predict using our Model
forecast_prediction = lr.predict(X_forecast_out)
print(forecast_prediction)
last_date = Data_For_Regression.iloc[-1].name 
last_date
todays_date = datetime.strptime(last_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
todays_date = todays_date + timedelta(days=1)
todays_date = datetime.strptime(todays_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
index = pd.date_range(todays_date, periods=30, freq='D')
columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand','temp','forecast']
temp_df = pd.DataFrame(index=index, columns=columns)
temp_df
j=0
for i in forecast_prediction:
    temp_df.iat[j,12] = i
    j= j+1

temp_df
Data_For_Regression['total_deaths'].plot(figsize=(12,5), color="red")
temp_df['forecast'].plot(figsize=(12,5), color="orange")
plt.xlabel('Date')
plt.ylabel('Death')
plt.show()

owid_covid_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
owid_covid_data.head()

#Clean_Data = TodaysData_Country.drop([ 'Incident_Rate','People_Tested','People_Hospitalized','UID'], axis=1) 
#Clean_Data = Clean_Data[Clean_Data['Province_State'].isnull()]
#Clean_Data.tail()
Data_To_Process =pd.DataFrame(columns=['date','location', 'total_deaths','total_cases'], data=owid_covid_data[['date','location', 'total_deaths','total_cases']].values)
Data_To_Process.head()
#Get data for US


Data_To_Process = Data_To_Process.query('location=="United States"' )
Data_To_Process.tail(10)
from fbprophet import Prophet
cases = Data_To_Process.groupby('date').sum()['total_cases'].reset_index()
deaths = Data_To_Process.groupby('date').sum()['total_deaths'].reset_index()
#Clean the Data
# Prphet expect Dataframe with columns "ds" and "y" with the dates and values respectively
deaths.rename(columns={'Last_Update': 'date','total_deaths':'y'}, inplace=True)
deaths.columns = ['ds', 'y']
deaths.tail()
#Create the model using Prophet 0.95 confidence
ML_Model = Prophet(interval_width=0.95)
#Fit the Model
ML_Model.fit(deaths)
#Create prediction Data
Death_Prediction = ML_Model.make_future_dataframe(periods=60)
Death_Prediction.tail()
#predicting the future with date, and upper and lower limit of y value
Death_Forecast = ML_Model.predict(Death_Prediction)
Death_Forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
Death_forecasting = ML_Model.plot(Death_Forecast)
# Susceptible equation
def fa(N, a, b, beta):
    fa = -beta*a*b
    return fa

# Infected equation
def fb(N, a, b, beta, gamma):
    fb = beta*a*b - gamma*b
    return fb

# Recovered/deceased equation
def fc(N, b, gamma):
    fc = gamma*b
    return fc
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):
    a1 = fa(N, a, b, beta)*hs
    b1 = fb(N, a, b, beta, gamma)*hs
    c1 = fc(N, b, gamma)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(N, ak, bk, beta)*hs
    b2 = fb(N, ak, bk, beta, gamma)*hs
    c2 = fc(N, bk, gamma)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(N, ak, bk, beta)*hs
    b3 = fb(N, ak, bk, beta, gamma)*hs
    c3 = fc(N, bk, gamma)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(N, ak, bk, beta)*hs
    b4 = fb(N, ak, bk, beta, gamma)*hs
    c4 = fc(N, bk, gamma)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c
def SIR(N, b0, beta, gamma, hs):
    
    """
    N = total number of population
    beta = transition rate S->I
    gamma = transition rate I->R
    k =  denotes the constant degree distribution of the network (average value for networks in which 
    the probability of finding a node with a different connectivity decays exponentially fast
    hs = jump step of the numerical integration
    """
    
    # Initial condition
    a = float(N-1)/N -b0
    b = float(1)/N +b0
    c = 0.

    sus, inf, rec= [],[],[]
    for i in range(10000): # Run for a certain number of time-steps
        sus.append(a)
        inf.append(b)
        rec.append(c)
        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)

    return sus, inf, rec
N = 7800*(10**6)
b0 = 0
beta = 0.7
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR_example.png')
plt.show()
TodaysData_Country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv')

TodaysData_Country.head()
TodaysData_Country = TodaysData_Country[TodaysData_Country['Province_State'].isnull()]
TodaysData_Country = TodaysData_Country.query('Country_Region=="US"' )
ColumnToClean = ['Confirmed', 'Deaths', 'Recovered', 'Active']
# filling missing values 
TodaysData_Country[['Country_Region']] = TodaysData_Country[['Country_Region']].fillna('')
TodaysData_Country[ColumnToClean] = TodaysData_Country[ColumnToClean].fillna(0)
TodaysData_Country = TodaysData_Country.drop([ 'Country_Region','Last_Update','Delta_Recovered','Incident_Rate','People_Tested','People_Hospitalized','Province_State','FIPS','UID','iso3'], axis=1) 
#TodaysData_Country['Report_Date_String'] = pd.to_datetime(TodaysData_Country['Report_Date_String'],"%Y-%m-%d")
TodaysData_Country.head()
population = float(331002651) # US population
country_df = pd.DataFrame()
country_df['ConfirmedCases'] = TodaysData_Country.Confirmed.diff().fillna(0)
country_df = country_df[10:]
country_df['day_count'] = list(range(1,len(country_df)+1))

ydata = [i for i in country_df.ConfirmedCases]
xdata = country_df.day_count
ydata = np.array(ydata, dtype=float)
xdata = np.array(xdata, dtype=float)

N = population
inf0 = ydata[0]
sus0 = N - inf0
rec0 = 0.0

def sir_model(y, x, beta, gamma):
    sus = -beta * y[0] * y[1] / N
    rec = gamma * y[1]
    inf = -(sus + rec)
    return sus, inf, rec

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted)
plt.title("Fit of SIR model for US infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()
print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
owid_covid_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv',parse_dates=["date"], skiprows=[1])
owid_covid_data = owid_covid_data.query('location=="United States"' )
Data_To_Process =pd.DataFrame(columns=['date','deaths','cases'], data=owid_covid_data[['date', 'total_deaths','total_cases']].values)
Data_To_Process = Data_To_Process.sort_values(by='date', ascending=False)
Data_To_Process["deaths"] = Data_To_Process['deaths'].astype(str).astype(float)
data = Data_To_Process["deaths"].values[::-1]
# parameters
gamma = 1.0/9.0
sigma = 1.0/3.0
Predict_For = 30
#US Population
N = 331002647
#Data from OWID Data
beds_per_100k = 34.7

params_init_min_max = {"R_0_start": (3.0, 2.0, 5.0), "k": (2.5, 0.01, 5.0), "x0": (90, 0, 180), "R_0_end": (0.9, 0.3, 3.5),
                       "prob_I_to_H": (0.05, 0.01, 0.1), "prob_H_to_D": (0.5, 0.05, 0.8),
                       "s": (0.003, 0.001, 0.01)}  # form: {parameter: (initial guess, minimum value, max value)}
def Calculate_SEIHRD(y, t, beta, gamma, sigma, N, p_I_to_C, p_C_to_D, Hospital_Beds):
    S, E, I, H, R, D = y

    suspected = -beta(t) * I * S / N
    exposed = beta(t) * I * S / N - sigma * E
    infected = sigma * E - 1/12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    hosipitalized = 1/12.0 * p_I_to_C * I - 1/7.5 * p_C_to_D * min(Hospital_Beds(t), H) - max(0, H-Hospital_Beds(t)) - (1 - p_C_to_D) * 1/6.5 * min(Hospital_Beds(t), H)
    Recovered = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1/6.5 * min(Hospital_Beds(t), H)
    Death = 1/7.5 * p_C_to_D * min(Hospital_Beds(t), H) + max(0, H-Hospital_Beds(t))
    return suspected, exposed, infected, hosipitalized, Recovered, Death
def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end
def SEIHRDModel(days, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_H, prob_H_to_D, s):

    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
  
    
    def Hospital_Beds(t):
        beds_0 = beds_per_100k / 100_000 * N
        return beds_0 + s*beds_0*t  # 0.003

    y0 = N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(0, days-1, days)
    #Solve Differential Equations with ODEINT
    ret = odeint(Calculate_SEIHRD, y0, t, args=(beta, gamma, sigma, N, prob_I_to_H, prob_H_to_D, Hospital_Beds))
    S, E, I, H, R, D = ret.T
    R_0_over_time = [beta(i)/gamma for i in range(len(t))]

    return t, S, E, I, H, R, D, R_0_over_time, Hospital_Beds, prob_I_to_H, prob_H_to_D
days = Predict_For + len(data)
y_data = np.concatenate((np.zeros(Predict_For), data))

x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array
def fitter(x, R_0_start, k, x0, R_0_end, prob_I_to_H, prob_H_to_D, s):
    ret = SEIHRDModel(days, beds_per_100k, R_0_start, k, x0, R_0_end, prob_I_to_H, prob_H_to_D, s)
    return ret[6][x]
mod = lmfit.Model(fitter)
#The special syntax kwargs in function definitions in python is used to pass a keyworded, variable-length argument list.
for kwarg, (init, mini, maxi) in params_init_min_max.items():
    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

params = mod.make_params()
fit_method = "leastsq"
result = mod.fit(y_data, params, method="least_squares", x=x_data)
result.plot_fit(datafmt="-");