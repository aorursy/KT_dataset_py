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
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

import numpy as np
import pandas as pd
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
#from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

color = sns.color_palette()
sns.set_style('darkgrid')

from numpy.random import seed
seed(1)


import tensorflow
tensorflow.random.set_seed(1)
age_details = pd.read_csv('/kaggle/input/agegroupdetailscsv/AgeGroupDetails.csv')
#india_covid_19 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
hospital_beds = pd.read_csv('/kaggle/input/hospitalbedsindiacsv/HospitalBedsIndia.csv')
#individual_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
#ICMR_details = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')
ICMR_labs = pd.read_csv('/kaggle/input/icmrtestinglabscsv/ICMRTestingLabs.csv')
state_testing = pd.read_csv('/kaggle/input/statewisetestingdetails-01csv/StatewiseTestingDetails.csv')
population = pd.read_csv('/kaggle/input/population-india-census2011-01csv/population_india_census2011.csv')

#world_population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
#confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
#recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
#latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')

#india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'],dayfirst = True)
state_testing['Date'] = pd.to_datetime(state_testing['Date'])
#ICMR_details['DateTime'] = pd.to_datetime(ICMR_details['DateTime'],dayfirst = True)
#ICMR_details = ICMR_details.dropna(subset=['TotalSamplesTested', 'TotalPositiveCases'])
covid_19_india=pd.read_csv('/kaggle/input/covid-19-india-06csv/covid_19_india_0.5.csv')
#covid_19_india=pd.read_csv('/kaggle/input/covid-19-india-04csv/covid_19_india_0.1.csv')

#covid_19_india=pd.read_csv('/kaggle/input/covid-19-india-may17csv/covid_19_india_May17.csv')

covid_19_india.tail()
individual_details=pd.read_csv('/kaggle/input/individualdetails-01csv/IndividualDetails.csv')
individual_details.head()
from collections import Counter
gender=individual_details.gender

gender.dropna(inplace=True)
gender=gender.value_counts()
per=[]
for i in gender:
    perc=i/gender.sum()
    per.append(format(perc,'.2f'))
plt.figure(figsize=(10,6))    
plt.title('Gender wise comparision',fontsize=20)
plt.pie(per,autopct='%1.1f%%')
plt.legend(gender.index,loc='best',title='Gender',fontsize=10)

print(gender.index)
agegroup=pd.read_csv('/kaggle/input/agegroupdetailscsv/AgeGroupDetails.csv')
agegroup.head()


perc=[]
for i in agegroup['Percentage']:
    per=float(re.findall("\d+\.\d+",i)[0])
    perc.append(per)
agegroup['Percentage']=perc
plt.figure(figsize=(20,10))
plt.title('Age-group case distribution',fontsize=20)
plt.pie(agegroup['Percentage'],autopct='%1.2f%%')
plt.legend(agegroup['AgeGroup'],loc='left',title='Age Group')
hospital_beds=pd.read_csv('/kaggle/input/hospitalbedsindiacsv/HospitalBedsIndia.csv')

top_20=hospital_beds.nlargest(20,'NumCommunityHealthCenters_HMIS')

top_20=hospital_beds.nlargest(20,'TotalPublicHealthFacilities_HMIS')
top_20=top_20[['State/UT','NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS'      
                    ,'NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS'                 
                    ,'NumRuralHospitals_NHP18' ,'NumUrbanHospitals_NHP18']]
sns.pairplot(top_20,hue='State/UT')

hospital_beds['Total_Beds'] = hospital_beds['NumPublicBeds_HMIS'] +  hospital_beds['NumRuralBeds_NHP18'] +  hospital_beds['NumUrbanBeds_NHP18']
hospital_beds
len(hospital_beds['State/UT'].values)
state = covid_19_india['State/UnionTerritory'].unique()
len(state)
covid_19_india.drop(['Date'], axis = 1, inplace = True)
covid_19_india.rename(columns={"Date.1": "Date"}, inplace = True)
covid_19_india.Date = pd.to_datetime(covid_19_india.Date, infer_datetime_format=True)
India_per_day = covid_19_india.groupby(["Date"])["Confirmed","Deaths", "Cured"].sum().reset_index().sort_values("Date", ascending = True)
#print (covid_19_india.loc[covid_19_india.last_valid_index()])
covid_19_india.tail(10)
covid_19_india['Active'] = covid_19_india['Confirmed'] - covid_19_india['Deaths'] - covid_19_india['Cured']
covid_19_india.tail(10)
kerala_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Kerala']

kerala_sort = kerala_covid.reset_index().sort_values("Date", ascending = True)
kerala_sort = kerala_sort.set_index("Date")
#kerala_sort.info()
plt.figure(figsize=(18,9))
plt.plot(kerala_sort.index, kerala_sort["Confirmed"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show();
kerala_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Kerala']

kerala_sort = kerala_covid.reset_index().sort_values("Date", ascending = True)
kerala_sort = kerala_sort.set_index("Date")
#kerala_sort.info()
plt.figure(figsize=(18,9))
plt.plot(kerala_sort.index, kerala_sort["Active"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Active Cases')
plt.show();
Delhi_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Delhi']

Delhi_sort = Delhi_covid.reset_index().sort_values("Date", ascending = True)
Delhi_sort = Delhi_sort.set_index("Date")
#Delhi_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Delhi_sort.index, Delhi_sort["Confirmed"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show();
Delhi_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Delhi']

Delhi_sort = Delhi_covid.reset_index().sort_values("Date", ascending = True)
Delhi_sort = Delhi_sort.set_index("Date")
#Delhi_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Delhi_sort.index, Delhi_sort["Active"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Active Cases')
plt.show();
Maharashtra_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Maharashtra']

Maharashtra_sort = Maharashtra_covid.reset_index().sort_values("Date", ascending = True)
Maharashtra_sort = Maharashtra_sort.set_index("Date")
#Maharashtra_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Maharashtra_sort.index, Maharashtra_sort["Confirmed"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show();
Maharashtra_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Maharashtra']

Maharashtra_sort = Maharashtra_covid.reset_index().sort_values("Date", ascending = True)
Maharashtra_sort = Maharashtra_sort.set_index("Date")
#Maharashtra_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Maharashtra_sort.index, Maharashtra_sort["Active"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Active Cases')
plt.show();
Haryana_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Haryana']

Haryana_sort = Haryana_covid.reset_index().sort_values("Date", ascending = True)
Haryana_sort = Haryana_sort.set_index("Date")
#Haryana_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Haryana_sort.index, Haryana_sort["Confirmed"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show();
Haryana_covid = covid_19_india[covid_19_india['State/UnionTerritory'] == 'Haryana']

Haryana_sort = Haryana_covid.reset_index().sort_values("Date", ascending = True)
Haryana_sort = Haryana_sort.set_index("Date")
#Haryana_sort.info()
plt.figure(figsize=(18,9))
plt.plot(Haryana_sort.index, Haryana_sort["Active"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Active Cases')
plt.show();
#Haryana_Last10_Days = 
#Haryana_sort[::-10]
Haryana_Last15_Days = Haryana_sort.tail(15)
Haryana_Last15_Days
plt.figure(figsize=(20,10))
plt.style.use('ggplot')
plt.title('Haryana Active cases',fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.xlabel('Date',fontsize=20)
#plt.ylabel('Confirmed Cases',fontsize=20)
plt.bar(Haryana_Last15_Days.index, Haryana_Last15_Days["Active"],edgecolor='black', color='blue',linewidth=3)
plt.show()
plt.figure(figsize=(20,10))
plt.style.use('ggplot')
plt.title('Haryana Confirmed cases',fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.xlabel('Date',fontsize=20)
#plt.ylabel('Confirmed Cases',fontsize=20)
plt.bar(Haryana_sort.index, Haryana_sort["Confirmed"],edgecolor='black',linewidth=3)
plt.show()
print(India_per_day.iloc[-1])
current = India_per_day.iloc[-1]
dead = current["Deaths"]
recov = current["Cured"]
act = current["Confirmed"] - current["Deaths"] - current["Cured"]
patient_state = [["Active",act],["Death",dead],["Recovered",recov]]
df = pd.DataFrame(patient_state, columns=["Patient State","Count"])
fig = px.pie(df, values="Count", names="Patient State", title="State of Patients in India", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
print(covid_19_india.loc[covid_19_india.last_valid_index()])

print(India_per_day.head())
print(India_per_day.tail())
print(India_per_day.shape)
India_per_day = covid_19_india.groupby(["Date"])["Confirmed","Deaths", "Cured"].sum().reset_index().sort_values("Date", ascending = True)
#India_per_day = covid_19_india.groupby(["Date"])["Confirmed","Deaths", "Cured"].sum().sort_values("Date", ascending = True)
India_per_day = India_per_day.sort_values(by='Date',ascending=True)
India_per_day.sort_index(inplace=True)
India_per_day
India_Last15_Days = India_per_day.tail(15)
India_Last15_Days

plt.figure(figsize=(20,10))
plt.style.use('ggplot')
plt.title('India Active cases',fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.xlabel('Date',fontsize=20)
#plt.ylabel('Confirmed Cases',fontsize=20)
plt.bar(India_Last15_Days.index, India_Last15_Days["Confirmed"] - India_Last15_Days["Deaths"] - India_Last15_Days["Cured"],edgecolor='black', color='red',linewidth=3)
plt.show()
plt.figure(figsize=(20,10))
plt.style.use('ggplot')
plt.title('Observed Cases',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel('Date',fontsize=20)
#plt.ylabel('Number of cases',fontsize=20)
plt.plot(India_per_day.index,India_per_day['Confirmed'],linewidth=3,label='Confirmed',color='black')
plt.plot(India_per_day.index,India_per_day['Cured'],linewidth=3,label='Cured',color='green')
plt.plot(India_per_day.index,India_per_day['Deaths'],linewidth=3,label='Death',color='red')
plt.plot(India_per_day.index,India_per_day['Confirmed'] - India_per_day['Cured'] - India_per_day['Deaths'],linewidth=3,label='Active',color='blue')
plt.legend(fontsize=20)
df2=covid_19_india[covid_19_india.Date == '06/06/2020'].groupby('State/UnionTerritory')[['Cured','Deaths','Confirmed']].sum()
#df2.Confirmed

df2=df2.nlargest(20,'Confirmed')
plt.figure(figsize=(20,10))
plt.title('top 20 states with confirmed cases',fontsize=30)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel('State',fontsize=20)
#plt.ylabel('Cases',fontsize=20)
plt.plot(df2.index,df2.Confirmed,marker='o',mfc='black',label='Confirmed',markersize=10,linewidth=5)
plt.plot(df2.index,df2.Deaths,marker='>',mfc='black',label='Deaths',markersize=10,linewidth=5)
plt.plot(df2.index,df2.Cured,marker='<',mfc='black',label='Cured',markersize=10,linewidth=5,color='green')
plt.plot(df2.index,df2.Confirmed - df2.Deaths - df2.Cured,marker='<',mfc='black',label='Active',markersize=10,linewidth=5,color='blue')
plt.legend(fontsize=20)
perc=[]
for i in df2.Confirmed:
    per=i/len(df2)
    perc.append(i)
plt.figure(figsize=(25,10))    
plt.title('Top 20 states with confirmed cases (Percentage distribution) ',fontsize=20)
plt.pie(perc,autopct='%1.1f%%')
plt.legend(df2.index,loc='right')
plt.figure(figsize=(30,40))
plt.subplot(311)
plt.title('Confirmed Cases',fontsize=30)
plt.xticks(rotation=90,fontsize=25)
plt.yticks(fontsize=25)
plt.bar(df2.index,df2.Confirmed,color='blue',linewidth=5,edgecolor='black')

plt.figure(figsize=(30,40))
plt.subplot(312)
plt.title('Active Cases',fontsize=30)
plt.xticks(rotation=90,fontsize=25)
plt.yticks(fontsize=25)
plt.bar(df2.index,df2.Confirmed - df2.Deaths - df2.Cured,color='green',linewidth=5,edgecolor='black')
plt.figure(figsize=(30,40))
plt.subplot(312)
plt.title('Cured Cases',fontsize=30)
plt.xticks(rotation=90,fontsize=25)
plt.yticks(fontsize=25)
plt.bar(df2.index,df2.Cured,color='green',linewidth=5,edgecolor='black')

plt.figure(figsize=(30,40))
plt.subplot(313)
plt.title('Deaths Cases',fontsize=30)
plt.xticks(rotation=90,fontsize=25)
plt.yticks(fontsize=25)
plt.bar(df2.index,df2.Deaths,color='red',linewidth=5,edgecolor='black')
df = India_per_day[['Date', 'Confirmed']]

df.reset_index()
df = df.set_index("Date")
df.info()
df.head()
df1 = df
df1 = df1.reset_index()
df.info()
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import math
import datetime

df1['time_stamp'] = df1.apply(
        lambda x: datetime.datetime.timestamp(x['Date']), axis=1)
df.head()
data = df1[['time_stamp', 'Confirmed']]
#data = df1[['time_stamp', 'Active']]

data.head()
X = data.iloc[:, 0:1].values 
y = data.iloc[:, 1].values

X = X / np.power(10, math.floor(math.log(X.max(), 10)))

plt.plot(X, y)


#------------------------------------------------------------------------------
# Fitting Linear Regression to the dataset (as a reference)
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X, y)

# The coefficients
print('Coefficients: \n', lin_reg_1.coef_)
y_pred_lin_reg_1 = lin_reg_1.predict(X)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred_lin_reg_1))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred_lin_reg_1))



# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
X_poly
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# The coefficients
print('Coefficients: \n', lin_reg_2.coef_)
y_pred_lin_reg_2 = lin_reg_2.predict(poly_reg.fit_transform(X))

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred_lin_reg_2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred_lin_reg_2))
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_1.predict(X), color = 'blue')
#plt.title('Truth or Bluff (Linear Regression)')
plt.title('Linear Regression')
#plt.xlabel('Time Stamp')
#plt.ylabel('Confirmed')
plt.xlim([min(X), max(X)])
plt.show()
# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
#plt.plot(X, lin_reg_2.predict(X_poly))
plt.title('Polynomial Regression')
#plt.xlabel('Time Stamp')
#plt.ylabel('Confirmed')
plt.xlim([min(X), max(X)])
plt.show()
plt.figure(figsize=(18,9))
plt.plot(df.index, df["Confirmed"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show()
#a = seasonal_decompose(df["Confirmed"], model = "add")
#a.plot()
# LSTM forecast
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# split the data into train and test set
train_data = df[:len(df)-10]
test_data = df[len(df)-10:]
train_data
# Scaling the information ...
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
#train_data.info()
data = pd.DataFrame(columns = ['ds','y'])
data['ds'] = train_data.index
data['y'] = scaled_train_data
#data['y'] = train_data.Confirmed.values
#train_data["Confirmed"].values
data.head()
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot


prop=Prophet()#growth = 'logistic')
prop.add_country_holidays(country_name='IND')
prop.fit(data)
future=prop.make_future_dataframe(periods=30, freq='D')
prop_forecast=prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(30)

forecast = scaler.inverse_transform(forecast.yhat.to_numpy().reshape(-1, 1))
fig = plot_plotly(prop, prop_forecast)
fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
plt.show()

#fig2 = prop.plot_components(forecast)
#plt.show()

#a = add_changepoints_to_plot(fig.gca(), prop, forecast)
forecast
#forecast = scaler.inverse_transform(lstm_predictions_scaled)
#forecast = scaler.inverse_transform(forecast.yhat.to_numpy().reshape(-1, 1))
#forecast

#fig2 = prop.plot_components(forecast)
#fig2

#from fbprophet.plot import plot_plotly
#import plotly.offline as py
#py.init_notebook_mode()

#fig = plot_plotly(prop, forecast)  # This returns a plotly Figure
#py.iplot(fig)
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf

n_input = 20
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
#lstm_model.add(LSTM(200, activation='tanh', input_shape=(n_input, n_features)))
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed = 123)

lstm_model.add(LSTM(200, activation='selu',kernel_initializer= 'RandomNormal', input_shape=(n_input, n_features)))
#lstm_model.add(LSTM(200, activation='selu',kernel_initializer= initializer, input_shape=(n_input, n_features)))

#lstm_model.add(LSTM(200, activation='relu', input_shape=(None)))
lstm_model.add(Dropout(0.15))
lstm_model.add(Dense(1))
#lstm_model.add(Dense(1))
#lstm_model.add(Dense(1))
lstm_model.compile(optimizer='Adadelta', loss='mse', metrics=['accuracy'])#, learning_rate=0.01)
# lstm_model.compile(optimizer='Adamax', loss='mse', learning_rate=0.002, beta_1=0.9, beta_2=0.999)
#lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator,epochs=20)
#lstm_model.fit(scaled_train_data,epochs=25)
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);
lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
#batch
current_batch = batch.reshape((1, n_input, n_features))
current_batch
#for i in range(len(test_data)):   
for i in range(30):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
lstm_predictions_scaled
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions
#test_data.drop('FB_PROP_Predicted_Cases',axis =1, inplace=True)
#test_data.drop('New_team',axis =1, inplace=True)
#print(forecast.yhat.to_numpy())
#test_data.assign(New_team = lambda x: forecast['yhat'])
test_data['LSTM_Predicted_Cases'] = lstm_predictions[:10]
test_data['FB_PROP_Predicted_Cases'] = forecast[:10]#forecast.yhat.to_numpy()[:10]
test_data
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_data['Confirmed'], test_data['FB_PROP_Predicted_Cases']))
import math
print(math.sqrt(mean_squared_error(test_data['Confirmed'], test_data['FB_PROP_Predicted_Cases'])))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_data['Confirmed'], test_data['FB_PROP_Predicted_Cases']))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_data['Confirmed'], test_data['LSTM_Predicted_Cases']))
import math
print(math.sqrt(mean_squared_error(test_data['Confirmed'], test_data['LSTM_Predicted_Cases'])))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_data['Confirmed'], test_data['LSTM_Predicted_Cases']))
test_data.head()
pd.plotting.register_matplotlib_converters()
test_data['FB_PROP_Predicted_Cases'].plot(figsize = (16,5), legend=True)
test_data['Confirmed'].plot(legend = True)
test_data['LSTM_Predicted_Cases'].plot(legend = True)
lstm_predictions_new = lstm_predictions[10:]
FB_Prop_predictions_new = forecast[10:]

#lstm_predictions_new.plot(figsize = (16,5), legend=True)
#FB_Prop_predictions_new.plot(legend = True)
FB_Prop_predictions_new
#covid_19_india_future=pd.read_csv('covid_19_india_future.csv')
covid_19_india_future=pd.read_csv('/kaggle/input/covid-19-india-future-jun07csv/covid_19_india_future_Jun07.csv')
#covid_19_india_future=pd.read_csv('/kaggle/input/covid-19-india-future-may18csv/covid_19_india_future_May18.csv')
covid_19_india_future
covid_19_india_future['LSTM_Predicted_Cases'] = lstm_predictions_new
covid_19_india_future['FB_PROP_Predicted_Cases'] = FB_Prop_predictions_new

covid_19_india_future
covid_19_india_future['LSTM_Predicted_Cases'].plot(figsize = (16,5), legend=True)
covid_19_india_future['FB_PROP_Predicted_Cases'].plot(legend = True)
hospital_beds =hospital_beds.drop([36])
cols_object = list(hospital_beds.columns[2:8])
cols_object
hospital_beds.fillna(hospital_beds.mean(), inplace=True)
hospital_beds
for cols in cols_object:
    hospital_beds[cols] = hospital_beds[cols].astype(int) #,errors = 'ignore')
top_10_primary = hospital_beds.nlargest(10,'NumPrimaryHealthCenters_HMIS')
top_10_community = hospital_beds.nlargest(10,'NumCommunityHealthCenters_HMIS')
top_10_district_hospitals = hospital_beds.nlargest(10,'NumDistrictHospitals_HMIS')
top_10_public_facility = hospital_beds.nlargest(10,'TotalPublicHealthFacilities_HMIS')
top_10_public_beds = hospital_beds.nlargest(10,'NumPublicBeds_HMIS')

plt.figure(figsize=(15,10))
plt.suptitle('Top 10 States in each Health Facility',fontsize=20)
plt.subplot(221)
plt.title('Primary Health Centers')
plt.barh(top_10_primary['State/UT'],top_10_primary['NumPrimaryHealthCenters_HMIS'],color ='#87479d');

plt.subplot(222)
plt.title('Community Health Centers')
plt.barh(top_10_community['State/UT'],top_10_community['NumCommunityHealthCenters_HMIS'],color = '#9370db');

plt.subplot(224)
plt.title('Total Public Health Facilities')
plt.barh(top_10_community['State/UT'],top_10_public_facility['TotalPublicHealthFacilities_HMIS'],color='#9370db');

plt.subplot(223)
plt.title('District Hospitals')
plt.barh(top_10_community['State/UT'],top_10_district_hospitals['NumDistrictHospitals_HMIS'],color = '#87479d');
world_data = pd.read_csv('/kaggle/input/full-datacsv/full_data.csv')

world_data.date = pd.to_datetime(world_data.date, infer_datetime_format=True)
world_data = world_data.sort_values(["date","total_cases"], ascending = True)
#world_data = world_data.sort_values("date", ascending = True)
world_data.rename(columns={"location": "Country"}, inplace = True)
world_data.tail(10)
# Get names of indexes for World only cases, store and drop ....
world_data_tot = world_data[world_data.Country == 'World'] 
world_data_tot.set_index(["date"], inplace = True)

indexNames = world_data[world_data.Country == 'World'].index

# Delete these row indexes from dataFrame
world_data.drop(indexNames , inplace=True)

# Ignoring USA data as of now ....
indexNames = world_data[world_data.Country == 'United States'].index

# Delete these row indexes from dataFrame
world_data.drop(indexNames , inplace=True)


world_data.tail(15)
world_data.set_index('date', inplace = True)

plt.figure(figsize=(18,9))
plt.plot(world_data_tot.index, world_data_tot["total_cases"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Confirmed Cases')
plt.show();
# Fetching top 15 countries data
world_data_top15 = world_data.tail(10)
world_data_top15 = world_data_top15.sort_values(["date","total_cases"], ascending = True)
#world_data_top15.set_index('date', inplace = True)
world_data_top15
world_data.info()
color_arr = ['b','g','r','c','m','y','k','w', 'chartreuse', 'burlywood']
arr = 0

plt.figure(figsize=(20,10))
plt.style.use('ggplot')
plt.title('Top-10 Countries Confirmed Cases',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for i in world_data_top15.Country.values:
    plt.plot(world_data[world_data.Country == i].index,world_data[world_data.Country == i]['total_cases'],linewidth=3,label=i,color=color_arr[arr])
    arr = arr + 1

plt.legend(fontsize=20)
