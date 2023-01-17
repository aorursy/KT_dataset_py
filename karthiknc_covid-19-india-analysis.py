import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

from scipy import stats
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression,LinearRegression
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,confusion_matrix,classification_report,roc_curve,auc
from sklearn import svm
from sklearn.svm import SVC,SVR

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
data = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
data.head(5)
#Understand the data
data = pd.DataFrame(data)
data.shape
data.columns
len(data.ConfirmedIndianNational)
len(data.Deaths)
len(data.Cured)
len(data.Date)
data.info()
data.describe()
data['ConfirmedIndianNational'].describe()
data.head(3)
#If needed we can also replace the name of column
data = data.rename(columns = {"State/UnionTerritory":"State",
                              "ConfirmedIndianNational":"Confirmed_Indian",
                              "ConfirmedForeignNational":"Confirmed_Foreginer",
                              "Cured":"Recovered"})
data.columns
#Drop the SNo or column
#ab = data.ix[:,1:]  Another method for droppping column
df = data.drop(['Sno','Time'],1)
df.columns
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
df['Date'] = pd.to_datetime(df['Date'])
#df['Confirmed_Total'] = df['Confirmed_Indian']+df['Confirmed_Foreginer']
df.columns
#missing values check
df.isna().sum()

df[df['Recovered'].isna()]
df[df['Deaths'].isna()]
df_per_day=df.groupby('Date')['Confirmed_Indian','Confirmed_Foreginer','Confirmed',
          'Deaths', 'Recovered'].sum()
df_per_day1=df.groupby('Date')['Confirmed_Indian','Confirmed_Foreginer','Confirmed',
          'Deaths', 'Recovered'].max()
#maximum number of cases
df_per_day['Confirmed'].max() 
df_per_day1['Confirmed'].max()

#minimum number of cases
df_per_day['Confirmed'].min()

#which day has max cases
df_per_day['Confirmed'].idxmax()

#which day has minimum cases
df_per_day['Confirmed'].idxmin()
#No of cases per country State

df.groupby(['State'])['Confirmed_Indian','Confirmed_Foreginer','Confirmed',
          'Deaths', 'Recovered'].max()
#no of cases per country by descending order
a=df.groupby(['State'])['Confirmed_Indian','Confirmed_Foreginer','Confirmed','Deaths', 'Recovered'].max().sort_values(by = 'Confirmed', ascending= False)

#how many countried affected
States = df['State'].unique()
len(df['State'].unique())
# WordCloud for Confirmed cases in Country
State = str(a.Confirmed)
cloud = WordCloud(max_words=70,background_color="white").generate(State)
plt.figure(figsize = (10,10))
plt.imshow(cloud, interpolation='Bilinear')
plt.axis("off")
plt.show()
plt.tight_layout()
## COVID-19 # Symptoms

symptoms={'symptom':['Fever','Dry cough','Fatigue','Sputum production',
                     'Loss of smell','Shortness of breath','Muscle pain or Joint pain',
                     'Sore throat','Headache','Chills','Nausea or vomiting',
                     'Nasal congestion','Diarrhoea','Haemoptysis','Conjunctival congestion']
,'percentage':[87.9,67.7,38.1,33.4,15,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

con_symptoms=pd.DataFrame(data=symptoms)
con_symptoms

# Graph for Symptoms and Percentage
plt.figure(figsize=(10,5))
plt.bar(con_symptoms['symptom'],con_symptoms['percentage'], color = 'm')
plt.legend()
plt.title('Conditions of Covid-19')
plt.xlabel('Symptoms')
plt.xticks(rotation=90)

# Pie plot for symptoms
plt.figure(figsize=(15,10))
plt.title('Symptoms of Coronavirus',fontsize=20) 
plt.pie(con_symptoms['percentage'],autopct='%1.1f%%')
plt.legend(symptoms['symptom'],loc='best')
plt.show()
#Graph for Cases observed per day

b = df.groupby(['Date'])['Recovered','Deaths','Confirmed',].sum().sort_values(by = 'Date', ascending = True)

import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
plt.plot(b['Confirmed'],'bo', label = 'Confirmed_Total', linewidth = 2, linestyle = ':')
plt.plot(b['Deaths'],'ro', label = 'Deaths',linewidth = 2, linestyle = '--',)
plt.plot(b['Recovered'],'go', label = 'Recovered',linewidth = 2,linestyle = '-.')
plt.title('Cases per day')
plt.xlabel('Dates')
#plt.xticks([0,9,19,29])
plt.ylabel('Cases')
plt.legend()
plt.show
# Cases observed per state
import seaborn as sns
c=df.groupby(['State'])['Confirmed','Deaths', 'Recovered'].max().sort_values(by = 'Confirmed', ascending= False)

c.head(45).plot.barh(color = ('m','r','g'), figsize = (10,10), width = 0.9)
plt.title('Cases per State')
plt.xlabel('States in India')
plt.ylabel('Cases', labelpad = 20)
plt.legend()
plt.show

#Cases in State in Stacked form
c.head(45).plot.barh(stacked = True, color = ('m','r','g'), figsize = (20,10))
plt.title('Cases per State')
plt.xlabel('State')
plt.ylabel('Cases')
plt.legend()
plt.show
#make another group for adding calculated columns
c = df.groupby(['State'])['Confirmed','Deaths', 'Recovered'].max().sort_values(by = 'Confirmed', ascending= False)

# percent Death Rate
c['Percent_Deaths'] = c['Deaths']/c['Confirmed']*100
c['Percent_Deaths']= round(c['Percent_Deaths'], 2)

#Death rate state wise
c['Percent_Deaths'].sort_values().head(55).plot.barh(figsize = (10,10), color = 'r')
plt.title('Death Rate')
plt.xlabel('State')
plt.ylabel('Percent Death Rate')
plt.legend()
plt.show
# percent Recoverey rate
c['Percent_Recovery']=c['Recovered']/c['Confirmed']*100
c['Percent_Recovery'] = round(c['Percent_Recovery'], 2)

# Percent recovery rate Country wise
c['Percent_Recovery'].sort_values().tail(55).plot.barh(figsize = (10,10), color = 'g')
plt.title('Recovery Rate')
plt.xlabel('State')
plt.ylabel('Percent Recovery Rate')
plt.legend()
plt.show
# Stacked plot for Recovery rate and Death rate
c[['Percent_Deaths', 'Percent_Recovery']].sort_values(by = 'Percent_Recovery', ascending = False).plot.barh(stacked = True, figsize = (10,10), color = ('r','g'))

c[['Percent_Deaths', 'Percent_Recovery']].sort_values(by = 'Percent_Deaths', ascending = False).plot.barh(stacked = True, figsize = (10,10), color = ('r','g'))

# Percent recovery and Percent Death rate Date wise in scatter plot
d = df.groupby(['Date'])['Confirmed','Deaths', 'Recovered'].sum().sort_values(by  ='Date',ascending= True)
# Active Cases in India Date Wise
d['Active'] = d['Confirmed']-d['Deaths']-d['Recovered']

d['Percent_Deaths'] = d['Deaths']/d['Confirmed']*100
d['Percent_Deaths']= round(d['Percent_Deaths'], 2)
d['Percent_Recovery']=d['Recovered']/d['Confirmed']*100
d['Percent_Recovery'] = round(d['Percent_Recovery'], 2)
d['Percent_Active']=d['Active']/d['Confirmed']*100
d['Percent_Active'] = round(d['Percent_Active'], 2)

plt.figure(figsize = (10,10))
plt.plot(d['Percent_Recovery'], 'b', label = 'Percent_Recovery')
plt.plot(d['Percent_Deaths'], 'r', label = 'Percent_Deaths')
plt.plot(d['Percent_Active'],'y',label = 'Percent_Active')
plt.title('Recovery Rate Vs Death Rate Vs Active Rate')
plt.xlabel('Date')
plt.ylabel('Percencent Rate')
plt.legend()
plt.show

# Datewise growth Rate
g = df.groupby(['Date'])['Confirmed','Deaths', 'Recovered'].sum().sort_values(by = 'Date', ascending = True)
print(g.iloc[-1])

increased_Confirmed=[]
increased_Recovered=[]
increased_Deaths=[]
z = 0
for z in range(g.shape[0]-1):
    increased_Confirmed.append(((g['Confirmed'].iloc[z+1])/g['Confirmed'].iloc[z]))
    increased_Recovered.append(((g['Recovered'].iloc[z+1])/g['Recovered'].iloc[z]))
    increased_Deaths.append(((g['Deaths'].iloc[z+1])/g['Deaths'].iloc[z]))
increased_Confirmed.insert(0,1)
increased_Recovered.insert(0,1)
increased_Deaths.insert(0,1)

plt.figure(figsize=(10,5))
plt.plot(g.index,increased_Confirmed,'bo',label="Growth Rate of Confirmed Cases",linestyle = ':')
plt.plot(g.index,increased_Recovered,'go',label="Growth Rate of Recovered Cases",linestyle = '-.')
plt.plot(g.index,increased_Deaths,'ro',label="Growth Rate of Death Cases",linestyle = '--')
plt.xticks(rotation=90)
plt.title("Datewise Growth Rate of different Types of Cases")
plt.ylabel("Growth Rate")
plt.xlabel("Date")
plt.legend()

# Daily increase in Case
g = df.groupby(['Date'])['Confirmed','Deaths', 'Recovered'].sum().sort_values(by = 'Date', ascending = True)

ts=g.reset_index().sort_values('Date')
Confirmed=ts.Confirmed
Deaths=ts.Deaths
Recovered=ts.Recovered
New_Confirmed=[Confirmed[0]]
New_Deaths=[Deaths[0]]
New_Recovered=[Recovered[0]]
for i in range(1,len(Confirmed)):
    New_Confirmed.append(Confirmed[i]-Confirmed[i-1])
    New_Deaths.append(Deaths[i]-Deaths[i-1])
    New_Recovered.append(Recovered[i]-Recovered[i-1])
ts['New_Confirmed']=New_Confirmed
ts['New_Deaths']=New_Deaths
ts['New_Recovered']=New_Recovered
ts.head()

plt.figure(figsize=(10,5))
plt.plot(ts['Date'],ts['New_Confirmed'],'bo',label="New Confirmed Cases", linestyle = ':')
plt.plot(ts['Date'],ts['New_Recovered'],'go',label="New Recovered Cases",linestyle = '-.')
plt.plot(ts['Date'],ts['New_Deaths'],'ro',label="New Death Cases",linestyle = '--')
plt.xticks(rotation=90)
plt.title("New Cases added each day")
plt.ylabel("Cases")
plt.xlabel("Date")
plt.legend()
## Date Time
from datetime import date
e = df.copy()
e = e.drop(['State','Confirmed_Indian','Confirmed_Foreginer'],1)
e =e.groupby('Date').sum().reset_index()
i = 0
e['Days'] = 1
#####RUN THIS ONLY ONE TIME #################
for ind in e.index: 
    e['Days'][ind] = i
    i=i+1
#############################################

# Select only required variables and make new data frame
f = e.ix[:,(3,4)]
f.head(3)

#taking value into two variables X and y
X = f.ix[:,1] # Predictor # No of Days
X.head(3)
X_matrix = X.values.reshape(-1,1)
y = f.ix[:,0] # Response Variable # Total Confirmed Cases
y.head(3)


# splitting of training and testing data
X_matrix_train,X_matrix_test,y_train,y_test = train_test_split(X_matrix,y, test_size = 0.15,shuffle=False)
len(X_matrix_train)
len(X_matrix_test)
len(y_train)
len(y_test)

X_matrix_train.shape
X_matrix_test.shape
y_train.shape
y_test.shape

# New data created for prediction
new_data=pd.DataFrame(data=[0,60,70,80,90,100,110,120,130,140,150],columns=['Days'])
new_data
new_data_matrix = new_data.values.reshape(-1,1)

# LINEAR REGRESSION #

# Actual Model with Confirmed cases and no of Days
linear_model = LinearRegression(normalize=True, fit_intercept=True)
linear_model.fit(X_matrix_train, y_train)
linear_model.score(X_matrix_train,y_train)
print(linear_model.intercept_)
print(linear_model.coef_)

#--Training Accuracy ---#
pred_y=linear_model.predict(X_matrix_train)
pred_y

print('MAE Training set:', mean_absolute_error(pred_y, y_train))
print('MSE Training set:',mean_squared_error(pred_y, y_train))
MSE_tlr = mean_squared_error(pred_y, y_train)
print('RMSE Training set:',np.sqrt(MSE_tlr)) 

# Testing Accuaracy#
y_pred = linear_model.predict(X_matrix_test)
y_pred

print('MAE Testing set:', mean_absolute_error(y_pred, y_test))
print('MSE Testing set:',mean_squared_error(y_pred, y_test)) 
MSE_lr = mean_squared_error(y_pred, y_test)
print('RMSE Testing set:',np.sqrt(MSE_lr))

# Prediction for unknow of future data created with new data matrix
linear_pred = linear_model.predict(new_data_matrix) #prediction of future days 60,70,80,90,100
linear_pred

# Plot for Linear Regression
plt.figure(figsize=(10,5))
plt.plot(f['Days'], f['Confirmed'])
plt.plot(new_data_matrix, linear_pred, linestyle='dashed', color='orange')
plt.title('# of Coronavirus Cases Over Time')
plt.xlabel('Days')
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])
plt.xticks([0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149])
plt.show()
#Linear Regrssion is not good fit for the data 

# SUPPORT VECTOR MACHINE #

svCT = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1).fit(X_matrix_train, y_train)
print(svCT)


#--Training Accuracy ---#
pred_svm_y=svCT.predict(X_matrix_train)
pred_svm_y

print('MAE training set:', mean_absolute_error(pred_svm_y, y_train))
print('MSE training set:',mean_squared_error(pred_svm_y, y_train))
MSE_tsvm = mean_squared_error(pred_svm_y, y_train)
print('RMSE training set:',np.sqrt(MSE_tsvm)) 

# Testing Accuaracy#
svm_y_pred = svCT.predict(X_matrix_test)
svm_y_pred

print('MAE testing set:', mean_absolute_error(svm_y_pred, y_test))
print('MSE testing set:',mean_squared_error(svm_y_pred, y_test)) 
MSE_svm = mean_squared_error(svm_y_pred, y_test)
print('RMSE testing set:',np.sqrt(MSE_svm))

plt.plot(svm_y_pred)
plt.plot(y_test)

svm_new_data_pred = svCT.predict(new_data_matrix) #prediction of future days 60,70,80,90,100
# Plot for SVM predictions

plt.figure(figsize=(10,5))
plt.plot(f['Days'], f['Confirmed'])
plt.plot(new_data_matrix, svm_new_data_pred,'mo', linestyle='dashed')
plt.title('Coronavirus Cases Over Time')
plt.xlabel('Days')
plt.ylabel('Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM Predictions'])
plt.xticks([0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149])
plt.show()
# Polynomial Regression #
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5)
poly_X_matrix_train = poly.fit_transform(X_matrix_train)
poly_X_matrix_test = poly.fit_transform(X_matrix_test)
poly_new_data_matrix = poly.fit_transform(new_data_matrix)

poly_linear_model = LinearRegression(normalize=True, fit_intercept=False)
poly_linear_model.fit(poly_X_matrix_train, y_train)

#--Training Accuracy ---#
pred_poly_y=poly_linear_model.predict(poly_X_matrix_train)
pred_poly_y

print('MAE training set:', mean_absolute_error(pred_poly_y, y_train))
print('MSE training set:',mean_squared_error(pred_poly_y, y_train))
MSE_tpr = mean_squared_error(pred_poly_y, y_train)
print('RMSE training set:',np.sqrt(MSE_tpr))

# Testing Accuaracy#
poly_y_pred = poly_linear_model.predict(poly_X_matrix_test)
poly_y_pred

print('MAE testing set:', mean_absolute_error(poly_y_pred, y_test))
print('MSE testing set:',mean_squared_error(poly_y_pred, y_test))
MSE_pr = mean_squared_error(poly_y_pred, y_test)
print('RMSE testing set:',np.sqrt(MSE_pr))

plt.plot(poly_y_pred)
plt.plot(y_test)

poly_new_data_pred = poly_linear_model.predict(poly_new_data_matrix) #prediction of future days 60,70,80,90,100

#Plot for polynomial regression
plt.figure(figsize=(10,5))
plt.plot(f['Days'], f['Confirmed'])
plt.plot(new_data_matrix, poly_new_data_pred, 'mo',linestyle='dashed')
plt.title('# of Coronavirus Cases Over Time')
plt.xlabel('Days')
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Poly Rgression Predictions'])
plt.xticks([0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149])
plt.show()


# Time Series Forecasts #
g = df.groupby(['Date'])['Confirmed','Deaths', 'Recovered'].sum().sort_values(by = 'Date', ascending = True)
X_TS_train, y_TS_test = train_test_split(g, test_size = 0.20,shuffle=False)
y_pred_TS = y_TS_test.copy()
model_scores=[]


from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(10, 5))
autocorrelation_plot(g["Confirmed"])

# Define Function #
def get_stationarity(timeseries):
# rolling statistics
    rolling_mean = timeseries.rolling(window=7).mean()
    rolling_std = timeseries.rolling(window=7).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

get_stationarity(X_TS_train["Confirmed"])

fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
import statsmodels.api as sm
results=sm.tsa.seasonal_decompose(X_TS_train["Confirmed"])
ax1.plot(results.trend)
ax2.plot(results.seasonal)
ax3.plot(results.resid)

log_series=np.log(X_TS_train["Confirmed"])
get_stationarity(log_series)

    
fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
import statsmodels.api as sm
results=sm.tsa.seasonal_decompose(log_series)
ax1.plot(results.trend)
ax2.plot(results.seasonal)
ax3.plot(results.resid)

movingavg = log_series.rolling(window = 2).mean()
logscaleminusmovingavg = log_series-movingavg
logscaleminusmovingavg=logscaleminusmovingavg.dropna()
get_stationarity(logscaleminusmovingavg)


fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
import statsmodels.api as sm
results=sm.tsa.seasonal_decompose(logscaleminusmovingavg)
ax1.plot(results.trend)
ax2.plot(results.seasonal)
ax3.plot(results.resid)

#Another method of making data stationary
rolling_mean_exp_decay = log_series.ewm(halflife=1, min_periods=0, adjust=True).mean()
df_log_exp_decay = log_series - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)

fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
import statsmodels.api as sm
results=sm.tsa.seasonal_decompose(df_log_exp_decay)
ax1.plot(results.trend)
ax2.plot(results.seasonal)
ax3.plot(results.resid)

#another way is substrating one point from other
df_log_shift = log_series - log_series.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)

fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,7))
import statsmodels.api as sm
results=sm.tsa.seasonal_decompose(df_log_shift)
ax1.plot(results.trend)
ax2.plot(results.seasonal)
ax3.plot(results.resid)

# we are using logscaleminusmovingavg ######
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(logscaleminusmovingavg, nlags = 2)
lag_pacf = pacf(logscaleminusmovingavg, nlags = 2, method = 'ols')


#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(logscaleminusmovingavg)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(logscaleminusmovingavg)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(logscaleminusmovingavg)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(logscaleminusmovingavg)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()

model_arima=ARIMA(log_series,(0,1,0))
model_arima_fit=model_arima.fit()
print(model_arima_fit.summary())
model_arima_fit.plot_predict(dynamic = False)
plt.show()

# Plot residual errors
residuals = pd.DataFrame(model_arima_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


model_arima_fit.plot_predict(1,100)
plt.show()

prediction_arima=model_arima_fit.forecast(len(y_TS_test))[0]
y_pred_TS["ARIMA Model Prediction"]=list(np.exp(prediction_arima))

model_scores.append(np.sqrt(mean_squared_error(list(y_TS_test["Confirmed"]),np.exp(prediction_arima))))
print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(list(y_TS_test["Confirmed"]),np.exp(prediction_arima))))
MSE_ts = np.sqrt(mean_squared_error(list(y_TS_test["Confirmed"]),np.exp(prediction_arima)))


plt.figure(figsize=(10,5))
plt.plot(X_TS_train.index,X_TS_train["Confirmed"],label="Train Set",marker='o')
plt.plot(y_TS_test.index,y_TS_test["Confirmed"],label="Validation Set",marker='*')
plt.plot(y_pred_TS["ARIMA Model Prediction"],label="ARIMA Model Prediction Set",marker='^')
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Confirmed Cases')
plt.xticks(rotation=90)
#Final 
#Comparision of results

scores = []
scores.append(np.sqrt(MSE_lr))
scores.append(np.sqrt(MSE_svm))
scores.append(np.sqrt(MSE_pr))
scores.append(MSE_ts)
scores

models=["Linear Regression","SVM","Polynomial Regression", "Time Series"]
final = pd.DataFrame(zip(models,scores),columns=["Models","RMSE"]).sort_values(["RMSE"])
final

#Polynomial Regression is most fit plot the preditction of confimred cases over time
