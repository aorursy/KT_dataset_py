import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
covid=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
covid.head()
covid.drop(["SNo"],1,inplace=True)
covid[covid['Province/State']=='Anhui']
print("Checking Data-type of each column:\n",covid.dtypes)
covid.info()
#Converting "Observation Date" into Datetime format
covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
#Grouping different types of cases as per the date
datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
print("Totol number of countries with Disease Spread: ",len(covid["Country/Region"].unique()))
datewise.iloc[-1]
print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])
datewise.shape
np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0])
plt.figure(figsize=(15,7))
sns.barplot(x=datewise.index.date, y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
plt.title("Distribution Plot for Active Cases Cases over Date")
plt.xticks(rotation=90)
datewise["WeekOfYear"]=datewise.index.weekofyear
datewise.head()
week_num=[]
weekwise_confirmed=[]
weekwise_recovered=[]
weekwise_deaths=[]
for i in list(datewise["WeekOfYear"].unique()):
    weekwise_confirmed.append(datewise[datewise["WeekOfYear"]==i]["Confirmed"].iloc[-1])
    weekwise_recovered.append(datewise[datewise["WeekOfYear"]==i]["Recovered"].iloc[-1])
    weekwise_deaths.append(datewise[datewise["WeekOfYear"]==i]["Deaths"].iloc[-1])
    week_num.append(i)
plt.figure(figsize=(8,5))
plt.plot(week_num,weekwise_confirmed,linewidth=3, label='Confirmed')
plt.plot(week_num,weekwise_recovered,linewidth=3, label = 'Recovered')
plt.plot(week_num,weekwise_deaths,linewidth=3, label = 'Death')
plt.ylabel("Number of Cases")
plt.xlabel("Week Number")
plt.title("Weekly progress of Different Types of Cases")
plt.legend()
datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100
datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100
datewise["Active Cases"]=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"]
datewise["Closed Cases"]=datewise["Recovered"]+datewise["Deaths"]
datewise.head()
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))
ax1.plot(datewise["Mortality Rate"],label='Mortality Rate',linewidth=3)
ax1.axhline(datewise["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")
ax1.set_ylabel("Mortality Rate")
ax1.set_xlabel("Timestamp")
ax1.set_title("Overall Datewise Mortality Rate")
ax1.legend()
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)

ax2.plot(datewise["Recovery Rate"],label="Recovery Rate",linewidth=3)
ax2.axhline(datewise["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")
ax2.set_ylabel("Recovery Rate")
ax2.set_xlabel("Timestamp")
ax2.set_title("Overall Datewise Recovery Rate")
ax2.legend()
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    
print("Average increase in number of Confirmed Cases every day: ")
print (np.round(datewise["Confirmed"].diff().fillna(0).mean()))
print("Average increase in number of Recovered Cases every day:")
print (np.round(datewise["Recovered"].diff().fillna(0).mean()))
print("Average increase in number of Deaths Cases every day")
print (np.round(datewise["Deaths"].diff().fillna(0).mean()))
plt.figure(figsize=(15,6))
plt.plot(datewise["Confirmed"].diff().fillna(0),label="Daily increase in Confiremd Cases",linewidth=3)
plt.plot(datewise["Recovered"].diff().fillna(0),label="Daily increase in Recovered Cases",linewidth=3)
plt.plot(datewise["Deaths"].diff().fillna(0),label="Daily increase in Death Cases",linewidth=3)
plt.xlabel("Timestamp")
plt.ylabel("Daily Increment")
plt.title("Daily increase in different Types of Cases Worldwide")
plt.xticks(rotation=90)
plt.legend()
datewise["Confirmed"]
datewise["Confirmed"].iloc[1]
datewise["Confirmed"].iloc[2]/datewise["Confirmed"].iloc[1]
#GROWTH FACTOR
daily_increase_confirm=[]
daily_increase_recovered=[]
daily_increase_deaths=[]
for i in range(datewise.shape[0]-1):
    daily_increase_confirm.append(((datewise["Confirmed"].iloc[i+1]/datewise["Confirmed"].iloc[i])))
    daily_increase_recovered.append(((datewise["Recovered"].iloc[i+1]/datewise["Recovered"].iloc[i])))
    daily_increase_deaths.append(((datewise["Deaths"].iloc[i+1]/datewise["Deaths"].iloc[i])))
daily_increase_confirm.insert(0,1)
daily_increase_recovered.insert(0,1)
daily_increase_deaths.insert(0,1)
plt.figure(figsize=(15,7))
plt.plot(datewise.index,daily_increase_confirm,label="Growth Factor Confiremd Cases",linewidth=3)
plt.plot(datewise.index,daily_increase_recovered,label="Growth Factor Recovered Cases",linewidth=3)
plt.plot(datewise.index,daily_increase_deaths,label="Growth Factor Death Cases",linewidth=3)
plt.xlabel("Timestamp")
plt.ylabel("Growth Factor")
plt.title("Growth Factor of different Types of Cases Worldwide")
plt.axhline(1,linestyle='--',color='black',label="Baseline")
plt.xticks(rotation=90)
plt.legend()
datewise["Confirmed"].iloc[[-1]]
c=1000
double_days=[]
C=[]
while(1):
    double_days.append(datewise[datewise["Confirmed"]<=c].iloc[[-1]]["Days Since"][0])
    C.append(c)
    c=c*2
    if(c<datewise["Confirmed"].max()):
        continue
    else:
        break
doubling_rate=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first Case"])
doubling_rate["Number of days for doubling"]=doubling_rate["Days since first Case"].diff().fillna(doubling_rate["Days since first Case"])
#doubling_rate
plt.figure(figsize=(10,5))
plt.plot(doubling_rate["No. of cases"],doubling_rate["Number of days for doubling"].dt.days,marker='o')
plt.axhline(1,color='black',linestyle='--')
plt.ylabel("Number of days for doubling")
plt.xlabel("Number of Confirmed Cases")
plt.title("Days required for rise in cases by double")
doubling_rate.head()
covid.head()
#Calculating countrywise Moratality and Recovery Rate
countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()]
countrywise = countrywise.groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
countrywise = countrywise.sort_values(["Confirmed"],ascending=False)
countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
countrywise.head()
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,12))
top_15_confirmed=countrywise.head(15)
top_15_deaths=countrywise.sort_values(["Deaths"],ascending=False).head(15)
sns.barplot(x=top_15_confirmed["Confirmed"],
            y=top_15_confirmed.index,
            ax=ax1)
ax1.set_title("Top 15 countries as per Number of Confirmed Cases")
sns.barplot(x=top_15_deaths["Deaths"],y=top_15_deaths.index,ax=ax2)
ax2.set_title("Top 15 countries as per Number of Death Cases")
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]
grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])
grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
plt.figure(figsize=(15,10))
for country in countrywise.head(10).index:
    sns.lineplot(x=grouped_country.loc[country]["log_confirmed"],y=grouped_country.loc[country]["log_active"],
                 label=country,linewidth=3)
plt.xlabel("Confirmed Cases (Logrithmic Scale)")
plt.ylabel("Active Cases (Logarithmic Scale)")
plt.title("COVID-19 Journey of Top 10 countries having Highest number of Confirmed Cases")
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from fbprophet import Prophet

std=StandardScaler()
X=countrywise[["Confirmed","Recovered","Deaths"]]
X=std.fit_transform(X)
countrywise[["Confirmed","Recovered","Deaths"]].head()
X
silhouette_score
wcss=[]
sil=[]
for i in range(2,11):
    clf=KMeans(n_clusters=i,init='k-means++',random_state=42)
    clf.fit(X)
    labels=clf.labels_
    centroids=clf.cluster_centers_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
    wcss.append(clf.inertia_)
wcss
sil
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,5))
x=np.arange(2,11)
ax1.plot(x,wcss,marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Within Cluster Sum of Squares (WCSS)")
ax1.set_title("Elbow Method")
x=np.arange(2,11)
ax2.plot(x,sil,marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score Method")
#Correct number of Clusters could be k=4
clf_final=KMeans(n_clusters=5,init='k-means++',random_state=42)
clf_final.fit(X)
countrywise["Clusters"]=clf_final.predict(X)
countrywise["Clusters"]
cluster_summary=pd.concat([countrywise[countrywise["Clusters"]==1],
                           countrywise[countrywise["Clusters"]==2],
                           countrywise[countrywise["Clusters"]==3],
                           countrywise[countrywise["Clusters"]==4],
                           countrywise[countrywise["Clusters"]==0].head(15)])
cluster_summary.style.background_gradient(cmap='Reds')
plt.figure(figsize=(10,5))
sns.scatterplot(x=countrywise["Confirmed"],y=countrywise["Deaths"],hue=countrywise["Clusters"],s=100)
datewise.iloc[0]
datewise["Days Since"]=datewise.index-datewise.index[0]
datewise["Days Since"]=datewise["Days Since"].dt.days
datewise.head()
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
model_scores=[]
lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),
            np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg = lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
prediction_valid_linreg
valid_ml["Confirmed"]
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],
                                               prediction_valid_linreg)))
np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg))
plt.figure(figsize=(11,6))
prediction_linreg=lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["Confirmed"],label="Actual Confirmed Cases")
plt.plot(datewise.index,prediction_linreg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
#The Linear Regression Model is absolutely falling aprat. As it is clearly visible that the trend of Confirmed Cases in absolutely not Linear
poly = PolynomialFeatures(degree = 6)
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]
train_poly
linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,predictions_poly, linestyle='--',label="Best Fit for Polynomial Regression",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Polynomial Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
new_prediction_poly=[]
for i in range(1,18):
    new_date_poly=poly.fit_transform(np.array(datewise["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
new_prediction_poly
#Support Vector Machine ModelRegressor for Prediction of Confirmed Cases
train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
#Intializing SVR Model
#Fitting model on the training data
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),
        np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm))
plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
plt.plot(datewise["Confirmed"],label="Train Confirmed Cases",linewidth=3)
plt.plot(datewise.index,prediction_svm, linestyle='--',label="Best Fit for SVR",color='black')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases Support Vector Machine Regressor Prediction")
plt.xticks(rotation=90)
plt.legend()
new_date=[]
new_prediction_lr=[]
new_prediction_svm=[]
for i in range(1,18):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option('display.float_format', lambda x: '%.6f' % x)
model_predictions=pd.DataFrame(zip(new_date, 
                                   new_prediction_lr, 
                                   new_prediction_poly,
                                   new_prediction_svm),
                               columns=["Dates","Linear Regression Prediction",
                                        "Polynonmial Regression Prediction","SVM Prediction"])
model_predictions.head()
# Time Series Forecasting
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
Holt
holt=Holt(np.asarray(model_train["Confirmed"]))
holt = holt.fit(smoothing_level=1.0, smoothing_slope=0.1,optimized=False)
y_pred=valid.copy()
len(valid)
holt.__dir__
y_pred["Holt"]=holt.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"]))
y_pred.Holt
plt.figure(figsize=(10,5))
plt.plot(model_train.Confirmed,label="Train Set",linewidth=3)
valid.Confirmed.plot(label="Validation Set",linewidth=3)
y_pred.Holt.plot(label="Holt's Linear Model Predicted Set",linewidth=3)
plt.ylabel("Confirmed Cases")
plt.xlabel("Date Time")
plt.title("Confirmed Holt's Linear Model Prediction")
plt.xticks(rotation=90)
plt.legend()
holt_new_date=[]
holt_new_prediction=[]
for i in range(1,18):
    holt_new_date.append(datewise.index[-1]+timedelta(days=i))
    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])
model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction
model_predictions.head()
## Holt's Winter Model for Daily Time Series
ExponentialSmoothing
model_train
es=ExponentialSmoothing(np.asarray(model_train['Confirmed']),
                        seasonal_periods=10,
                        trend='mul', 
                        seasonal='add').fit()
y_pred["Holt's Winter Model"]=es.forecast(len(valid))
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"])))
np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"]))
plt.figure(figsize=(10,5))
plt.plot(model_train.Confirmed,label="Train Set",linewidth=3)
valid.Confirmed.plot(label="Validation Set",linewidth=3)
y_pred["Holt\'s Winter Model"].plot(label="Holt's Winter Model Predicted Set",linewidth=3)
plt.ylabel("Confirmed Cases")
plt.xlabel("Date Time")
plt.title("Confiremd Cases Holt's Winter Model Prediction")
plt.xticks(rotation=90)
plt.legend()
holt_winter_new_prediction=[]
for i in range(1,18):
    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])
model_predictions["Holt's Winter Model Prediction"]=holt_winter_new_prediction
model_predictions.head()
#AR Model (using AUTO ARIMA)
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred=valid.copy()
!pip install pmdarima
from pmdarima.arima import auto_arima
model_ma = auto_arima(model_train["Confirmed"], trace=True, 
                     error_action='ignore', 
                     start_p=0, start_q=0, max_p=0, max_q=3,
                     suppress_warnings=True,
                     stepwise=False,
                     seasonal=False)
model_ma.fit(model_train["Confirmed"])
auto_arima
model_ar= auto_arima(model_train["Confirmed"],
                     trace=True, 
                     error_action='ignore',
                     start_p=0,
                     start_q=0,
                     max_p=5,
                     max_q=0,
                    suppress_warnings=True,stepwise=False,seasonal=False)
model_ar.fit(model_train["Confirmed"])
prediction_ar=model_ar.predict(len(valid))
y_pred["AR Model Prediction"]=prediction_ar
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))
np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"]))
plt.figure(figsize=(10,5))
plt.plot(model_train.index,model_train["Confirmed"],label="Train Set",linewidth=3)
plt.plot(valid.index,valid["Confirmed"],
         label="Validation Set",linewidth=3)
plt.plot(y_pred["AR Model Prediction"],label="AR Model Prediction set",linewidth=3)
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases AR Model Forecasting")
plt.xticks(rotation=90)
AR_model_new_prediction=[]
for i in range(1,18):
    AR_model_new_prediction.append(model_ar.predict(len(valid)+i)[-1])
model_predictions["AR Model Prediction"]=AR_model_new_prediction
model_predictions.head()
## MA Model (using AUTO ARIMA)
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred=valid.copy()
model_ma= auto_arima(model_train["Confirmed"],
                     trace=True,
                     error_action='ignore',
                     start_p=0,
                     start_q=0,
                     max_p=0,
                     max_q=3,
                     suppress_warnings=True,stepwise=False,seasonal=False)
model_ma.fit(model_train["Confirmed"])
prediction_ma=model_ma.predict(len(valid))
y_pred["MA Model Prediction"]=prediction_ma
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma)))
np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma))
plt.figure(figsize=(10,5))
plt.plot(model_train.index,model_train["Confirmed"],label="Train Set",linewidth=3)
plt.plot(valid.index,valid["Confirmed"],label="Validation Set",linewidth=3)
plt.plot(y_pred["MA Model Prediction"],label="MA Mode Prediction Set",linewidth=3)
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases MA Model Forecasting")
plt.xticks(rotation=90)
MA_model_new_prediction=[]
for i in range(1,18):
    MA_model_new_prediction.append(model_ma.predict(len(valid)+i)[-1])
model_predictions["MA Model Prediction"]=MA_model_new_prediction
model_predictions.head()
# ARIMA Model (using AUTOARIMA)
model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred=valid.copy()
model_arima= auto_arima(model_train["Confirmed"],
                        trace=True, 
                        error_action='ignore', 
                        start_p=1,
                        start_q=1,
                        max_p=2,
                        max_q=2,
                        suppress_warnings=True,stepwise=False,seasonal=False)
model_arima.fit(model_train["Confirmed"])
prediction_arima=model_arima.predict(len(valid))
y_pred["ARIMA Model Prediction"]=prediction_arima
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))
np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima))
plt.figure(figsize=(10,5))
plt.plot(model_train.index,model_train["Confirmed"],label="Train Set",linewidth=3)
plt.plot(valid.index,valid["Confirmed"],label="Validation Set",linewidth=3)
plt.plot(y_pred["ARIMA Model Prediction"],label="ARIMA Model Prediction Set",linewidth=3)
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases ARIMA Model Forecasting")
plt.xticks(rotation=90)
ARIMA_model_new_prediction=[]
for i in range(1,18):
    ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+i)[-1])
model_predictions["ARIMA Model Prediction"]=ARIMA_model_new_prediction
model_predictions.head()
## SARIMA Model (using AUTO ARIMA)

model_sarima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', 
                         start_p=0,start_q=0,max_p=3,max_q=3,m=24,
                   suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima.fit(model_train["Confirmed"])
prediction_sarima=model_sarima.predict(len(valid))
y_pred["SARIMA Model Prediction"]=prediction_sarima
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["SARIMA Model Prediction"])))
np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["SARIMA Model Prediction"]))
plt.figure(figsize=(10,5))
plt.plot(model_train.index,model_train["Confirmed"],label="Train Set",linewidth=3)
plt.plot(valid.index,valid["Confirmed"],label="Validation Set",linewidth=3)
plt.plot(y_pred["SARIMA Model Prediction"],label="SARIMA Model Prediction Set",linewidth=3)
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Confirmed Cases')
plt.title("Confirmed Cases SARIMA Model Forecasting")
plt.xticks(rotation=90)
SARIMA_model_new_prediction=[]
for i in range(1,18):
    SARIMA_model_new_prediction.append(model_sarima.predict(len(valid)+i)[-1])
model_predictions["SARIMA Model Prediction"]=SARIMA_model_new_prediction
model_predictions.head()