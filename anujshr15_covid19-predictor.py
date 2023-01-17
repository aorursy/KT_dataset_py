import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# import folium 
# from folium import plugins

plt.rcParams['figure.figsize'] = 15, 12

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit

from joblib import dump,load
df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',parse_dates=True)
df.shape
df.drop('Sno',axis=1,inplace=True)
df.head()
df.drop(['ConfirmedIndianNational','ConfirmedForeignNational'],axis=1,inplace=True)
df.head()
df.tail()
df.dropna(how='any',axis=0,inplace=True)
df['Cured']=df['Cured'].apply(lambda x: int(x))
df['Deaths']=df['Deaths'].apply(lambda x: int(x))

current_date=df['Date'].iat[-1]
total_so_far = df[df['Date']==current_date]['Confirmed'].sum()
print(f"Total cases in India as of {current_date} are {total_so_far}")
df['active_cases']=df['Confirmed']-df['Cured']-df['Deaths']
df[df['Date']==current_date][['State/UnionTerritory','Cured','Deaths','Confirmed']].sort_values('Confirmed',ascending=False).style.background_gradient(cmap='Reds').hide_index()

df[df['Date']==current_date][['State/UnionTerritory','active_cases']].sort_values('active_cases',ascending=False).rename(columns= {'active_cases':'Active Cases'}).style.background_gradient(cmap='Reds').hide_index()

f, ax = plt.subplots(figsize=(12, 10))
data = df[df['Date']==current_date][['State/UnionTerritory','Cured','Deaths','Confirmed','active_cases']]
data.sort_values('Confirmed',ascending=False,inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="State/UnionTerritory", data=data,label="Total", color="r")

sns.set_color_codes("muted")
sns.barplot(x="Cured", y="State/UnionTerritory", data=data, label="Cured", color="g")

max_cases=data['Confirmed'].iat[0]

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, max_cases), ylabel="",xlabel="Cases")
sns.despine(left=True, bottom=True)
datewise_df= df[['Date','Confirmed','Cured','Deaths']].groupby(['Date'],sort=False).sum().reset_index()


datewise_df
fig = go.Figure()
fig.add_trace(go.Scatter(x=datewise_df['Date'], y = datewise_df['Confirmed'], mode='lines+markers',name='Total Cases'))
fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',xaxis_title='Date',yaxis_title='Confirmed Cases',plot_bgcolor='rgb(230, 230, 230)')
fig.show()


# fig = go.Figure()
# fig.add_trace(go.Scatter(x=datewise_df['Date'], y = np.log10(datewise_df['Confirmed']), mode='lines+markers',name='Total Cases in Log Scale'))
# fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',xaxis_title='Date',yaxis_title='Confirmed Cases in Log Scale',plot_bgcolor='rgb(230, 230, 230)')
# fig.show()


fig = px.bar(datewise_df, x="Date", y="Confirmed", barmode='group', height=400)
fig.update_layout(title_text='Coronavirus Cases in India on daily basis',plot_bgcolor='rgb(230, 230, 230)')

fig.show()
age_df= pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
age_df.drop('Sno',axis=1,inplace=True)

age_df
individual_df= pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
individual_df.tail()
total_data= pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
total_data_ind=total_data[total_data['location']=='India']
total_data_ind.tail()
total_data.columns
fig = px.bar(total_data_ind, x=total_data_ind["date"], y=total_data_ind["new_tests"], barmode='group', height=400)
fig.update_layout(title_text='New tests in India',xaxis_title='Date',yaxis_title='New Tests',plot_bgcolor='rgb(230, 230, 230)')
fig.show()


import datetime
days = np.array([i for i in range(len(datewise_df['Date']))]).reshape(-1,1)
confirmed_cases=np.array(datewise_df['Confirmed']).reshape(-1,1)
cured_cases=np.array(datewise_df['Cured']).reshape(-1,1)
death_cases=np.array(datewise_df['Deaths']).reshape(-1,1)
days_in_future=20
future_forecast = np.array([i for i in range(len(datewise_df['Date'])+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-days_in_future]

start=datewise_df['Date'].iat[0]
start_date = datetime.datetime.strptime(start,'%d/%m/%y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%d/%m/%y'))
from sklearn.model_selection import train_test_split
X_train_confirmed,X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days,confirmed_cases,test_size=0.25,shuffle=False,stratify=None)

poly = PolynomialFeatures(degree=3)
poly.fit(X_train_confirmed)
poly_X_train_confirmed = poly.transform(X_train_confirmed)
poly_X_test_confirmed = poly. transform(X_test_confirmed)
poly_future_forecast = poly. transform(future_forecast)
param_grid={'poly__degree':[2,3,4,5,6,7,8]}
pipeline = Pipeline(steps=[('poly', PolynomialFeatures()), ('ridge', Ridge())])
tscv = TimeSeriesSplit(n_splits=2)
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_confirmed, y_train_confirmed)
test_linear_pred=grid_search.predict(X_test_confirmed)
print("MAE: ",mean_absolute_error(y_test_confirmed,test_linear_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test_confirmed,test_linear_pred)))
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data',f'Polynomial regression with d={grid_search.best_params_["poly__degree"]}'])
plt.show()
future_linear_pred=grid_search.predict(future_forecast)


pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_linear_pred).reshape(-1,)})


actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
fig.update_layout(title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
fig.show()
from IPython.display import display, Markdown
display(Markdown(f'## <font color="#661400">The number of cases might reach upto {int(pred_df["Cases"].iat[-1])} on {pred_df["Date"].iat[-1]} as per LinearRegression predictor</font>'))
from sklearn.svm import SVR
param_grid_svr = {'degree':[4,5,6]}
tscv = TimeSeriesSplit(n_splits=2)
grid_search = GridSearchCV(SVR(kernel='poly',gamma=0.01), param_grid_svr, cv=tscv,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,n_jobs=-1)
grid_search.fit(X_train_confirmed, y_train_confirmed)

test_svr_pred=grid_search.predict(X_test_confirmed)
print("MAE: ",mean_absolute_error(y_test_confirmed,test_svr_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test_confirmed,test_svr_pred)))
grid_search.best_params_
plt.plot(y_test_confirmed)
plt.plot(test_svr_pred)
plt.legend(['Test Data','SVR regression'])
plt.show()
future_svr_pred=grid_search.predict(future_forecast)


pred_df_svr=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_svr_pred).reshape(-1,)})
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_df_svr['Date'], y = pred_df_svr['Cases'] , mode='lines+markers',name='Prediction using SVR',line={'color':'red'}))
fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
fig.update_layout(title_text='Prediction of Coronavirus Cases in India (SVR)',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
fig.show()
from IPython.display import display, Markdown
display(Markdown(f'## <font color="#661400">The number of cases might reach upto {int(pred_df_svr["Cases"].iat[-1])} on {pred_df_svr["Date"].iat[-1]} as per support vector regression predictor</font>'))
#storing the model
svr_model=grid_search.best_estimator_
dump(svr_model,'svr_model.joblib')
svr_model=load('svr_model.joblib')
future_svr_pred=svr_model.predict(future_forecast)
pred_df_svr=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_svr_pred).reshape(-1,)})
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_df_svr['Date'], y = pred_df_svr['Cases'] , mode='lines+markers',name='Prediction using SVR',line={'color':'red'}))
fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
fig.update_layout(title_text='Prediction of Coronavirus Cases in India (SVR)',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
fig.show()
