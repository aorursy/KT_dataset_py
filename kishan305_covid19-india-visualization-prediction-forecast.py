import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import plotly.graph_objects as go

import plotly.express as px
from IPython.core.display import HTML

HTML('''<div class="flourish-embed flourish-cards" data-src="visualisation/2557727" data-url="https://flo.uri.sh/visualisation/2557727/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
data.info() #Checking for Null and DataTypes
data['ObservationDate'] = pd.to_datetime(data['ObservationDate']) #Convertong Observation Date into Datetime format
data_india = data[data['Country/Region'] == 'India'] #Extracting Indian Data

data_india = data_india.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}) #Aggregating Attributes

data_india["WeekofYear"]=data_india.index.weekofyear #Adding new WEEK Column

data_india["Days Since"]=(data_india.index-data_india.index[0])

data_india["Days Since"]=data_india["Days Since"].dt.days

data_india["Active"] = data_india["Confirmed"] - data_india["Recovered"] - data_india["Deaths"]

data_india = data_india.reset_index()

data_india.head()
fig = px.bar(data_india, x='ObservationDate', y='Confirmed',color='Confirmed', height=500)

fig.update_layout(title='Confirmed Cases in India',

                 xaxis_title="Date",

                 yaxis_title="Confirmed Cases")
print("Number of Confirmed Cases as of ",data_india["ObservationDate"].iloc[-1], " is ",data_india["Confirmed"].iloc[-1])
fig = px.bar(data_india, x='ObservationDate', y='Deaths',color='Deaths',template='ggplot2', height=500)

fig.update_layout(title='Deaths in India',

                 xaxis_title="Date",

                 yaxis_title="Deaths")
print("Number of Confirmed Deaths as of ",data_india["ObservationDate"].iloc[-1], " is ",data_india["Deaths"].iloc[-1])
fig = px.bar(data_india, x='ObservationDate', y='Recovered',color='Recovered',template='plotly_white', height=500)

fig.update_layout(title='Recovered Cases in India',

                 xaxis_title="Date",

                 yaxis_title="Recovered Cases")
print("Number of Recovered Cases as of ",data_india["ObservationDate"].iloc[-1], " is ",data_india["Recovered"].iloc[-1])
fig = px.bar(data_india, x='ObservationDate', y='Active',color='Active',template='plotly_white', height=500)

fig.update_layout(title='Active Cases in India',

                 xaxis_title="Date",

                 yaxis_title="Active Cases")
print("Number of Active Cases as of ",data_india["ObservationDate"].iloc[-1], " is ",data_india["Active"].iloc[-1])
fig=go.Figure()

fig.add_trace(go.Scatter(x=data_india['ObservationDate'], y=data_india["Confirmed"],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=data_india['ObservationDate'], y=data_india["Recovered"],

                    mode='lines+markers',

                    name='Recovered Cases'))

fig.add_trace(go.Scatter(x=data_india['ObservationDate'], y=data_india["Deaths"],

                    mode='lines+markers',

                    name='Death Cases'))

#fig.update_layout(title="Confirmed vs Recovered vs Deaths due to CORONA in India",

 #                xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
cases = 1

double_days=[]

C=[]

while(1):

    double_days.append(int(data_india[data_india["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))

    C.append(cases)

    cases=cases*2

    if(cases<data_india["Confirmed"].max()):

        continue

    else:

        break

        
india_doubling=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first case"])

india_doubling["Number of days required to Double the cases"]=india_doubling["Days since first case"].diff().fillna(india_doubling["Days since first case"].iloc[0])

india_doubling.loc[india_doubling['No. of cases']==1, 'Number of days required to Double the cases'] = 0

india_doubling.style.background_gradient(cmap='Reds')
data_india['Active'] = data_india['Confirmed'] - data_india['Recovered'] - data_india['Deaths']

fig=go.Figure(data=go.Pie(labels=['Active','Recovered','Deaths'],

                values=[data_india.iloc[data_india['ObservationDate'].idxmax(axis=1)]['Active'],

                        data_india.iloc[data_india['ObservationDate'].idxmax(axis=1)]['Recovered'],

                        data_india.iloc[data_india['ObservationDate'].idxmax(axis=1)]['Deaths']

                       ]),layout={'template':'presentation'})

fig.update_layout(title_text="Coronavirus Cases in India as of "+data_india['ObservationDate'].max().strftime("%d-%b'%y"))

fig.show()
columns=['Active','Recovered','Deaths']

meltedDF=pd.melt(data_india[columns[::-1]+['ObservationDate']],id_vars=['ObservationDate'], var_name='Value Type', value_name='Share Percentage')

fig = px.bar(meltedDF, 

       x = "Share Percentage",

       animation_frame = meltedDF['ObservationDate'].astype(str), 

       color = 'Value Type', 

       barmode = 'stack', height=400,

       template='seaborn',

       title='Cases percentage share over time',

       orientation='h')

fig.show()
data_india.head()
train_ml=data_india.iloc[:int(data_india.shape[0]*0.95)]

valid_ml=data_india.iloc[int(data_india.shape[0]*0.95):]

model_scores=[]
Confirmed = valid_ml['Confirmed'].reset_index(drop=True)
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression(normalize=True)



lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
Prediction_Linear_Regression = prediction_valid_linreg.tolist()
from sklearn.metrics import mean_squared_error,r2_score

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))

print("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
plt.figure(figsize=(11,6))

prediction_linreg=lin_reg.predict(np.array(data_india["Days Since"]).reshape(-1,1))

linreg_output=[]

for i in range(prediction_linreg.shape[0]):

    linreg_output.append(prediction_linreg[i][0])



fig=go.Figure()

fig.add_trace(go.Scatter(x=data_india.index, y=data_india["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=data_india.index, y=linreg_output,

                    mode='lines',name="Linear Regression Best Fit Line",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Linear Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 10) 
train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))

valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))

y=train_ml["Confirmed"]
linreg=LinearRegression(normalize=True)

linreg.fit(train_poly,y)
prediction_poly=linreg.predict(valid_poly)

rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))

model_scores.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
Prediction_Polynomial_Regression = prediction_poly.tolist()
comp_data=poly.fit_transform(np.array(data_india["Days Since"]).reshape(-1,1))

plt.figure(figsize=(11,6))

predictions_poly=linreg.predict(comp_data)



fig=go.Figure()

fig.add_trace(go.Scatter(x=data_india.index, y=data_india["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=data_india.index, y=predictions_poly,

                    mode='lines',name="Polynomial Regression Best Fit",

                   line=dict(color='black', dash='dot' )))

fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",

                 legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
from sklearn.svm import SVR
train_ml=data_india.iloc[:int(data_india.shape[0]*0.95)]

valid_ml=data_india.iloc[int(data_india.shape[0]*0.95):]
#Intializing SVR Model

svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)



#Fitting model on the training data

svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))



prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))

print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
Prediction_SVM_Regression = prediction_valid_svm.tolist()
plt.figure(figsize=(11,6))

prediction_svm=svm.predict(np.array(data_india["Days Since"]).reshape(-1,1))

fig=go.Figure()

fig.add_trace(go.Scatter(x=data_india.index, y=data_india["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=data_india.index, y=prediction_svm,

                    mode='lines',name="Support Vector Machine Best fit Kernel",

                    line=dict(color='black', dash='dot')))

fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
from fbprophet import Prophet



prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)

prophet_confirmed=pd.DataFrame(zip(list(data_india['ObservationDate']),list(data_india["Confirmed"])),columns=['ds','y'])
prophet_c.fit(prophet_confirmed)





forecast_c=prophet_c.make_future_dataframe(periods=17)

forecast_confirmed=forecast_c.copy()

confirmed_forecast=prophet_c.predict(forecast_c)



model_scores.append(np.sqrt(mean_squared_error(data_india["Confirmed"],confirmed_forecast['yhat'].head(data_india.shape[0]))))

print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(data_india["Confirmed"],confirmed_forecast['yhat'].head(data_india.shape[0]))))
Prediction_prophet = confirmed_forecast['yhat'].head(data_india.shape[0]).tolist()

n = len(Prediction_SVM_Regression)

Prediction_prophet =Prediction_prophet[-n:]
print(prophet_c.plot(confirmed_forecast))
print(prophet_c.plot_components(confirmed_forecast))
print(Prediction_prophet)

print(Prediction_SVM_Regression)

print(Prediction_Polynomial_Regression)

print(Prediction_Linear_Regression)

print(Confirmed)
Prediction_Polynomial_Regression = ([int(i) for i in Prediction_Polynomial_Regression])

Prediction_SVM_Regression = ([int(i) for i in Prediction_SVM_Regression])

Prediction_prophet = ([int(i) for i in Prediction_prophet])

Confirmed = ([int(i) for i in Confirmed])
dict1 = {'Confirmed': Confirmed, 'Prediction_Polynomial_Regression': Prediction_Polynomial_Regression, 'Prediction_SVM_Regression':Prediction_SVM_Regression, 'Prediction_Prophet':Prediction_prophet, 'Prediction_Linear_Regression': Prediction_Linear_Regression}  
compare = pd.DataFrame(dict1)
compare