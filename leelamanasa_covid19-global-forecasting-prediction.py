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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates
from fbprophet import Prophet
Covid19_G_Data = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
Covid19_G_Data.head()
data_Clear= Covid19_G_Data.drop(['Province_State','Id'],axis=1)
data_Clear
total_cases = data_Clear.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()
total_cases.style.background_gradient(cmap='Reds')
total_case=total_cases.reset_index().head(10)
total_case
f,ax = plt.subplots(figsize =(10,10))
sns.set_color_codes("muted")
chart=sns.barplot(x='Country_Region',y='ConfirmedCases',data=total_case,label="Total Cases")
for item in chart.get_xticklabels():
    item.set_rotation(90)
fig = go.Figure(data=go.Scatter(x=total_case['Country_Region'], y=total_case['ConfirmedCases'],
                                mode='lines+markers',
                                hovertemplate = "Cases: %{y}<br> %{x}<extra></extra>",
                                showlegend = False
                               ))
fig.show()
ConfirmedCases_By_Date=data_Clear.groupby(["Date"])['ConfirmedCases'].sum().to_frame()
ConfirmedCases_By_Date=ConfirmedCases_By_Date.reset_index()
ConfirmedCases_By_Date.columns=['ds','y']
m = Prophet(changepoint_prior_scale=0.1)
m.fit(ConfirmedCases_By_Date)
future = m.make_future_dataframe(periods=300,freq='H')
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
d=m.plot(forecast)
d1=m.plot_components(forecast)
Fatalities_By_Date_Data = Covid19_G_Data.drop(['Country_Region','ConfirmedCases','Id','Province_State'],axis=1)
Fatalities_By_Date_Data
Fatalities_By_Date=data_Clear.groupby(["Date"])['Fatalities'].sum().to_frame()
Fatalities_By_Date=ConfirmedCases_By_Date.reset_index()
Fatalities_By_Date_Data.columns=['ds','y']
m = Prophet(changepoint_prior_scale=0.01)
m.fit(Fatalities_By_Date)
future = m.make_future_dataframe(periods=300,freq='H')
forecast_Fatalities = m.predict(future)
forecast_Fatalities[['ds','yhat','yhat_lower','yhat_upper']].tail()
d=m.plot(forecast_Fatalities)
d=m.plot_components(forecast_Fatalities)
forcastupdate = forecast.rename(columns={"yhat": "ConfirmedCases"})
Output_data1=forcastupdate[['ConfirmedCases']]
Output_data1
forecast_FatalitiesUpdate = forecast_Fatalities.rename(columns={"yhat": "Fatalities"})
Output_data2 = forecast_FatalitiesUpdate[['Fatalities']]
Output_data2
res = pd.concat([Output_data1, Output_data2], axis=1, sort=False)
res.index = np.arange(1, len(res)+1)
res.index.name ='ForecastId'
final_res = res.reset_index()
final_res.head()
final_res.to_csv("submission.csv",index=False)