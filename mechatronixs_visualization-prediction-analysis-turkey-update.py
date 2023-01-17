# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.shape
obs_data = data.drop(['SNo', 'Last Update','Province/State'], axis=1)
obs_data.head()
data_turkey = obs_data[obs_data['Country/Region'] =='Turkey']
data_turkey['Active Cases'] = data_turkey['Confirmed'] - data_turkey['Deaths']-data_turkey['Recovered']
data_turkey.tail(5)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], 
                         y=data_turkey['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Black', width=3)))

fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], 
                         y=data_turkey['Deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='Red', width=3)))

fig.add_trace(go.Scatter(x=data_turkey['ObservationDate'], 
                         y=data_turkey['Recovered'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='Green', width=3)))

fig.update_layout(
    title="Corona Virus(Covid-19) Cases in Turkey",
    xaxis_title="Date",
    yaxis_title="Values",
    font=dict(
        family="Times New Roman, Bold",
        size=18,
        color="#000000"
    )
)


fig.show()
plt.figure(figsize=(20,20))
g = sns.PairGrid(data_turkey, vars=['Confirmed', 'Deaths', 'Recovered','Active Cases'],palette="husl")
g.map(plt.scatter, alpha=0.8)
g.add_legend();
f,ax1 = plt.subplots(figsize =(15,15))
sns.lineplot(x="Confirmed", y="Deaths",
                  hue="Country/Region", data=data_turkey,style = "Country/Region", palette = "hot", dashes = False, 
            markers = ["o", "<"],  legend="brief")
plt.title("Confirmed-Deaths Numbers in Turkey", fontsize = 20)
plt.xlabel("Total Confirmed Numbers by Time",fontsize = 15 , color = 'green')
plt.ylabel("Total Deaths Numbers by Time",fontsize = 15 , color = 'red')
plt.show()
data_turkey.describe()
x = data_turkey.loc[:,'Confirmed'].values.reshape(-1,1)
y = data_turkey.loc[:,'Deaths'].values.reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Confirmed')
plt.ylabel('Death')
plt.show()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(x)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(x, predicted, color='red', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Confirmed')
plt.ylabel('Death')
plt.title('Confirmed-Death Regression Prediction')
plt.show()

df = pd.DataFrame({'Actual': y.flatten(), 'Predicted': predicted.flatten()})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Comparison Actual and Prediction Numbers')
plt.show()
x = data_turkey.loc[:,'Confirmed'].values.reshape(-1,1)
z = data_turkey.loc[:,'Recovered'].values.reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=z)
plt.xlabel('Confirmed')
plt.ylabel('Recovered')
plt.show()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,z)
# Fit
reg.fit(x,z)
# Predict
predicted_rec = reg.predict(x)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(x, predicted_rec, color='red', linewidth=3)
plt.scatter(x=x,y=z)
plt.xlabel('Confirmed')
plt.ylabel('Recovered')
plt.title('Confirmed- Recovered Linear Regression Prediction')
plt.show()
