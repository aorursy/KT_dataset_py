import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
cases = pd.read_csv("../input/sarscov2-ch-ti/cases.csv")

cases.count()
cases
df_long=pd.melt(cases, id_vars=['date'], value_vars=['confirmed_cases', 'deaths'])

fig = px.line(df_long, x='date', y='value', color='variable', width=1000, height=600, title="Total SARS-CoV-2 confirmed cases", labels={ "x": "Date"})

fig.show()
rows = len(cases)

change = [cases['confirmed_cases'][i]-cases['confirmed_cases'][i-1] for i in range(1,rows)]

days=[cases.index[x+1] for x in range(rows-1)]

fig = px.bar(x=days, y=change, color=change, orientation='v', width=1000, height=600, title='New SARS-CoV-2 confirmed cases by day', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
rows = len(cases)

perc_change = [100.0*(cases['confirmed_cases'][i]-cases['confirmed_cases'][i-1])/cases['confirmed_cases'][i-1] for i in range(14,rows)]

days=[cases.index[x+14] for x in range(rows-14)]

fig = px.bar(x=days, y=perc_change, color=perc_change, orientation='v', width=1000, height=600, title='New SARS-CoV-2 confirmed cases by day in %', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
dp_cases = pd.read_csv("../input/sarscov2-ch-ti/diamond_princess.csv")

mr = dp_cases["deaths"] / dp_cases["confirmed_cases"]
'Based on the SARS-CoV-2 cases on the Diamond Princess ship, we estimate a mortality rate of: {} %'.format(round(mr[0]*100, 2))
cases_with_mr1 = cases['deaths'] / mr.to_numpy()

cases_with_mr2 = cases['deaths'] / (mr.to_numpy() * 2.0)
days=[cases.index[x] for x in range(rows)]

fig = go.Figure()

fig.add_trace(go.Scatter(y=cases['confirmed_cases'], x=days, name='Confirmed cases'))

fig.add_trace(go.Scatter(y=cases_with_mr1, x=days, name='Cases if MR: {} %'.format(round(mr[0]*100, 2))))

fig.add_trace(go.Scatter(y=cases_with_mr2, x=days, name='Cases if MR: {} %'.format(round(mr[0]*100*2, 2))))

fig.update_layout(width=1000,height=600, title='Potential SARS-CoV-2 cases in Ticino')

fig.show()
days_to_predict = 50
cases = pd.read_csv("../input/sarscov2-ch-ti/cases.csv")

cases = cases.set_index("date")

cases = cases.iloc[:, :-1]
X = np.asarray([x for x in range(len(change))]).reshape(-1,1)

y = np.asarray(change).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
size = len(cases)

X_pred = np.asarray([x for x in range(size,size+days_to_predict)]).reshape(-1,1)

y_pred = regressor.predict(X_pred)
y_list=[]

cumul_sum=0

for y in list(y_pred.flatten()):

    cumul_sum+=y

    y_list.append(cumul_sum)

    

total_y = list(y_train.flatten())

total_y.extend(y_list)



total_x = [x for x in range(len(total_y))]
x_orig = [x for x in range(len(cases))]

plt.figure(figsize=(15, 10))

plt.scatter(x_orig, cases,  color='gray')

plt.plot(total_x, total_y, color='red', linewidth=2)

plt.xlabel('Nth Day of Coronavirus in Ticino')

plt.ylabel('Cases')

plt.title('Predicted SARS-CoV-2 cases in Ticino')

plt.show()
estimated_peak_day = 40

total_y[estimated_peak_day-1]