import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm
train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
world=train.groupby('Date').sum()
fig = plt.figure(figsize=(17,6))

fig.suptitle('COVID-19 Cases in the world',size=15)

ConfirmedCases, = plt.plot(world.index,world['ConfirmedCases'], 'go-', label='ConfirmedCases')

Fatalities, = plt.plot(world.index,world['Fatalities'], 'ro-', label='Fatalities')

plt.legend(handles=[ConfirmedCases, Fatalities])

plt.xticks(np.arange(0,70,step=2),rotation=45)

plt.xlabel('Date',size=15)

plt.ylabel('No. of peoples',size=15)

plt.show()
day=[]

for i in range(1,len(world)+1):

    day.append(i)
world['Day']=day
y=[world['ConfirmedCases'][0]]

for i in range(len(world)-1):

    k=world.iloc[i+1,1]-world.iloc[i,1]

    y.append(k)
world['CC_on_that_day']=y

world
poisson_regression=sm.GLM(world['CC_on_that_day']/1000,world['Day'], family=sm.families.Poisson()).fit()
poisson_regression.summary()
total_days=[]

for i in range(100):

    total_days.append(i+1)
prediction_world=poisson_regression.get_prediction(total_days)
prediction_world.summary_frame().head()
pred_cases=prediction_world.summary_frame()['mean']
Dates=pd.merge(train,test,how='outer',on='Date')
pred_cases_cf=[pred_cases[0]]

for i in range(len(pred_cases)-1):

    pred_cases_cf.append(pred_cases[i]+pred_cases[i+1])
fig = plt.figure(figsize=(15,6))

fig.suptitle('Predicted versus actual counts of COVID-19 Cases in the World',size=15)

predicted, = plt.plot(Dates['Date'].unique(),pred_cases_cf, 'go-', label='Predicted counts')

actual, = plt.plot(world.index,np.array(world['CC_on_that_day'])/1000, 'ro-', label='Actual counts')

plt.legend(handles=[predicted, actual])

plt.xticks(np.arange(1,110,step=7),rotation=45)

plt.xlabel('Date',size=15)

plt.ylabel('No. of peoples',size=15)

plt.text(Dates['Date'][len(Dates)-1],pred_cases_cf[len(pred_cases_cf)-1],str(round(sum(pred_cases)*1000)),ha='right')

plt.show()
fatalities=[world['Fatalities'][0]]

for i in range(len(world)-1):

    fatalities.append(world['Fatalities'][i+1]-world['Fatalities'][i])
world['Fatalities_on_that_day']=fatalities
f=sm.GLM(world['Fatalities_on_that_day'],world['Day'], family=sm.families.Poisson()).fit()
d=f.get_prediction(total_days)
d.summary_frame()['mean'].head()
fig = plt.figure(figsize=(15,6))

fig.suptitle('Predicted versus actual fatalities counts of COVID-19 Cases in the World',size=15)

predicted, = plt.plot(Dates['Date'].unique(),d.summary_frame()['mean'], 'go-', label='Predicted counts')

actual, = plt.plot(world.index,world['Fatalities_on_that_day'], 'ro-', label='Actual counts')

plt.legend(handles=[predicted, actual])

plt.xticks(np.arange(1,110,step=7),rotation=45)

plt.xlabel('Date',size=15)

plt.ylabel('No. of peoples',size=15)

plt.show()
train['Province_State'].replace({'Georgia':'GeorgiaUS'},inplace=True)
for i in range(len(train)):

    if train['Province_State'].isnull()[i]==True:

        train.iloc[i,1]=train.iloc[i,2]
ps=train['Province_State'].unique()
on_that_day=[]

for j in range(len(ps)):

    on_that_day.append(train[train['Province_State']==ps[j]].iloc[0,4])

    for i in range(len(train[train['Province_State']==ps[j]])-1):

        on_that_day.append(train[train['Province_State']==ps[j]].iloc[i+1,4]-train[train['Province_State']==ps[j]].iloc[i,4])
train['CC_on_that_day']=on_that_day
Day=[]

for i in range(len(ps)):

    for j in range(len(train[train['Province_State']==ps[i]])):

        Day.append(j+1)
train['Day']=Day
pd.concat([train.head(),train.tail()])
prf=[]

for i in range(len(ps)):

    prf.append(sm.GLM(train[train['Province_State']==ps[i]]['CC_on_that_day'],train[train['Province_State']==ps[i]]['Day'], family=sm.families.Poisson()).fit())
for i in range(len(test)):

    if test['Province_State'].isnull()[i]==True:

        test.iloc[i,1]=test.iloc[i,2]
day=[]

for j in range(len(ps)):

    for i in range(58,58+len(test[test['Province_State']==ps[j]])):

        day.append(i)
test['day']=day
pred=[]

for i in range(len(ps)):

    pred.append(prf[i].get_prediction(test[test['Province_State']==ps[i]]['day']))
prediction=[]

for i in range(len(pred)):

    for j in range(len(pred[i].summary_frame()['mean'])):

        prediction.append(round(pred[i].summary_frame()['mean'].iloc[j]))
test['Predicted_Cases_on_that_day']=prediction
pd.concat([test.head(10),test.tail()])
a=pd.pivot_table(test,index='Province_State',values='Predicted_Cases_on_that_day',aggfunc=np.cumsum)
df=pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':a['Predicted_Cases_on_that_day'],'Fatalities':np.repeat(0,len(test))})
df.to_csv('submission.csv',index=False)