# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline  

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from time import time

import math

import warnings 
# warnings.simplefilter("default")
warnings.filterwarnings('ignore')




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import arrow
# import datetime

# today = arrow.utcnow().to('Asia/Calcutta').format('YYYY-MM-DD')
# display(time_span0[0])
# # datetime.timestamp(now)
# # display(datetime.datetime.timestamp(datetime.datetime.today()))
# display(pd.to_datetime(pd.Series(datetime.datetime.today()))[0])
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

train['Province_State'].fillna('',inplace=True)
train['Date'] = pd.to_datetime(train['Date'])
train['day'] = train.Date.dt.dayofyear
train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

test['Province_State'].fillna('', inplace=True)
test['Date'] = pd.to_datetime(test['Date'])
test['day'] = test.Date.dt.dayofyear
test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

train.sort_values(by="Date",inplace=True)
test.sort_values(by="Date",inplace=True)

day_min = train['day'].min()
train['day'] -= day_min
test['day'] -= day_min

test['ConfirmedCases']=np.nan
test['Fatalities']=np.nan

train["ForecastId"]=np.nan
test["Id"]=np.nan

min_date_train=train['Date'].min()
min_date_test=test['Date'].min() 
max_date_train=train['Date'].max()
max_date_test=test['Date'].max()

num_of_days_train=(max_date_train-min_date_train)/np.timedelta64(1, 'D')+1
num_of_days=int((max_date_test-min_date_train)/np.timedelta64(1, 'D'))+1

#two formats for the x-axis, for plotting purpose
time_span0=pd.date_range(min_date_train, max_date_test)
time_span=[str(s.month)+"/"+str(s.day) for s in time_span0]

forcast_days=int((max_date_test-max_date_train)/np.timedelta64(1, 'D'))
row=7;col=2; 

fig = plt.figure(figsize=(36,11));fig.tight_layout(pad=3.0)
fig.set_figheight(30)
fig.set_figwidth(20)

# fig,ax=plt.subplots(row,col,figsize=(36,11))
#fig.tight_layout(pad=3.0)
n=0

def plot_Country_State(Country,Province_State,n,row,col,fig):
    data=train[(train['Province_State']==Province_State)&(train['Country_Region']==Country)]\
       .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]
    
    if Country in ["United Kingdom","Canada"]:
        data=train[train['Country_Region']==Country].groupby("Date").sum().reset_index()\
       .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]
        
    if Country=="US" and Province_State=="":
        data=train[train['Country_Region']==Country].groupby("Date").sum().reset_index()\
       .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]    
#     pos=n//col, n%col
#     if row==1 or col==1:pos=n
        
    ax = fig.add_subplot(row,col,n)
    color='tab:blue'
    line1,=ax.plot(data["day"],data["ConfirmedCases"],label="ConfirmedCases")
    ax.set_title(Country+", "+Province_State); 
    ax.tick_params(axis='y', labelcolor=color)
#     ax[pos].legend()

    ax2 = ax.twinx();color='tab:red'
    line2,=ax2.plot(data["day"],data["Fatalities"],label='Fatalities',color=color);  
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend((line1,line2), ("ConfirmedCases","Fatalities"),loc="upper left")

Country='US';     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='US';     Province_State="New York";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='US';     Province_State="Louisiana";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='US';     Province_State="California";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='Italy';     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='Spain';     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="Singapore";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="Korea, South";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="United Kingdom";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="Germany";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="Taiwan*";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country="Canada";     Province_State="";  n+=1;  plot_Country_State(Country,Province_State,n,row,col,fig)

Country='India';    Province_State="";  n+=1; plot_Country_State(Country,Province_State,n,row,col,fig)

Country='Japan';    Province_State="";  n+=1; plot_Country_State(Country,Province_State,n,row,col,fig)

# from datetime import datetime
# datetime_str=pd.date_range(min_date_train, max_date_test)
# ss=[str(s.month)+"/"+str(s.day) for s in datetime_str]
# len(ss)

# display(min_date_test)
# display(max_date_train)
# display(max_date_test)
# display(int((max_date_test-min_date_train)/np.timedelta64(1, 'D')))+1
# display(test.Date.dt.dayofyear.max()-train.Date.dt.dayofyear.min()+1)
countries=dict()
for cnt in train['Country_Region'].unique():
    countries[cnt]=train.loc[train['Country_Region']==cnt,'Province_State'].unique()

countries_test=dict()
for cnt in test['Country_Region'].unique():
    countries_test[cnt]=test.loc[test['Country_Region']==cnt,'Province_State'].unique()
res=[]
for country in countries:
    for state in countries[country]:
        country_state_filter_train=(train['Province_State']==state)&(train['Country_Region']==country)
        sliced_data=train.loc[country_state_filter_train,:]
        history=sliced_data.loc[sliced_data['ConfirmedCases']>0,'ConfirmedCases'].to_list() 
        res.append(num_of_days_train-len(history))
plt.figure()        
plt.hist(res,color="blue",bins=10 ,range=(0,80))
plt.title("first Confirmed Case histogram: # of countries/provinces .VS. days from Wuhan Lockdown(1/22/2020)")


res=[]
for country in countries:
    for state in countries[country]:
#         country_state_filter_test=(test['Province_State']==state)&(test['Country_Region']==country)
        country_state_filter_train=(train['Province_State']==state)&(train['Country_Region']==country)
        sliced_data=train.loc[country_state_filter_train,:]
        history=sliced_data.loc[sliced_data['Fatalities']>0,'Fatalities'].to_list() 
        res.append(num_of_days_train-len(history))
plt.figure()          
plt.hist(res,color="red",bins=10 ,range=(0,80))
plt.title("first death histogram: # of countries/provinces .VS. days from Wuhan Lockdown(1/22/2020)")


display(train.head())
display(test.head())
set(train.columns).difference(test.columns)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(np.log(np.abs(y_pred[i] + 1)) - np.log(np.abs(y[i] + 1))) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# evaluate an ARIMA model for a given order (p,d,q)   
def evaluate_arima_model(X,forecast_days, arima_order):
    # prepare training dataset
    X=[x for x in X]
    train_size = int(len(X) * 0.9)
    train, test1 = X[0:train_size], X[train_size:]
    # make predictions
    history=train
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    predictions = list()
    predictions =model_fit.forecast(steps=len(test1))[0]

    model = ARIMA(X, order=arima_order)
    model_fit = model.fit(disp=0)
    if np.isnan(model_fit.forecast(steps=forecast_days)[0]).sum()>0:
        return float('inf')
#     print("herehere3333333333333333")
#     print("error=",rmsle(test1, predictions))
    error = rmsle(test1, predictions) 
    
    return error
    
def evaluate_models(dataset,forcast_days, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), (0,0,0)
  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset,forcast_days, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue 
        

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    try: 
        model=ARIMA(dataset, order=best_cfg)
        model_fit=model.fit(disp=0)
        new_data=model_fit.forecast(steps=forcast_days)[0]
    except:
        new_data=np.array([np.float("inf")])
    
    return best_cfg, best_score, new_data
done=pd.DataFrame({"A":[1]})
done.to_csv('done.csv',index=False)
def predict_country_state(country,state):
    country_state_filter_train=(train['Province_State']==state)&(train['Country_Region']==country)
    country_state_filter_test=(test['Province_State']==state)&(test['Country_Region']==country)

    sliced_data=train.loc[country_state_filter_train,:]

    Targets=['ConfirmedCases', "Fatalities"]; Subs=["_Cases","_Deaths"]; Preds=dict(); history_Preds=dict()
    for i,target in enumerate(Targets):
        history=sliced_data.loc[sliced_data[target]>0,target].to_list()  
    #     display(history[:5])
        start_time=time()
        best_cfg,best_score,pred=evaluate_models(history,forcast_days,range(10),range(7),range(7)) 
        if (pred!=np.float("inf")).all():
            Preds["Pred"+Subs[i]]=[round(p) if p>0 else 0 for p in pred] 
            history_Preds["Pred"+Subs[i]]=history+Preds["Pred"+Subs[i]]

            print("CPU time for "+target+ " costed: ",time()-start_time)
            print("Country=",country,", Province/State=", state)
            print("________________________")

            test.loc[country_state_filter_test&(test["Date"]<=max_date_train),target]\
                 =train.loc[country_state_filter_train&(train["Date"]>=min_date_test)&(train["Date"]<=max_date_train),target].values
            test.loc[country_state_filter_test&(test["Date"]>max_date_train),target]=Preds["Pred"+Subs[i]]
        else:
            return None

    fig=plt.figure()

    ss=history_Preds["Pred_Cases"]; hl=len(ss);
    ss_plot=np.zeros(num_of_days);ss_plot[-hl:]=ss

    ax=fig.add_subplot();  color='tab:blue'; 
    line1,=ax.plot(time_span0,ss_plot,label='ComfirmedCases',color=color )
    ax.plot(time_span0[-len(Preds["Pred_Cases"]):],Preds["Pred_Cases"],'*',color=color)
    ax.set_title(country+","+state)
    ax.set_ylabel("Comfirmed Cases")
    ax.set_xlabel("Date")
    ax.tick_params(axis='y', labelcolor=color)
    # ax.tick_params(axis='x',labelrotation=45, labelcolor=color)
    s=[time_span[i]  for i in range(len(time_span)) if  i%10==0];
    # plt.xticks(ticks=s,labels=s)

    ax.set_xticklabels([])


    scale=50
    ss=history_Preds["Pred_Deaths"]; hl=len(ss);
    ss_plot=np.zeros(num_of_days);ss_plot[-hl:]=np.array(ss)*scale

    ax2 = ax.twinx();       color='tab:red';   
    line2,=ax2.plot(time_span0,ss_plot,label='Fatalities',color=color); 
    ax2.plot(time_span0[-len(Preds["Pred_Deaths"]):],np.array(Preds["Pred_Deaths"])*scale,'+',color=color)

    ax2.set_title(country+","+state)
    ax2.set_ylabel("Fitalities (x 50)")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend((line1,line2), ("ConfirmedCases","Fatalities"),loc="upper left")
    s0=[time_span0[i]  for i in range(len(time_span)) if  i%16==15];
    s=[time_span[i]  for i in range(len(time_span)) if  i%16==15];
    # ax2.set_xticklabels(labels=s,minor=False)
    ax2.set_xticklabels([])
    aa=plt.xticks(ticks=s0, labels=s)
    aa=plt.axvline(x=pd.to_datetime(pd.Series(datetime.today()))[0], ymin=0, ymax = ss_plot.max(), linewidth=2, color='g')
    
    plt.show()

    sumb=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
    output=pd.DataFrame()
    output['ForecastId']=test['ForecastId'].astype(int)
    output['ConfirmedCases']=test[test['ConfirmedCases'].notnull()]['ConfirmedCases'].astype(int)
    output['Fatalities']=test[test['Fatalities'].notnull()]['Fatalities'].astype(int)
    output.to_csv('submission.csv',index=False)
    
    done=pd.read_csv('done.csv')
    done=done.append([{"A":country+" "+state}],ignore_index=True)
    done.to_csv('done.csv',index=False)
    return None

    # Preds
# for country in countries:
#     for state in countries[country]:
#         predict_country_state(country,state)
from collections import OrderedDict 
countries_dict = OrderedDict() 
countries_dict["Afghanistan"]=[""];
countries_dict["Italy"]=[""]
countries_dict["India"]=[""]
countries_dict["Germany"]=[""]
countries_dict["Spain"]=[""]
countries_dict["US"]=["Louisiana","New York","California"]


for country in countries_dict:
#     if country in ["India","Italy",]
    for state in countries_dict[country]:
        predict_country_state(country,state)
        
for country in countries:
    if (countries[country]!="").any():
        print(country,len(countries[country]))
countries["US"]
