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



import seaborn as sns
import warnings 
# warnings.simplefilter("default")
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# For interactive plot in Kaggle notebook, I found a helpful guide here: https://www.kaggle.com/harisyammnv/interactive-eda-with-plotly-ipywidget
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%matplotlib inline

import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True,  theme='pearl')
import folium
import altair as alt
import missingno as msg
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from ipywidgets import interact, interactive, fixed
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")



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
from collections import OrderedDict 

countries_dict = OrderedDict() 
countries_dict["Afghanistan"]=[""];
countries_dict["Italy"]=[""]
countries_dict["India"]=[""]
countries_dict["Germany"]=[""]
countries_dict["Spain"]=[""]
countries_dict["Taiwan*"]=[""]
countries_dict["Japan"]=[""]
countries_dict["Spain"]=[""]
countries_dict["Germany"]=[""]
countries_dict["Singapore"]=[""]
countries_dict["Korea, South"]=[""]
countries_dict["United Kingdom"]=[""]
countries_dict["US"]=["","Louisiana","New York","California","Minnesota"]
from copy import deepcopy
n=50
# countries_dict["US"]=[""]

N_places=sum([ len(value) for key, value in countries_dict.items()])
False_mask_0=[False]*(N_places*2+1)


labels=time_span[-n-30:-30]
x=time_span0[-n-30:-30]

data=[];   manu_list=[];

data.append(go.Bar(x=x,y=[0]*len(x),name='cases'))

False_mask=deepcopy(False_mask_0)
False_mask[0]=True         
manu_list.append(dict(label = "Select",
                 method = 'update',      
                 args = [{'visible': False_mask},{'title': "Select country/state"}]))




n_place=-1



for country in countries_dict:
    for state in countries_dict[country]:
            sp=" "
            if state!="": sp=', '
            n_place+=1   
            data_i=train[(train['Province_State']==state)&(train['Country_Region']==country)]\
                   .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]
            
            
            if country in ["United Kingdom","Canada"]:
                data_i=train[train['Country_Region']==country].groupby("Date").sum().reset_index()\
                       .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]
        
            if country=="US" and state=="":
                data_i=train[train['Country_Region']==country].groupby("Date").sum().reset_index()\
                          .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]    

            cases=country+state+" Cases_daily";
            deaths=country+state+" deaths_daily";

            data_i[cases]=data_i["ConfirmedCases"].diff()
            data_i[deaths]=data_i["Fatalities"].diff()
            
            trace1=go.Bar(x=x,y=data_i[cases][-n:],name='cases')
            trace2=go.Bar(x=x,y=data_i[deaths][-n:],name='deaths')
            
            data+=[trace1,trace2]
             
            False_mask=deepcopy(False_mask_0)
            False_mask[(2*n_place+1):(2*n_place+2+1)]=[True,True]
            
            manu_list.append(dict(label = country+sp+state,
                 method = 'update',      
                 args = [{'visible': False_mask},{'title': country+sp+state}]))

            

updatemenus = [
    dict(active=0,
        buttons=manu_list,
         direction = 'down'
#          ,
#          showactive = True, 
    )
]

layout = dict(title = 'Select Countries and states',
              yaxis=dict(title='daily count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
              xaxis= dict(title= 'Date',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
              margin=go.Margin(l=50,r=20),paper_bgcolor='rgb(105,105,105)',
              plot_bgcolor='RGB(228, 235, 234)',barmode='group',font= {'color': 'RGB(179, 217, 82)'},updatemenus=updatemenus,showlegend=True)


fig = dict(data=data, layout=layout)

py.iplot(fig, filename='relayout_option_dropdown')





countries=dict()
for cnt in train['Country_Region'].unique():
    countries[cnt]=train.loc[train['Country_Region']==cnt,'Province_State'].unique()

countries_test=dict()
for cnt in test['Country_Region'].unique():
    countries_test[cnt]=test.loc[test['Country_Region']==cnt,'Province_State'].unique()
res=[]
for country in countries:
    for state in countries[country]:
        if country!="China":
            country_state_filter_train=(train['Province_State']==state)&(train['Country_Region']==country)
            sliced_data=train.loc[country_state_filter_train,:]
            history=sliced_data.loc[sliced_data['ConfirmedCases']>0,'ConfirmedCases'].to_list() 
            res.append(num_of_days_train-len(history))
aa=plt.figure()        
aa=plt.hist(res,color="blue",bins=10 ,range=(0,80))
aa=plt.title("first Confirmed Case histogram: # of countries/provinces(except China) .VS. days from Wuhan Lockdown(1/22/2020)")


res=[]
for country in countries:
    for state in countries[country]:
#         country_state_filter_test=(test['Province_State']==state)&(test['Country_Region']==country)
        if country!="China":
            country_state_filter_train=(train['Province_State']==state)&(train['Country_Region']==country)
            sliced_data=train.loc[country_state_filter_train,:]
            history=sliced_data.loc[sliced_data['Fatalities']>0,'Fatalities'].to_list() 
            res.append(num_of_days_train-len(history))
aa=plt.figure()          
aa=plt.hist(res,color="red",bins=10 ,range=(0,80))
aa=plt.title("first death histogram: # of countries/provinces(except China) .VS. days from Wuhan Lockdown(1/22/2020)")


def daily_plot(Country,Province_State,n1):
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']*2
    labels=time_span[-n1-30:-30]*2
    data=train[(train['Province_State']==Province_State)&(train['Country_Region']==Country)]\
       .sort_values(by="Date").loc[:,["day",'ConfirmedCases','Fatalities']]
    
    data["Cases_daily"]=data["ConfirmedCases"].diff()
    data["deaths_daily"]=data["Fatalities"].diff()
    
    cases= data["Cases_daily"].tail(n1).astype(int) 
    deaths = data["deaths_daily"].tail(n1).astype(int)

    x = np.arange(2*len(cases))  # the label locations
    width = 0.7  # the width of the bars



    fig = plt.figure(figsize=(36,11));fig.tight_layout(pad=3.0)
    fig.set_figheight(6)
    fig.set_figwidth(10)

    ax=fig.add_subplot()
    rects1 = ax.bar(x[:len(cases)] - width/2, cases, width, label='Daily cases')

    ax2 = ax.twinx(); color="red"
    rects2 = ax2.bar(x[len(cases):] - width/2, deaths, width, label='Daily deaths',color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Cases')
    ax2.set_ylabel('deaths')
    ax.set_title('{} , {}'.format(Country, Province_State))
    ax.set_xticks(x-0.5*width)
    ax.set_ylim(0, max(cases)*1.2)
    ax.set_xticklabels(labels)
    ax2.legend((rects1,rects2), ("ConfirmedCases","Fatalities"),loc="upper left")
    ax2.set_ylim(0, max(deaths)*1.2)
    # ax.set_xticks(x+len(cases))
    # ax.set_xticklabels(labels)


    def autolabel_1(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    def autolabel_2(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax2.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')



    autolabel_1(rects1)
    autolabel_2(rects2)

    fig.tight_layout()

    plt.show()


Country='US';     Province_State="";
daily_plot(Country,Province_State,50)
Country='US';     Province_State="New York";
daily_plot(Country,Province_State,50)
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

    sumb=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
    output=pd.DataFrame()
    output['ForecastId']=test['ForecastId'].astype(int)
    output['ConfirmedCases']=test[test['ConfirmedCases'].notnull()]['ConfirmedCases'].astype(int)
    output['Fatalities']=test[test['Fatalities'].notnull()]['Fatalities'].astype(int)
    output.to_csv('submission.csv',index=False)
    
    done=pd.read_csv('done.csv')
    done=done.append([{"A":country+" "+state}],ignore_index=True)
    done.to_csv('done.csv',index=False)
    return None

for country in countries_dict:
#     if country in ["India","Italy",]
    for state in countries_dict[country]:
        predict_country_state(country,state)
        