import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

from plotly.offline import init_notebook_mode, plot

from scipy.interpolate import make_interp_spline, BSpline

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os



import warnings

warnings.filterwarnings('ignore')



import plotly.graph_objects as go

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import datetime

from datetime import date as datefun

from datetime import timedelta

import requests



data_raw=requests.get('https://api.covid19india.org/raw_data.json').json()  # read API

head=list(data_raw['raw_data'][0].keys())

head # Keys
from shutil import copyfile



copyfile(src = "../input/my-data/util_functions.py", dst = "../working/util_functions.py")



from util_functions import *


temp = pd.DataFrame([]) 

for i in range(0,len(data_raw['raw_data'])):

    data1=pd.DataFrame([data_raw['raw_data'][i].values()], columns=head)

    temp=temp.append(data1,ignore_index = True)



#------------  Remove No data rows ------------------------------------

temp1= list(temp.loc[0:len(data_raw['raw_data']),'currentstatus'])

valid_data=[i for i, item in enumerate(temp1) if item != '']

data_raw=temp[0:len(valid_data)]
data_raw 

for index, row in data_raw.iterrows():

    if row['detectedstate'] in 'Delhi':

            data_raw.at[index,'detecteddistrict'] = 'Delhi'



start_date='2020-03-01'; end_date='2020-04-13'

df_state = get_state_df_from_api(data_raw,start_date,end_date)



#df_district = get_district_df_from_api(data_raw,start_date,end_date)

#df_district
latest_date=df_state['Date'].max()

df_state_latest = df_state[df_state['Date'] == latest_date].sort_values(by='ConfirmedCases',ascending=False)

#df_state_= df_latest.groupby(['District']).sum().reset_index() # get sum of cases for each province

df_state_latest.style.background_gradient(cmap='Reds')
df_k= pd.read_csv('../input/covid19-in-india/covid_19_india.csv')



df_k['Date']  = pd.to_datetime(df_k['Date'],format="%d/%m/%y") # new clean date columnn

df_k_latest = df_k[(df_k['Date'] == end_date)].sort_values(by='Confirmed',ascending=False)

#df_temp=df[df['Sno']==latest_index]

#latest_date=df_temp['Date'].max()

#df_latest = df[df['Date'] == latest_date]

df_k
print("Total number of Cases According to Kaggle Data base as on ",end_date, "is = ",df_k_latest['Confirmed'].sum())

print("Total number of Cases According to Covid19India.org as on ",end_date, "is = ",df_state_latest['ConfirmedCases'].sum())
df_state=df_state.sort_values(by='Date',ascending=True)

#gb_state_time = df_state.groupby(['Date']).sum().reset_index()

gb_state_time = df_state.pivot_table(index=['Date'], 

            columns=['State'], values='ConfirmedCases').fillna(0)

gb_state_time['Date']=gb_state_time.index

gb_state_time.index.name = None

gb_state_time = gb_state_time.sort_values(by='Date',ascending=True)

gb_state_time['Date'] = pd.to_datetime(gb_state_time['Date'],format="%Y-%m-%d")

gb_state_time.index = pd.to_datetime(gb_state_time['Date'],format="%Y-%m-%d")

gb_state_time=gb_state_time.drop(['Date'],axis = 1) 

gb_state_time.tail(5)
highlight=["Maharashtra","Tamil Nadu","Delhi","Telangana","Rajasthan"]

First_n=15

Days=30

Threshold=50

Label_X="Days ( Referenced to Threshold)"

Label_Y="Number of Confirmed Cases (Log Scale)"

Title="Trend Comparison of Different State (confirmed)\n Top 15 states in terms of COVID-19 cases"

plot_trend_rowdf(gb_state_time,Threshold,Days,First_n,highlight,Label_X,Label_Y,Title)
S = np.arange(0,200)

def predict(t,tau,theta,maxvalue):

    z = maxvalue*(1-np.exp(-(t-theta)/tau))

   # S[len(t)] = 0

    for i in range(len(t)):

        if t[i]<theta:

            S[i]=0

        else :

            S[i]=1

        z[i]=z[i]*S[i]

    return z
from scipy.optimize import curve_fit

def predict_n_plot_trend_for_Country(df,threshold,Days,Label_X,Label_Y,Title,df_next_week):

# modified from the awesome work by Tarun Kumar at https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons and modified for use

# Added the Prediction part

    

    temp_I = df#.sort_values(df.columns[-1], ascending= True)

    temp_I_sum = temp_I.sum(axis=1)

    last_row=temp_I.tail(1)

    last_row1 = last_row.sort_values(by=last_row.last_valid_index(),ascending=False, axis=1)

    f = plt.figure(figsize=(10,12))

    ax = f.add_subplot(111)

    x = Days

    t1_I = temp_I_sum.to_numpy()

    t2_I = t1_I[t1_I>threshold][:x]

    date = np.arange(1,len(t2_I[:x])+1)

    date1 = np.arange(1,len(t2_I[:x])+8)

    xnew = np.linspace(1, date.max(), Days)

    xnew1 = np.linspace(1, date1.max(), Days)

                

    a=int(len(t2_I))

    spl = make_interp_spline(date, t2_I, k=1)  # type: BSpline

    power_smooth1 = np.log2(t2_I)

    power_smooth1=power_smooth1*power_smooth1

    c,cov=curve_fit(predict,date[0:a],power_smooth1[0:a])

    power_smooth_predict=2**(np.sqrt(predict(date1,c[0],c[1],c[2])))

    df_next_week['India']=power_smooth_predict[len(t2_I):len(t2_I)+7]

               

    marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=6, markerfacecolor='#ffffff')

    plt.plot(date,t2_I,"-.",label = 'All The Cases',**marker_style)

    

    marker_style = dict(linewidth=4, linestyle='--', marker='o',markersize=6, markerfacecolor='#000000')

    plt.plot(date1,power_smooth_predict,"-.",label = 'Predicted Cases',**marker_style)

   

    plt.tick_params(labelsize = 14)        

    plt.xticks(np.arange(0,Days,7),[ "Day "+str(i) for i in range(Days)][::7])     



    # Reference lines 

    x = np.arange(0,Days/3)

    y = 2**(x+np.log(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days/2)

    y = 2**(x/2+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/7+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/30+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/4+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "Red")

    plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)



    # plot Params

    plt.xlabel(Label_X,fontsize=17)

    plt.ylabel(Label_Y,fontsize=17)

    plt.title(Title,fontsize=22)

    plt.legend(loc = "upper left")

    plt.yscale("log")

    plt.grid(which="both")

    plt.show()
Days=30

Threshold=1000

Label_X="Days ( Referenced to Threshold)"

Label_Y="Number of Confirmed Cases (Log Scale)"

Title="Prediction of COVID-19 cases for Next Seven Days from 14/04/20"

df_next_week_1=pd.DataFrame()

df_next_week_1['Days']=['Today_Prediction','Day2_Prediction','Day3_Prediction','Day4_Prediction','Day5_Prediction','Day6_Prediction','Day7_Prediction']

predict_n_plot_trend_for_Country(gb_state_time,Threshold,Days,Label_X,Label_Y,Title,df_next_week_1)

df_next_week_1
from scipy.optimize import curve_fit

def predict_n_plot_trend_rowdf(df,threshold,Days,First_n,highlight,Label_X,Label_Y,Title,df_next_week):

# modified from the awesome by Tarun Kumar work at https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons and modified for use

## Addded the prediction part

    

    temp_I = df#.sort_values(df.columns[-1], ascending= True)

    temp_I_sum = temp_I.sum(axis=1)

   # print(temp_I_sum)

    last_row=temp_I.tail(1)

    last_row1 = last_row.sort_values(by=last_row.last_valid_index(),ascending=False, axis=1)



 #   threshold = 50

 #   Days=51

    f = plt.figure(figsize=(10,12))

    ax = f.add_subplot(111)

    x = Days

    t1_I = temp_I_sum.to_numpy()

    t2_I = t1_I[t1_I>threshold][:x]

    date = np.arange(0,len(t2_I[:x]))

    xnew = np.linspace(date.min(), date.max(), Days)

    spl = make_interp_spline(date, t2_I, k=1)  # type: BSpline

    power_smooth = spl(xnew)

    marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=6, markerfacecolor='#ffffff')

    plt.plot(xnew,power_smooth,"-.",label = 'All The Cases',**marker_style)

          

    for i,col in enumerate(last_row1.columns):

        if col not in  ['Date','date']:

            x = Days

            threshold = int(temp_I[col].max()/10);

            if threshold <50:

                threshold=50

          #  print(threshold)

            t1_I = temp_I[col].to_numpy()

            t2_I = t1_I[t1_I>threshold][:x]

            

            if t2_I.size>3 :

                date = np.arange(1,len(t2_I[:x])+1)

                date1 = np.arange(1,len(t2_I[:x])+8)

                xnew = np.linspace(1, date.max(), Days)

                xnew1 = np.linspace(1, date1.max(), Days)

                

                a=int(len(t2_I))

                

                spl = make_interp_spline(date, t2_I, k=1)  # type: BSpline

                

                power_smooth = spl(xnew)

                power_smooth1 = np.log2(t2_I)

                power_smooth1 = power_smooth1*power_smooth1

                c,cov=curve_fit(predict,date[0:a],power_smooth1[0:a])

                power_smooth_predict=2**(np.sqrt(predict(date1,c[0],c[1],c[2])))

                df_next_week[col]=(power_smooth_predict[len(t2_I):len(t2_I)+7])

                

                if col in highlight:

                    marker_style = dict(linewidth=4, linestyle='-',markersize=6, markerfacecolor='#000000')

                    plt.plot(date,t2_I,"-.",label = col,**marker_style)

                    

                    marker_style = dict(linewidth=4, linestyle='--', marker='o',markersize=6, markerfacecolor='#000000')

                    plt.plot(date1,power_smooth_predict,"-.",label = col,**marker_style)

                



    plt.tick_params(labelsize = 14)        

    plt.xticks(np.arange(0,Days,7),[ "Day "+str(i) for i in range(Days)][::7])     



    # Reference lines 

    x = np.arange(0,Days/3)

    y = 2**(x+np.log(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days/2)

    y = 2**(x/2+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/7+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/30+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/4+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "Red")

    plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)



    # plot Params

    plt.xlabel(Label_X,fontsize=17)

    plt.ylabel(Label_Y,fontsize=17)

    plt.title(Title,fontsize=22)

    plt.legend(loc = "upper left")

    plt.yscale("log")

    plt.grid(which="both")

    plt.show()
highlight=["Maharashtra","Tamil Nadu","Delhi","Telangana","Rajasthan"]

#highlight=["Maharashtra"]

First_n=15

Days=30

Threshold=75

Label_X="Days ( Referenced to Threshold)"

Label_Y="Number of Confirmed Cases (Log Scale)"

Title="TPrediction of Covid19 cases from 14/04/20 for different states"

df_next_week=pd.DataFrame()

df_next_week['Days']=['Today_Prediction','Day2_Prediction','Day3_Prediction','Day4_Prediction','Day5_Prediction','Day6_Prediction','Day7_Prediction']

predict_n_plot_trend_rowdf(gb_state_time,Threshold,Days,First_n,highlight,Label_X,Label_Y,Title,df_next_week)

df_next_week
print("prediction for next seven days based on prediction for states\n",df_next_week.sum(axis=1))