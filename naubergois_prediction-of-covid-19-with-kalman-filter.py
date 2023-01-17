# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df_china=df[df['Country/Region']=='China']
df_china_grouped=df_china.groupby(['Date']).sum()
df_china_grouped
from math import sqrt,pi
import matplotlib.pyplot as plt
import numpy as np

X=df_china_grouped['Confirmed']-df_china_grouped['Deaths']-df_china_grouped['Recovered']
!pip install filterpy
import numpy as np
import filterpy
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from sklearn.metrics import r2_score
import plotly.express as px



def kalman_filter(X,noise):
    my_filter = KalmanFilter(dim_x=2, dim_z=1)
    my_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

    my_filter.F = np.array([[1.,1.],
                [0.,1.]])    # state transition matrix

    my_filter.H = np.array([[1.,0.]])    # Measurement function
    my_filter.P *= 1000.                 # covariance matrix
    my_filter.R = 5                      # state uncertainty
    my_filter.Q =  Q_discrete_white_noise(dim=2, dt=0.2, var=noise)
    
    preds=[]
    for x in X:
        my_filter.predict()
        my_filter.update(x)
        local,speed= my_filter.x
        preds.append((local+speed)[0])
        
    visual_data=pd.DataFrame()
    count=0
    for x in X:
        visual_data=visual_data.append({'x':count,'y':x,'class':'Real'},ignore_index=True)
        count+=1
    count=0
    for x in preds:
        visual_data=visual_data.append({'x':count,'y':x,'class':'Predicted'},ignore_index=True)
        count+=1
        
    score=r2_score(X,preds)
    
    return visual_data,score
visual_data,r2=kalman_filter(X,0.2)
print('R2 score for q=0.2 ',r2)
px.line(visual_data,x='x',y='y',color='class')

visual_data,r2=kalman_filter(X,2)
print('R2 score for q=0.2 ',r2)
px.line(visual_data,x='x',y='y',color='class')

visual_data,r2=kalman_filter(X,50)
print('R2 score for q=0.2 ',r2)
px.line(visual_data,x='x',y='y',color='class')
visual_data,r2=kalman_filter(X,100)
print('R2 score for q=0.2 ',r2)
px.line(visual_data,x='x',y='y',color='class')
countries=df['Country/Region'].unique()
scores=[]
for country in countries:
    df_country=df[df['Country/Region']==country]
    df_country_grouped=df_country.groupby(['Date']).sum()
    X=df_country_grouped['Confirmed']-df_country_grouped['Deaths']-df_country_grouped['Recovered']
    visual_data,r2=kalman_filter(X,50)
    scores.append((country,r2))
scores
def kalman_filter_predict(X,init,noise,days):
    
    X_=X[init:init+days]
    my_filter = KalmanFilter(dim_x=2, dim_z=1)
    my_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

    my_filter.F = np.array([[1.,1.],
                [0.,1.]])    # state transition matrix

    my_filter.H = np.array([[1.,0.]])    # Measurement function
    my_filter.P *= 1000.                 # covariance matrix
    my_filter.R = 5                      # state uncertainty
    my_filter.Q =  Q_discrete_white_noise(dim=2, dt=0.2, var=noise)
    
    preds=[]
    for x in X_:
        my_filter.predict()
        my_filter.update(x)
        local,speed= my_filter.x
        preds.append((local+speed)[0])

    pred_forecast=[]
    for i in range(days):
        my_filter.predict()
        #my_filter.update(x)
        local,speed= my_filter.x
        preds.append((local+speed)[0])
        pred_forecast.append((local+speed)[0])
    visual_data=pd.DataFrame()
    count=0
    
    for x in X[init:2*days+init]:
        visual_data=visual_data.append({'x':count,'y':x,'class':'Real'},ignore_index=True)
        count+=1
    count=0
    for x in preds:
        visual_data=visual_data.append({'x':count,'y':x,'class':'Predicted'},ignore_index=True)
        count+=1
        
    
    return visual_data,pred_forecast

df_brazil=df[df['Country/Region']=='Brazil']
df_brazil=df_brazil[df_brazil['Confirmed']>0]

df_brazil_grouped=df_brazil.groupby(['Date']).sum()

X=df_brazil_grouped['Confirmed']-df_brazil_grouped['Deaths']-df_brazil_grouped['Recovered']

visual_data,forecast=kalman_filter_predict(X,20,50,50)
px.line(visual_data,x='x',y='y',color='class')



df_brazil=df[df['Country/Region']=='Brazil']
df_brazil=df_brazil[df_brazil['Confirmed']>0]

df_brazil_grouped=df_brazil.groupby(['Date']).sum()

X=df_brazil_grouped['Confirmed']-df_brazil_grouped['Deaths']-df_brazil_grouped['Recovered']

visual_data,forecast=kalman_filter_predict(X,30,50,10)
px.line(visual_data,x='x',y='y',color='class')