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

import pandas as pd

import numpy as np

import seaborn as sns



df_patient=pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

df_patient=df_patient.loc[:,['id','country','confirmed_date','state']]

df_patient = df_patient.dropna()

Korea=df_patient[df_patient['country']=='Korea']

Korea['confirmed_date']=pd.to_datetime(Korea['confirmed_date'],format="%Y-%m-%d")

Date=set(Korea['confirmed_date'])

Date=pd.DataFrame(Date,columns=['date'])

Date=sorted(Date['date'])



Date=pd.DataFrame(Date,columns=['date'])

number_patients=[]

number_date=0

for col in Date['date']:

    date=Korea[Korea['confirmed_date']== col]

    number_date=number_date+len(date)

    number_patients.append(number_date)



number_patients=pd.DataFrame(number_patients,columns=['accum_patients'])

number_patients=pd.concat([Date,number_patients], axis=1, sort=True)
sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,8))



plt.xticks(rotation=60)

ax = sns.lineplot(number_patients['date'],number_patients['accum_patients'],color='red')



from matplotlib import ticker#目盛間隔の調整

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
state=Korea['state']

len_isolated=0

len_released=0

len_deceased=0





isolated=Korea[Korea['state']=='isolated']

number_isolated=[]

for i in Date['date']:

    len_isolated=len_isolated+len(isolated[isolated['confirmed_date']==i])



number_isolated.append(len_isolated)    

number_isolated=pd.DataFrame(number_isolated,index=['number_isolated'],columns=['number'])





released=Korea[Korea['state']=='released']

number_released=[]

for i in Date['date']:

    len_released=len_released+len(released[released['confirmed_date']==i])

    

number_released.append(len_released)

number_released=pd.DataFrame(number_released,index=['number_released'],columns=['number'])





deceased=Korea[Korea['state']=='deceased']

number_deceased=[]

for i in Date['date']:

    len_deceased=len_deceased+len(deceased[deceased['confirmed_date']==i])

    

number_deceased.append(len_deceased)

number_deceased=pd.DataFrame(number_deceased,index=['number_deceased'],columns=['number'])



state=pd.concat([number_isolated,number_released,number_deceased])

state
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)



trace = go.Bar(

    x=state.index,

    y=state['number'],

    name="Category",

    marker=dict(color=['red','blue','green'])

)



layout = go.Layout(title='Bar', 

                   legend=dict(orientation='h'))



data=[trace]



fig = go.Figure(data, layout=layout)

iplot(fig)
accum_deceased=Korea[Korea['state']=='deceased']

number_accum_deceased=[]

len_accum_deceased=0



for i in Date['date']:

    len_accum_deceased=len_accum_deceased+len(accum_deceased[accum_deceased['confirmed_date']==i])

    number_accum_deceased.append(len_accum_deceased)



number_accum_deceased=pd.DataFrame(number_accum_deceased,columns=['number_accum_deceased'])



number_accum_deceased=pd.concat([Date,number_accum_deceased],axis=1,sort=True)

#number_accum_deceased.head()
sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,8))



plt.xticks(rotation=60)

ax = sns.lineplot(number_accum_deceased['date'],number_accum_deceased['number_accum_deceased'],color='red')



from matplotlib import ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
per_deceased=[]

for i in Date['date']:

    date_patients=number_patients[number_patients['date']==i]

    date_deceased=number_accum_deceased[number_accum_deceased['date']==i]

    rate=date_deceased['number_accum_deceased'].values/date_patients['accum_patients'].values

    rate=round(pd.DataFrame(rate)*100,1)

    

    per_deceased.append(rate)

    

per_deceased=pd.DataFrame(per_deceased,columns=['rate_deceased'])

per_deceased=pd.concat([Date,per_deceased],axis=1,sort=True)

#per_deceased.head()
sns.set(style='darkgrid')

fig = plt.figure(figsize=(12,8))



plt.xticks(rotation=60)

ax = sns.lineplot(per_deceased['date'],per_deceased['rate_deceased'],color='red')



plt.ylabel('per_deceased(%)')

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))