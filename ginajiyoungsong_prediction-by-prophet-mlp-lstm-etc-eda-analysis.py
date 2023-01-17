import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import date, timedelta



from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



import seaborn as sns

import scipy as sp
case = pd.read_csv('../input/coronavirusdataset/case.csv')

case
region = case.groupby(['province','city']).confirmed.sum()

region =pd.DataFrame(region)

region
count = region.groupby(['province']).sum() 

count = pd.DataFrame(count)

count
count.reset_index(inplace=True)

count.index=["Busan","Chungcheongbuk-do","Chungcheongnam-do","Daegu","Daejeon","Gangwon-do","Gwangju","Gyeonggi-do","Gyeongsangbuk-do","Gyeongsangnam-do",

            "Incheon","Jeju-do","Jeollabuk-do","Jeollanam-do","Sejong","Seoul","Ulsan"]

fig = px.pie(count, values=count.confirmed, names=count.index)

fig.show()
daily_report =pd.read_csv('../input/coronavirusdataset/time.csv')

daily_report.head(5)
col = ['confirmed','negative','test']

dataset = daily_report[col]

print(dataset.info())



import datetime

end = datetime.datetime.now() - datetime.timedelta(1)

date_index = pd.date_range('2020-01-20', end)

dataset.index = date_index



fig = px.area(dataset, x=dataset.index, y='test' )

fig.show()
dataset.tail()
import cufflinks as cf 

cf.go_offline(connected=True)

dataset.iplot( fill=True)
fatality_rate = pd.DataFrame({'Confirmed':[80814, 15113, 10075, 7979, 3146, 2876, 2745, 1762],

                              'Death':     [3177,   1016,  429,  67,   86,   61,    6,  41]}) 

fatality_rate = fatality_rate.rename(index={0:"China",1:"Italy", 2:"Iran", 3:"Korea", 4:"Spain", 5: "France", 6:" Germany" , 7:"USA" })

fatality_rate['fatality'] = fatality_rate['Death']  /fatality_rate['Confirmed'] *100

fatality_rate
plt.rcParams['figure.figsize']=15,7

ax= fatality_rate['Confirmed'].plot(kind='bar',  title = "International nCOV-19 Confirmed patients", color=['slateblue'])

for p in ax.patches:

  left, bottom, width, height = p.get_bbox().bounds

  ax.annotate("%.f"%(height), (left+width/2, height*1.01), ha='center')

    

plt.sca(ax)

plt.box(False) # 그래프 외곽선 지우기

plt.show()
plt.rcParams['figure.figsize']=18,7

ax = fatality_rate['fatality'].plot(kind='barh', stacked=True, title= "International nCOV-19 fatality_rate", rot=0, color=['red', 'skyblue','darkgreen', 'orange','lightgreen','slateblue','yellow', 'darkblue']) 

for p in ax.patches: 

  left, bottom, width, height = p.get_bbox().bounds 

  ax.annotate("%.2f%%"%(width), xy=(left+width*1.05, bottom+height/2), ha='center', va='center') 



plt.sca(ax)

plt.box(False) 

plt.show()

df_patient = pd.read_csv('../input/coronavirusdataset/patient.csv')

df_patient
df_patient.info() # most data have missing value...
# get daily confirmed count 

# 하루하루 확진자로 판명받는 데이터의 id 갯수를 세면 = 당일 확진자수가 나옵니다

daily_count = df_patient.groupby('confirmed_date').patient_id.count()

 

# get accumulated confirmed count

# 당일 확진자수를 모두 cumsum 합쳐서 누적확진자수를 구합니다.



accumulated_count = daily_count.cumsum()



plt.figure(figsize=(18,5))



color = 'tab:red'

ax = accumulated_count.plot(title='Korea nCOV19 Confirmed individuals', rot=90, color=color)

ax.set_ylabel('Accumulated_count', color=color)

plt.box(False)



# double-y axis graph = 하나의 그래프에 2개의 y축 사용

ax2 = ax.twinx()

color = 'tab:blue'

ax2 = daily_count.plot(kind='bar',color=color, alpha=0.1)

for p in ax2.patches:

  left, bottom, width, height = p.get_bbox().bounds

  ax2.annotate("%.f"%(height), (left+width/2, height*1.01), ha='center') # ha =horizontal align

ax2.set_ylabel('Confirmed_count', color=color)

ax2.tick_params(axis='y', color=color)



plt.box(False)

plt.show()
daily_count = pd.DataFrame(daily_count)

daily_count_data =daily_count.reset_index()

daily_count_data
# get accumulated confirmed count

accumulated_count = daily_count.cumsum()

accumulated_count.reset_index(inplace=True)

accumulated_count
fig = go.Figure()

fig.add_trace(

    go.Scatter(x=daily_count_data.confirmed_date, y=daily_count.patient_id,

        name='nCOV-19 in Korea'))

# Add figure title

fig.update_layout(title_text="daily Confirmed_patient of nCOV-19 in Korea")



# Set x-axis title

fig.update_xaxes(title_text="2020 01 ~03")
fig = go.Figure()

fig.add_trace(

    go.Scatter(x=accumulated_count.confirmed_date, y=accumulated_count.patient_id, 

               name='nCOV-19 in Korea'))



# Add figure title

fig.update_layout(title_text="Accumulated Confirmed_patient of nCOV-19 in Korea")



# Set x-axis title

fig.update_xaxes(title_text="2020 01 ~03")
from plotly.subplots import make_subplots



# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])





# Add traces

fig.add_trace(

    go.Scatter(x=daily_count_data.confirmed_date, y=daily_count.patient_id,

        name='Confirmed in Korea'), secondary_y=False,)



fig.add_trace(

    go.Scatter(x=accumulated_count.confirmed_date, y=accumulated_count.patient_id,

        name='Accumulated in Korea'), secondary_y=True,)



# Add figure title

fig.update_layout(title_text="Corona19 in Korea")



# Set x-axis title

fig.update_xaxes(title_text="2020 01 ~03")



# Set y-axes titles

fig.update_yaxes(title_text="<b>Confirmed</b> individual", secondary_y=False)

fig.update_yaxes(title_text="<b>Accumulated</b> individual", secondary_y=True)



fig.update_layout(

    legend=dict(x=0, y=1,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=14,

            color="black" ),

        bgcolor="white",

        bordercolor="Black",

        borderwidth=2 ))



fig.show()
df_prophet = accumulated_count.rename(columns={ 'confirmed_date': 'ds', 'patient_id': 'y' })



df_prophet
m = Prophet(

    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

    changepoint_range=0.95, # place potential changepoints in the first 98% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



m.fit(df_prophet)



future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)





forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
fig = plot_plotly(m, forecast)

py.iplot(fig) 



fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)

sns.distplot(m.params["delta"], kde=False, fit=sp.stats.laplace)

plt.box(False)

day_plus = pd.DataFrame(index=['2020-03-12', '2020-03-13','2020-03-14' ]) 

daily_count_= pd.concat([daily_count, day_plus],sort=True)

  

accumulated_count_ = daily_count_.cumsum()

accumulated_count_
plt.figure(figsize=(25,8))

color = 'tab:red'

forecast.yhat.plot(color=color)

plt.ylabel('Accumulated_Confirmed', color=color)

plt.xlabel('days')

plt.title('Prediction by Prophet', color=color, fontdict ={'fontsize':15})

plt.ylim=[0,14000,2000]

plt.box(False)



accumulated_count_['patient_id'].plot()





daily_count_['patient_id'].plot(kind='bar', color=['coral'] )

plt.text(x=19, y=200, s='169',color='red', horizontalalignment='center') ;plt.text(x=20, y=300, s='231',color='red' ,horizontalalignment='center') 

plt.text(x=21, y=200, s='143',color='red',horizontalalignment='center') ;plt.text(x=22, y=330, s='285',color='red',horizontalalignment='center') 

plt.text(x=23, y=560, s='505',color='red',horizontalalignment='center') ;plt.text(x=24, y=610, s='571',color='red',horizontalalignment='center') 

plt.text(x=25, y=900, s='813',color='red',horizontalalignment='center') ;plt.text(x=26, y=1100, s='1061',color='red',horizontalalignment='center')

plt.text(x=27, y=650, s='600',color='red',horizontalalignment='center') ;plt.text(x=28, y=600, s='516',color='red',horizontalalignment='center')

plt.text(x=29, y=500, s='438',color='red',horizontalalignment='center') ;plt.text(x=30, y=600, s='518',color='red',horizontalalignment='center')

plt.text(x=31, y=500, s='483',color='red',horizontalalignment='center') ;plt.text(x=32, y=400, s='367',color='red',horizontalalignment='center')

plt.text(x=33, y=300, s='248',color='red',horizontalalignment='center') ;plt.text(x=34, y=200, s='131',color='red',horizontalalignment='center')

plt.text(x=35, y=300, s='242',color='red',horizontalalignment='center') ;plt.text(x=36, y=200, s='114',color='red',horizontalalignment='center')



plt.annotate('',xy=(14,10), xytext=(14,700), arrowprops=dict(facecolor='black'))

plt.text(x=13, y=800, s='secodary infection by 31 patient',color='green', fontweight='bold')

plt.annotate('',xy=(19,1000), xytext=(19,3300), arrowprops=dict(facecolor='black'))

plt.text(x=19, y=3500, s='president\'s Declaration',color='blue',fontweight='bold')





plt.show()
daily_released_count = df_patient.groupby('released_date').patient_id.count()

daily_released_count = pd.DataFrame(daily_released_count)



daily_released_count = daily_released_count.drop(['2020-03-05','2020-03-07'])



daily_plus = pd.DataFrame([47,20,10,12,36,81,41,45,177],

                          index=['2020-03-05','2020-03-06','2020-03-07','2020-03-08','2020-03-09','2020-03-10', '2020-03-11','2020-03-12','2020-03-13'],

                          columns=['patient_id'])

daily_released_count =pd.concat( [daily_released_count ,daily_plus] )

daily_released_count
accumulated_released_count = daily_released_count.cumsum()

accumulated_released_count.tail()
plt.rcParams['figure.figsize']=10,6

color = 'tab:blue'

ax = daily_released_count.plot(kind='bar', title = 'Korea nCOV-19 Recovered individu1es', color=color, alpha=0.1, legend=False)

for p in ax.patches:

  left, bottom, width, height = p.get_bbox().bounds

  ax.annotate("%.f"%(height), (left+width/2, height*1.01), ha='center') 

ax.set_ylabel('Recovered_count', color=color)

ax.tick_params(axis='y', color=color)

ax.hlines(y = 10, xmin=0, xmax=28, colors=color, linestyles='dotted')

ax.hlines(y = 40, xmin=0, xmax=28, colors=color, linestyles='dotted')

plt.box(False)





color='tab:green'

ax2= accumulated_released_count.plot( color=color, legend=False)

ax2.set_ylabel('Accumulated_count', color=color)

ax2.tick_params(axis='y', color=color)

ax2.hlines(y = 50, xmin=0, xmax=28, colors='green', alpha=0.4, linestyles='dotted')

ax2.hlines(y = 100, xmin=0, xmax=28, colors='green', alpha=0.4,  linestyles='dotted')

ax2.hlines(y = 150, xmin=0, xmax=28, colors='green', alpha=0.4,  linestyles='dotted')

ax2.hlines(y = 200, xmin=0, xmax=28, colors='green', alpha=0.4,  linestyles='dotted')

data = df_patient[["patient_id","sex","birth_year","confirmed_date","released_date","deceased_date","state"]]

released_base_data = data.dropna(subset=['released_date'])

released_base_data = released_base_data.drop(['deceased_date'], axis=1)

released_base_data.info() # 03-04 data / not updated on 03-05 / 47 peoples data 
released_base_data.confirmed_date = pd.to_datetime(df_patient.confirmed_date)

released_base_data.released_date = pd.to_datetime(df_patient.released_date)



# recovered_period =치료기간(회복기간) 계산하기

released_base_data['recovered_period'] = released_base_data['released_date'].dt.date - released_base_data['confirmed_date'].dt.date

released_base_data['recovered_period'] = released_base_data['recovered_period'].dt.days.astype(int)

# Age

released_base_data["age"] = released_base_data["confirmed_date"].dt.year - released_base_data["birth_year"]

released_base_data = released_base_data.drop(['birth_year'], axis=1)

# Age_class

released_base_data["age_class"] = pd.cut(released_base_data["age"], np.arange(0, 100, 10), include_lowest=True, right=False)

released_base_data
boxplot = pd.DataFrame(released_base_data['recovered_period'].describe()).T

boxplot
red_square = dict(markerfacecolor='r', marker='s')

fig, ax1 = plt.subplots(figsize=(8,2))

ax1.set_title('Recovered time on Korea nCOV-19 patient')

ax1.set_ylabel('Period')

ax1.boxplot(boxplot,  vert=False, flierprops=red_square)
released_period= released_base_data.pivot_table( index="state", columns='age_class', values='recovered_period', aggfunc='count').fillna('0')

released_period = released_period.T

released_period.reset_index(inplace=True)

released_period
released_period.index=['10-19','20-29','30-39','40-49','50-59','60-69', '70-']



import plotly.express as px



fig = go.Figure(data=[

    go.Bar(name='All', x=released_period.index, y=released_period.released)])





fig.show()
released_by_age= released_base_data.pivot_table( index="sex", columns='age_class', values='released_date', aggfunc='count').fillna('0')

released_by_age = released_by_age.T

released_by_age.reset_index(inplace=True)

released_by_age
released_by_age.index=['10-19','20-29','30-39','40-49','50-59','60-69', '70-']





fig = go.Figure(data=[

    go.Bar(name='Female', x=released_by_age.index, y=released_by_age.female),

    go.Bar(name='Male', x=released_by_age.index, y=released_by_age.male)])



fig.update_layout(barmode='group')

fig.show()
Confirmed_sex = pd.DataFrame({'sex':[4661,2852], 'rate':[62.0,38.0]})

Confirmed_sex = Confirmed_sex.rename(index={0:"female", 1:"male"})

Confirmed_sex


plt.rcParams['figure.figsize']=12,8

group_explodes = (0.05, 0) #explode 1st slice

plt.pie(Confirmed_sex['sex'], explode=group_explodes,labels=["female","male"], colors=['lightcoral', 'lightskyblue'],autopct='%1.2f%%', shadow=True, startangle=90, textprops={'fontsize':14})

plt.axis('equal')

plt.title('Confirmed individual by sex')

plt.show()
Confirmed_age = pd.DataFrame({'Confirmed':[67,393,2213,789,1030,1416,926,454,222], 'rate':[0.9, 5.2, 29.5, 10.5, 13.7, 18.8, 12.4, 6.0, 3.0]})

Confirmed_age = Confirmed_age.rename(index={0:"0-9", 1:"10-19", 2:"20-29", 3:"30-39", 4:"40-49", 5:"50-59", 6:"60-69", 7:"70-79", 8:"80-"})

Confirmed_age
plt.rcParams['figure.figsize']=12,5

color = 'tab:red'

ax= Confirmed_age['Confirmed'].plot( title = " nCOV19 - Confirmed individule by Age", rot=0,color=color)

ax.set_ylabel('Confirmed_count',color=color)

ax.tick_params(axis='y',color=color)

ax.legend()



ax2 = ax.twinx()

color = 'tab:blue'

ax2 = Confirmed_age['rate'].plot(kind='bar', color=color, alpha=0.1)

ax2.set_ylabel(' %')



for p in ax2.patches: 

  left1, bottom1, width1, height1 = p.get_bbox().bounds 

  ax2.annotate("%.f%%"%(height1), xy=(left1+width1/2, bottom1+height1/2), ha='center', va='center')



plt.sca(ax)

plt.show()
df_patient.info() # infection_reason    154 non-null object
pd.DataFrame(df_patient["infection_reason"].value_counts())
reason_dict = {   

    "contact with the patient": "contact with patient", 

    "visit to Shincheonji Church" : "contact with patient", 

    "contact with patient in Japan":  "visit to other Country",

    "contact with patient in Singapore":  "visit to other Country",

    "visit to Daegu" : "contact with patient",  

    "visit to Shincheonji Church" : "contact with patient",



    "visit to Wuhan" : "visit to China",

    " visit to China" :  "visit to China",

    "visit to China": "visit to China",

    "residence in Wuhan" : "residence in Wuhan",



    "visit to Thailand":  "visit to other Country",

    "visit to Japan": "visit to other Country",

    "visit to Italy":  "visit to other Country",

    "visit to Vietnam":  "visit to other Country",



    "contact with patient in Japan":  "visit to other Country",

    "visit to Cheongdo Daenam Hospital" :"contact with patient",

    "pilgrimage to Israel":  "visit to other Country",

    "visit to ooo" :"contact with patient",

    "contact with patient in Daegu" :"contact with patient"



}

infection_reason =pd.DataFrame(df_patient["infection_reason"].replace(reason_dict).value_counts())

infection_reason



infection_reason.reset_index(inplace=True)
region = pd.DataFrame(df_patient["region"].value_counts())

region.reset_index(inplace=True)

region
infection_reason.index=['contact with patient','visit to other Country','visit to China','residence in Wuhan']

region.index = ['capital area','Gyeongsangbuk-do','Daegu','Daejeon','Gwangju','Gangwon-do','Jeju-do',

                'filtered at airport','Jeollabuk-do','Jeollanam-do', 'Ulsan','Chungcheongbuk-do','Busan','Chungcheongnam-do']



fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Bar(x=infection_reason.index, y=infection_reason.infection_reason,

                    marker=dict(color=infection_reason.infection_reason, coloraxis="coloraxis")),  1, 1)





fig.add_trace(go.Bar(x=region.index, y=region.region,

                    marker=dict(color=region.region, coloraxis="coloraxis")),  2, 1)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(height=1000, showlegend=False)

fig.show()
region_by3_9 = pd.DataFrame({'patient':[5571,1107,152,130,102, 96, 83], 'rate':[75.47, 15.0, 2.06, 1.76, 1.38, 1.3, 1.12]})

region_by3_9 = region_by3_9.rename(index={0:"Deagu", 1:"Gyeongbuk", 2:"Gyeonggi", 3:"Seoul", 4:"Chungnam", 5:"Busan", 6:"Gyeongnam"})

region_by3_9
plt.rcParams['figure.figsize']=15,4

color = 'tab:red'

ax= region_by3_9['patient'].plot( title = " nCOV19 - Confirmed individule by region", rot=0,color=color)

ax.set_ylabel('Confirmed_count',color=color)

ax.tick_params(axis='y',color=color)

ax.legend()



ax2 = ax.twinx()

color = 'tab:blue'

ax2 = region_by3_9['rate'].plot(kind='bar', color=color, alpha=0.1)

ax2.set_ylabel(' %')



for p in ax2.patches: 

  left1, bottom1, width1, height1 = p.get_bbox().bounds 

  ax2.annotate("%.f%%"%(height1), xy=(left1+width1/2, bottom1+height1/2), ha='center', va='center')



plt.sca(ax)

plt.show()
region_by3_9.reset_index(inplace=True)

region_by3_9.index=["Deagu","Gyeongbuk","Gyeonggi","Seoul","Busan","Chungnam","Gyeongnam"]

fig = px.pie(region_by3_9, values='rate', names='index')

fig.show()
count.reset_index(inplace=True)

count.index=["Busan","Chungcheongbuk-do","Chungcheongnam-do","Daegu","Daejeon","Gangwon-do","Gwangju","Gyeonggi-do","Gyeongsangbuk-do","Gyeongsangnam-do",

            "Incheon","Jeju-do","Jeollabuk-do","Jeollanam-do","Sejong","Seoul","Ulsan"]

fig = px.pie(count, values='confirmed', names='index')

fig.show()
sex = pd.DataFrame(df_patient["sex"].value_counts())

sex = pd.DataFrame(df_patient["sex"].replace({'feamle':'female'}).value_counts())

sex
sex.reset_index(inplace=True)

sex.index=['female','male']

import plotly.express as px

fig = px.pie(sex, values='sex', names='index')

fig.show()
plt.rcParams['figure.figsize']=12,8

group_explodes = (0.05, 0) #explode 1st slice

plt.pie(Confirmed_sex['sex'], explode=group_explodes,labels=["female","male"], colors=['lightcoral', 'lightskyblue'],autopct='%1.2f%%', shadow=True, startangle=90, textprops={'fontsize':14})

plt.axis('equal')

plt.title('Confirmed individual by sex')

plt.show()
data = df_patient[["patient_id","sex","birth_year","confirmed_date","deceased_date","state"]]

death_data = data.dropna(subset=['deceased_date'])

death_data
death_data.confirmed_date = pd.to_datetime(df_patient.confirmed_date)

death_data.deceased_date = pd.to_datetime(df_patient.deceased_date)



# deceased_period 사망까지 소요된 시간 계산

death_data['deceased_period'] = death_data['deceased_date'].dt.date - death_data['confirmed_date'].dt.date

# Age

death_data["age"] = death_data["confirmed_date"].dt.year - death_data["birth_year"]

death_data = death_data.drop(['birth_year'], axis=1)

# Age_class

death_data["age_class"] = pd.cut(death_data["age"], np.arange(0, 100, 10), include_lowest=True, right=False)

death_data
mean = death_data['deceased_period'].dt.days.mean()

print('평균 사망까지 걸린 시간 %.2f%%'%(mean),'days')
death_data = death_data.dropna(subset=['age_class'])

death_data
death_data_by_age = death_data.pivot_table( index="state", columns='age_class', values='deceased_date', aggfunc='count').fillna('0')

death_data_by_age = death_data_by_age.T

death_data_by_age.reset_index(inplace=True)

death_data_by_age
death_data_by_age.index=['30-39','40-49','50-59','60-69', '70-79','80-89']



import plotly.express as px



fig = go.Figure(data=[

    go.Bar(name='All', x=death_data_by_age.index, y=death_data_by_age.deceased)])





fig.show()

death_data= death_data.pivot_table( index="sex", columns='age_class', values='deceased_date', aggfunc='count').fillna('0')

death_data = death_data.T

death_data.reset_index(inplace=True)

death_data
death_data.index=['30-39','40-49','50-59','60-69', '70-79', '80-']



import plotly.express as px



fig = go.Figure(data=[

    go.Bar(name='Female', x=death_data.index, y=death_data.female),

    go.Bar(name='Male', x=death_data.index, y=death_data.male)])



fig.update_layout(barmode='group')

fig.show()
death_count = df_patient.groupby('deceased_date').patient_id.count()

death_count = pd.DataFrame(death_count)

death_accumulated = death_count.cumsum()
plt.rcParams['figure.figsize']=10,6



color='tab:blue'

ax= accumulated_released_count.plot(kind='bar' ,title = 'Korea nCOV-19 Recoved & Death individules',

                                    alpha=0.5, color=color, legend=False)

ax.set_ylabel('Recovered_count', color=color)

ax.tick_params(axis='y', color=color)

plt.box(False)

color = 'black'

ax2 = death_accumulated.plot(kind='bar', color=color, alpha=0.5, legend=False)

ax2.set_ylabel('Deceased_count', color=color)

ax2.tick_params(axis='y', color=color)

plt.box(False)



daily_count = df_patient.groupby('confirmed_date').patient_id.count()

daily_count = pd.DataFrame(daily_count)

data = daily_count.cumsum()

data 
dataset = data.iloc[14:]

dataset
# Future forcasting

days_in_future = 3

dates = pd.date_range('2020-2-18','2020-3-11')

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

# future_forcast #3일을 추가해서 > 미래 3일의 그래프를 그려본다.



days = np.array([i for i in range(len(dates))]).reshape(-1, 1) # index -> ndarray

data = np.array(dataset).reshape(-1, 1) # count->ndarray
kernel = ['linear', 'rbf']

c = [0.01, 0.1, 1, 10, 100]

gamma = [ 0.001, 0.01, 0.1, 10]

epsilon = [0.01, 0.1, 1, 10, 100]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

svm = SVR()

svm_search = GridSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1,verbose=1)



# 모델 fitting 전에 train, test 를 나눠서 해봤으나, 현재 데이터셋이 너무 범위가 작아 확진자증가를 반영하지 못해. 전체 데이터셋으로 fitting 함.

svm_search.fit(days, data)
svm_search.best_params_
svm_search.best_estimator_
svm_pred = svm_search.best_estimator_.predict(future_forcast)

svm_pred
import statsmodels.api as sm



lm= sm.OLS(data,days)

results =lm.fit()

results.summary()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error



linear_model = LinearRegression(fit_intercept=False, normalize=True)

linear_model.fit(days, data)

linear_pred = linear_model.predict(future_forcast)

linear_model.coef_
plt.plot(svm_pred, color='green', ls='-.', label = 'Prediction by SVM')

plt.plot(linear_pred, color='red', ls='--', label='Prediction by Linear Regression')

plt.plot(dataset, label='Accumulated real count')

plt.xlabel('days')

plt.vlines(x=22, ymin=0, ymax=10000, alpha=0.3, linestyles='--')

plt.text(x=24, y=5000, s='prediction',color='black', fontsize =20,horizontalalignment='center') 

plt.xticks(rotation=90, ha='left')

plt.legend()

plt.box(False)
dataset.columns = ['Confirmed']

len(dataset)
data = np.array(dataset).reshape(-1, 1) # count->ndarray
train_data = dataset[:len(dataset)-3]

test_data = dataset[len(dataset)-3:]
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#시계열 자료 {  Yt } 의 상관 함수는 acf, pacf, iacf 가 있는데 이는 ARMA 모형 진단에 사용된다.

diff_1=train_data.diff(periods=1).iloc[1:] # diff() 차분함수 => 1차 차분함수

diff_1.plot()

plot_acf(diff_1)

plot_pacf(diff_1)
from statsmodels.tsa.statespace.sarimax import SARIMAX



arima_model = SARIMAX(train_data['Confirmed'].values, order = (1,2,1))

arima_result = arima_model.fit(trend='c', full_output=True, disp=True)

arima_result.summary()
pred_Arima = arima_result.forecast(steps=3)

pred_Arima
x = days[:20] # train test split

y = train_data.values

test = days[20:]
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[20, 10, 10, 3], max_iter=20000, random_state=42)

MLP = model.fit(x, y)
pred_MLP = model.predict(test)

pred_MLP
# Before creating LSTM model we should create a Time Series Generator object.



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)

scaled_test_data = scaler.transform(test_data)



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM



from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

n_input =2

n_features =1

                             

generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)



lstm_model = Sequential()

lstm_model.add(LSTM(19, activation='relu', input_shape = (n_input, n_features)))

lstm_model.add(Dense(10))

lstm_model.add(Dense(5))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator, epochs=20)
lstm_model.history.history.keys()
losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize = (12,4))

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.xticks(np.arange(0,21,1))

plt.plot(range(len(losses_lstm)), losses_lstm)
lstm_predictions_scaled = []



batch = scaled_train_data[-n_input:]

current_batch = batch.reshape((1, n_input, n_features))



for i in range(len(test_data)):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    lstm_predictions_scaled.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)





# As you know we scaled our data that’s why we have to inverse it to see true predictions.

lstm_predictions_scaled
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled) # 열로 넣어줘야 시각화 할수 있음

lstm_predictions
test_data['ARIMA'] = pred_Arima

test_data['MLP regression'] = pred_MLP

test_data['LSTM_Predictions'] = lstm_predictions

test_data
test_data.plot()
print('MAE of ARIMA Model ', mean_absolute_error(test_data['Confirmed'], test_data['ARIMA']))

print('MAE of MLP Model ',mean_absolute_error(test_data['Confirmed'], test_data['MLP regression']))

print('MAE of LSTM Model ',mean_absolute_error(test_data['Confirmed'], test_data['LSTM_Predictions']))

print('----------------------------------------------------')

print('MSE of ARIMA Model ', mean_squared_error(test_data['Confirmed'], test_data['ARIMA']))

print('MSE of MLP Model ',mean_squared_error(test_data['Confirmed'], test_data['MLP regression']))

print('MSE of LSTM Model ',mean_squared_error(test_data['Confirmed'], test_data['LSTM_Predictions']))
from sklearn.metrics import r2_score

print('Coefficient of determination : r2_score of ARIMA Model ', r2_score(test_data['Confirmed'], test_data['ARIMA']))

print('Coefficient of determination : r2_score of MLP Model ',r2_score(test_data['Confirmed'], test_data['MLP regression']))

print('Coefficient of determination : r2_score of LSTM Model ',r2_score(test_data['Confirmed'], test_data['LSTM_Predictions']))



# 결정계수가 - 음수로 나온다는건, 모델이 의미가 없다는걸로 해석되네요.. 

# 시계열분석인 ARIMA 만 설명력이 양수로 0.50 으로 확인됩니다.. (비계절성 ARIMA 모형입니다.)
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = np.array(dataset).reshape(-1, 1)

arima_model = SARIMAX(data, order = (1,2,1))

arima_result = arima_model.fit(trend='c', full_output=True, disp=True)

arima_result.summary()
pred_Arima = arima_result.forecast(steps=6) 

pred_Arima = pd.DataFrame(pred_Arima, columns=['predict'])

pred_Arima
forecast[['ds', 'yhat']].tail(7)