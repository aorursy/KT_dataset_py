# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

!pip install chart_studio


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

from wordcloud import WordCloud


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
timeage = pd.read_csv("/kaggle/input/coronavirusdataset/TimeAge.csv")
region = pd.read_csv("/kaggle/input/coronavirusdataset/Region.csv")
time = pd.read_csv("/kaggle/input/coronavirusdataset/Time.csv")
weather = pd.read_csv("/kaggle/input/coronavirusdataset/Weather.csv")
searchtrend = pd.read_csv("/kaggle/input/coronavirusdataset/SearchTrend.csv")
timeprovince = pd.read_csv("/kaggle/input/coronavirusdataset/TimeProvince.csv")
timegender = pd.read_csv("/kaggle/input/coronavirusdataset/TimeGender.csv")
patientinfo = pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")
patientroute = pd.read_csv("/kaggle/input/coronavirusdataset/PatientRoute.csv")
case = pd.read_csv("/kaggle/input/coronavirusdataset/Case.csv")

case.head()
searchtrend.head()
timeage.head()
time.head()
timegender.head()
patientinfo.head()
time.head()
time.isnull().any()
time.info()
#barplot
plt.subplots(figsize = (20,17))
date_list = [i for i in time.date]
sns.barplot(x=time.test, y=date_list,color ='b',label="Number of test",alpha=0.7)
sns.barplot(x=time.negative, y=date_list,color ='lime',label = "Number of negative results",alpha=0.7)
sns.barplot(x=time.confirmed, y=date_list,color='red',label = "Number of confirmed cases",alpha=1)
plt.xlabel("VALUES",fontsize = 16)
plt.legend(prop={'size': 15})
plt.show()

time.head()
confirmednorm = time.confirmed/max(time.confirmed)
deceasednorm = time.deceased/max(time.deceased)
releasednorm = time.released/max(time.released)  # SIMPLE LINEARIZATION

data3 = pd.DataFrame({'confirmed': confirmednorm,'deceased' : deceasednorm,'released':releasednorm})


f,ax = plt.subplots(figsize = (27,17))

ax.plot_date(x=time.date,y= "deceased",data=data3,color ="red",label="Deceased People",linestyle = '-')
#ax.plot_date(x=time.date,y= "confirmed",data=data3,color ="mediumorchid",label="Confirmed cases",linestyle = '-')
ax.plot_date(x=time.date,y= "released",data=data3,color ="dodgerblue",label="Released People",linestyle = '-')
plt.xticks(rotation = 90)
plt.xlabel("DATE")
plt.ylabel("VALUES")
plt.title("DECEASED AND CONFIRMED BY DATE (NORMALIZED)")
plt.xlabel("DATES",fontsize = 16)
plt.legend(prop={'size': 20})
plt.grid()
plt.show()
timeage.head()
timeage.info()
f,ax = plt.subplots(figsize = (14,14))
ax = sns.barplot(x="age",y="deceased",data=timeage,palette="Blues_d")
plt.xlabel("AGE",fontsize = 20)
plt.xticks(fontsize = 17)
plt.show()
#The age's in the timeage dataframe has 's' in the and, so we have to remove them to make them float
timeage.age = [i.rsplit('s',1)[0] if 's' in i else i for i in timeage.age]
timeage.age.value_counts()
patientinfo.head()
patientinfo.state.value_counts()
patientinfo.info()
#Finding null rows with respect to age column
patientinfo[patientinfo.age.isnull()]
#Dropping the null values in the age column
patientinfo.dropna(subset=['age'],inplace=True)
patientinfo.age.isnull().any() # False = There are no null entries in the age column
#The age's in the patientinfo dataframe has 's' in the end, so we have to remove them to make them integer
patientinfo.age = [i.rsplit('s',1)[0] if 's' in i else i for i in patientinfo.age]
patientinfo.age.value_counts()
patientinfo.age = patientinfo.age.astype(int) #making entries integer
f,ax = plt.subplots(figsize = (23,15))
ax = sns.boxplot(x='sex',y="age",hue="state",data=patientinfo)
plt.legend(prop={'size' : 20})
plt.xlabel("GENDER",fontsize = 20)
plt.ylabel("AGE",fontsize = 20)
f,ax = plt.subplots(figsize = (10,15))
ax = sns.swarmplot(x="sex",y="age",hue="state",data=patientinfo)
plt.xlabel("GENDER",fontsize = 20)
plt.ylabel("AGE",fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()
timeage.head()
timeage.age = timeage.age.astype(float)
import scipy.stats as stats

j = sns.jointplot(x="age",y='deceased',data=timeage,kind="kde",size = 10)
j.annotate(stats.pearsonr)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 30)
plt.show()
j = sns.jointplot(x="age",y='deceased',data=timeage,size = 10,color="r")
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 30)
j.annotate(stats.pearsonr)
plt.show()
sns.lmplot(x="age",y='deceased',data=timeage,size = 10)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel("AGE",fontsize = 20)
plt.ylabel("DECEASED",fontsize = 20)
plt.show()
plt.subplots(figsize = (14,10))
sns.kdeplot(timeage.age,timeage.deceased,shade = True)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel("AGE",fontsize = 20)
plt.ylabel("DECEASED",fontsize = 20)
time.head()
#data4 = pd.concat([(time.released/max(time.released)),(time.deceased/max(time.deceased))],axis=1)
confirmed_ratio = (time.confirmed/time.test)
negative_ratio = (time.negative/time.test)
data4 = pd.DataFrame({'Confirmed_Ratio': confirmed_ratio, 'Negative_Ratio':negative_ratio})
#data4 = pd.concat([confirmed_ratio, negative_ratio],axis=1)
data4.head()
plt.subplots(figsize = (15,15))
mypalet = sns.color_palette("Set1", n_colors=2, desat=0.7)
sns.violinplot(data = data4, inner='point',palette = mypalet,linewidth=0.7)
plt.title("Confirmed and Negative Ratio to ALL TESTS")
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.show()
data5 = pd.concat([time.deceased, time.confirmed],axis=1)
data5.head()
f,ax = plt.subplots(figsize = (7,7))
ax = sns.heatmap(data5.corr(),annot = True,lw= 0.6, fmt='0.2f')
sns.pairplot(data5,size=3)
plt.show()
timegender.head()
timegender.sex.dropna(inplace = True)
labels = ["Confirmed","Deceased"] #or easily kill.race.value_counts().index
colors =  ['dodgerblue','gold']
explode = (0,0.8)
sizes = [np.mean(timegender.confirmed),np.mean(timegender.deceased)]

plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode,  labels=labels, colors=colors, autopct='%1.1f%%',shadow = True,startangle=18,wedgeprops={"edgecolor":"k",'linewidth': 1.5, 'antialiased': True},pctdistance = 0.7,textprops={'fontsize': 16})
plt.title('Deceased and Confirmed Ratio With Respect to All Tests',color = 'darkred',fontsize = 17)
plt.legend(prop={'size': 17})
plt.show()

time.head()
time15 = time.iloc[15:]

#Create Figure
fig = go.Figure()

#Add Traces
fig.add_trace(go.Scatter(x=time15.date, y=time15.released, name='Negative', text=time15.released))
fig.add_trace(go.Scatter(x=time15.date, y=time15.deceased, name='Deceased', text=time15.deceased))

#Update Layout
fig.update_layout(title='Released and Deceased Logarithmic',font_size=14, yaxis_type="log")

fig.show()
searchtrend.tail()
#Create Figures
fig = go.Figure()

#Add Traces
fig.add_trace(go.Scatter(x=searchtrend.date, y=searchtrend.cold, mode='markers',text=searchtrend.cold, name="Cold"))
fig.add_trace(go.Scatter(x=searchtrend.date, y=searchtrend.flu, mode='markers',text=searchtrend.flu, name="Flue"))
fig.add_trace(go.Scatter(x=searchtrend.date, y=searchtrend.pneumonia, mode='markers',text=searchtrend.pneumonia, name="Pneumonia"))
fig.add_trace(go.Scatter(x=searchtrend.date, y=searchtrend.coronavirus, mode='markers', text=searchtrend.coronavirus, name="Coronovirus"))

#Update Layout
fig.update_layout(title='Search Trends',font_size=13,font_color='black', yaxis_type="log")

#Update Traces
fig.update_traces(marker_size = 3.5,opacity=0.8)

fig.show()
timegender.head()
a = timegender.groupby('sex')['confirmed','deceased'].sum()
a
gender = ['Female', 'Male']

fig = go.Figure(data = [     go.Bar(x=a.index,y=a.confirmed,name="Confirmed")    ,      go.Bar(x=a.index, y=a.deceased,name="Deceased")]    )

fig.update_layout(barmode='group', yaxis_type="log")

fig.update_traces(marker_line_width=1,marker_line_color='black')

fig.show()

timeage.age = timeage.age.astype(int)
timeage.head()
death_by_age = timeage.groupby('age')['deceased'].sum()
death_by_age
#Create labels
labels = [str(each) +'s Years Old' for each in death_by_age.index]

#Create Figure
fig = go.Figure()

#Add traces
fig.add_trace(go.Pie(values= death_by_age, labels = labels, name="Deaths by Age",hoverinfo='label+percent+name'))

#Customize layout
fig.update_layout(title='Death Ratio by Age',font_size=15)

fig.show()
              

time.head()


fig= go.Figure()

fig.add_trace(go.Scatter(x=time.date, y=time.test, mode='markers', marker={'color' : time.confirmed, 'size':time.deceased, 'showscale':True,'colorbar':{'title':"Confirmed"} }, text='Size↑ → Deceased↑',textposition='bottom center' ) )


fig.update_layout(title='Coronovirus',font_size=14,font_color='black',xaxis_title='Time',yaxis_title='Number of Tests')

fig.show()
patientinfo.head()
#Create Figure
fig = go.Figure()

#Add Trace
fig.add_trace(go.Histogram(x=patientinfo.infection_case,opacity=0.8))

#Update Layout
fig.update_layout(title="Which kind of transmission infected how many people ?",font_size=14,xaxis_title="How?",yaxis_title="Value")

fig.show()
patientinfo2 = patientinfo.dropna(subset=["infection_case"])
patientinfo2[patientinfo2.infection_case.isnull()]

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(background_color='white', width=512, height=384).generate(" ".join(patientinfo2.infection_case))

plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#Dropping the null values in the age column
patientinfo.dropna(subset=['age'],inplace=True)
patientinfo.age.isnull().any() # False = There are no null entries in the age column
patientinfo.head()

fig = go.Figure()


fig.add_trace(go.Box(x=patientinfo.state,y=patientinfo.age, name="Deceased By Age"))

fig.update_layout(title="Status by Age",boxmode='group')


fig.show()
time.head()
import plotly.figure_factory as ff

mydata = time.loc[:,["test",'negative','confirmed']]
mydata["index"] = np.arange(1,len(mydata)+1)

fig = ff.create_scatterplotmatrix(mydata, index='index', diag='box',colormap='Blues',colormap_type='cat',height=700, width=700)

iplot(fig)
time.head()
fig = go.Figure()


fig.add_trace(go.Scatter3d(x=time.date, y=time.test, z=time.confirmed, mode='markers', marker_size = 6, marker_color=time.deceased, marker_colorscale = "Sunsetdark"))

fig.update_layout(scene=dict(xaxis_title='x:Date', yaxis_title="y:Number of Tests", zaxis_title="z:Number of Confirmed Cases"),font_size=8.5)

fig.show()
from plotly.subplots import make_subplots

fig = make_subplots(rows=2 , cols = 2)

fig.add_trace(go.Scatter(x=time.date, y=time.test, mode='lines', name="Tests"),row=1,col=1)
fig.add_trace(go.Scatter(x=time.date, y=time.confirmed, mode='markers', name="Confirmed"),row=1,col=2)
fig.add_trace(go.Scatter(x=time.date, y=time.negative, mode='lines', name="Negative"),row=2,col=1)
fig.add_trace(go.Scatter(x=time.date, y=time.deceased, mode='markers', name="Deceased"),row=2,col=2)



fig.show()



case.head()

fig = go.Figure(data=go.Scattergeo(
    lon =case.longitude,
    lat = case.latitude,
    text = case.city,
    mode = 'markers',
    marker= dict(
        size=8,
        line = dict(width=1,color = "white"),
        opacity=.8,
        color = 'red'
    )
))

fig.update_layout(title="Patients Locations",hovermode='closest')               
fig.show()