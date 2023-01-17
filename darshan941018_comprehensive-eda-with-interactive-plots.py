import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

pd.options.mode.chained_assignment = None





import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/crime.csv')

df.head()
msno.bar(df,sort=True)
df.drop(['incident_id','case_number','clearance_type','address_2','state','country','location'],1,inplace=True)
df['incident_year']=df['incident_datetime'].apply(lambda x: x.split()[0].split('/')[2])

df['incident_date']=df['incident_datetime'].apply(lambda x: x.split()[0])

df['incident_date']=pd.to_datetime(df.incident_date)

df=df[df['incident_date'].dt.year!=1969]
plt.figure(figsize=(12,10))

sns.countplot(x="hour_of_day",data=df,order=df.hour_of_day.value_counts().index)

plt.xticks(size=13)

plt.yticks(size=13)

plt.ylabel("Number of Crimes Committed",size=20)

plt.xlabel("Hour of Day",size=20)

plt.show()
plt.figure(figsize=(12,10))

sns.countplot(x="parent_incident_type",data=df,order=df.parent_incident_type.value_counts().index,palette="RdBu")

plt.xticks(size=13,rotation=90)

plt.yticks(size=13)

plt.ylabel("Number of Crimes Committed",size=20)

plt.xlabel("Hour of Day",size=20)

plt.show()
a=df.groupby('incident_date').count()['parent_incident_type']

df2=pd.DataFrame(a)
trace = go.Scatter(x=df2.index,y=df2.parent_incident_type)



data = [trace]



high_annotations=[dict(x=df2.parent_incident_type.idxmax(), y=df2.parent_incident_type.max(),

                       xref='x', yref='y',

                       text='Maximum number of Crimes took place on '+str(df2.parent_incident_type.idxmax()).split()[0]

                       +"<br>"+str(df2.parent_incident_type.max()), ax=0, ay=-40)]

low_annotations=[dict(x=df2.parent_incident_type.idxmin(),y=df2.parent_incident_type.min(),xref='x',yref='y',

                      text='Minimum number of Crimes took place on '+str(df2.parent_incident_type.idxmin()).split()[0]+"<br>"+str(df2.parent_incident_type.min()), ax=-40, ay=37)]                



updatemenus = list([

        dict(type="buttons",active=1,

             buttons=list([

                    dict(label = 'All',method = 'update',args = [{'visible': [True,True]},{'title': 'Crimes per day','annotations':high_annotations+low_annotations }]),

                    dict(label = 'Maximum',method = 'update',args = [{'visible': [True,False]},{'title': ' Maximum Crimes per day','annotations':high_annotations }]),

                    dict(label = 'Minimum',method = 'update',args = [{'visible': [True,True]},{'title': ' Minimum Crimes per day','annotations':low_annotations }])

                ]),

    )]

                  

)



layout = dict(

    title='Time Series Plot',

    xaxis=dict(

        range=['2011-01-01','2017-06-01'],

        rangeslider=dict(),

        type='date'

    ),

    updatemenus=updatemenus

    

)

fig = dict(data=data, layout=layout)

iplot(fig, filename = "Time Series with Rangeslider")
# Find out the types of parent incident type 

incident_type=df.parent_incident_type.value_counts()

incident_type=list(incident_type.index)





from collections import deque



## We have to build a list of true/false, in order to support a dropdown menu

## The for loop right shifts True by 1 position

visibility=[True]+[False]*22

d=[0]*24

for i in range(0,24):

    a=deque(visibility)

    a.rotate(i) 

    d[i]=list(a)
import plotly.plotly as py

from plotly.graph_objs import *



## Arrays for Maximum and Minimum crime incident of each incident type

max_annotations=[0]*len(incident_type)

min_annotations=[0]*len(incident_type)



## To calculate the total crimes committed for each incident type

sub_total=[0]*len(incident_type)



#To store traces for each of the type of Incident

data=[]
# Trace for all of the Incidents

df2=df.groupby('incident_date').count()['parent_incident_type']

df2=pd.DataFrame(df2)

data.append(Scatter(x=df2.index,y=df2.parent_incident_type,name="All"))



# The for loop will create a Scatter plot for each of the incident type and append it to data

for i in range(0,22):

    df2=df[df['parent_incident_type']==incident_type[i]]

    df2=df2.groupby('incident_date').count()['parent_incident_type']

    df2=pd.DataFrame(df2)

    data.append(Scatter(x=df2.index,y=df2.parent_incident_type,name=incident_type[i]))

    # max and min_annotations will help us to point to the day 

    # Maximum and Mimimum number of crimes committed for each Incident type

    max_annotations[i]=[dict(x=df2.parent_incident_type.idxmax(), y=df2.parent_incident_type.max(),

                       xref='x', yref='y',

                       text='Max on '+str(df2.parent_incident_type.idxmax()).split()[0]

                       +"<br>"+str(df2.parent_incident_type.max()), ax=0, ay=-40)]   

    min_annotations[i]=[dict(x=df2.parent_incident_type.idxmin(), y=df2.parent_incident_type.min(),

                       xref='x', yref='y',

                       text='Min on '+str(df2.parent_incident_type.idxmin()).split()[0]

                       +"<br>"+str(df2.parent_incident_type.min()), ax=-40, ay=37)] 

    # Total number of crimes for each incident type

    sub_total[i]=sum(df2.parent_incident_type)

    
data=Data(data)

updatemenus=list([

        dict(

            x=0.05,

            y=1.15,

            active=0,

            yanchor='top',

            buttons=list([

                dict(label = "All",method = 'update',args = [{'visible': d[0]},{'title': "Total crimes per day",'annotations':high_annotations+low_annotations}]),

                dict(label = incident_type[0],method = 'update',args = [{'visible': d[1]},{'title': incident_type[0]+' per day<br>Sub-total: '+str(sub_total[0]),'annotations':max_annotations[0]+min_annotations[0] }]),

                dict(label = incident_type[1],method = 'update',args = [{'visible': d[2]},{'title': incident_type[1]+' per day<br>Sub-total: '+str(sub_total[1]),'annotations':max_annotations[1]+min_annotations[1] }]),

                dict(label = incident_type[2],method = 'update',args = [{'visible': d[3]},{'title': incident_type[2]+' per day<br>Sub-total: '+str(sub_total[2]),'annotations':max_annotations[2]+min_annotations[2] }]),

                dict(label = incident_type[3],method = 'update',args = [{'visible': d[4]},{'title': incident_type[3]+' per day<br>Sub-total: '+str(sub_total[3]),'annotations':max_annotations[3]+min_annotations[3] }]),

                dict(label = incident_type[4],method = 'update',args = [{'visible': d[5]},{'title': incident_type[4]+' per day<br>Sub-total: '+str(sub_total[4]),'annotations':max_annotations[4]+min_annotations[4] }]),

                dict(label = incident_type[5],method = 'update',args = [{'visible': d[6]},{'title': incident_type[5]+' per day<br>Sub-total: '+str(sub_total[5]),'annotations':max_annotations[5]+min_annotations[5] }]),

                dict(label = incident_type[6],method = 'update',args = [{'visible': d[7]},{'title': incident_type[6]+' per day<br>Sub-total: '+str(sub_total[6]),'annotations':max_annotations[6]+min_annotations[6] }]),

                dict(label = incident_type[7],method = 'update',args = [{'visible': d[8]},{'title': incident_type[7]+' per day<br>Sub-total: '+str(sub_total[7]),'annotations':max_annotations[7]+min_annotations[7] }]),

                dict(label = incident_type[8],method = 'update',args = [{'visible': d[9]},{'title': incident_type[8]+' per day<br>Sub-total: '+str(sub_total[8]),'annotations':max_annotations[8]+min_annotations[8] }]),

                dict(label = incident_type[9],method = 'update',args = [{'visible': d[10]},{'title': incident_type[9]+' per day<br>Sub-total: '+str(sub_total[9]),'annotations':max_annotations[9]+min_annotations[9] }]),

                dict(label = incident_type[10],method = 'update',args = [{'visible': d[11]},{'title': incident_type[10]+' per day<br>Sub-total: '+str(sub_total[10]),'annotations':max_annotations[10]+min_annotations[10] }]),

                dict(label = incident_type[11],method = 'update',args = [{'visible': d[12]},{'title': incident_type[11]+' per day<br>Sub-total: '+str(sub_total[11]),'annotations':max_annotations[11]+min_annotations[11] }]),

                dict(label = incident_type[12],method = 'update',args = [{'visible': d[13]},{'title': incident_type[12]+' per day<br>Sub-total: '+str(sub_total[12]),'annotations':max_annotations[12]+min_annotations[12] }]),

                dict(label = incident_type[13],method = 'update',args = [{'visible': d[14]},{'title': incident_type[13]+' per day<br>Sub-total: '+str(sub_total[13]),'annotations':max_annotations[13]+min_annotations[13] }]),

                dict(label = incident_type[14],method = 'update',args = [{'visible': d[15]},{'title': incident_type[14]+' per day<br>Sub-total: '+str(sub_total[14]),'annotations':max_annotations[14]+min_annotations[14] }]),

                dict(label = incident_type[15],method = 'update',args = [{'visible': d[16]},{'title':incident_type[15]+' per day<br>Sub-total: '+str(sub_total[15]),'annotations':max_annotations[15]+min_annotations[15] }]),

                dict(label = incident_type[16],method = 'update',args = [{'visible': d[17]},{'title': incident_type[16]+' per day<br>Sub-total: '+str(sub_total[16]),'annotations':max_annotations[16]+min_annotations[16]}]),

                dict(label = incident_type[17],method = 'update',args = [{'visible': d[18]},{'title': incident_type[17]+' per day<br>Sub-total: '+str(sub_total[17]),'annotations':max_annotations[17]+min_annotations[17] }]),

                dict(label = incident_type[18],method = 'update',args = [{'visible': d[19]},{'title': incident_type[18]+' per day<br>Sub-total: '+str(sub_total[18]),'annotations':max_annotations[18]+min_annotations[18] }]),

                dict(label = incident_type[19],method = 'update',args = [{'visible': d[20]},{'title': incident_type[19]+' per day<br>Sub-total: '+str(sub_total[19]),'annotations':max_annotations[19]+min_annotations[19] }]),

                dict(label = incident_type[20],method = 'update',args = [{'visible': d[21]},{'title': incident_type[20]+' per day<br>Sub-total: '+str(sub_total[20]),'annotations':max_annotations[20]+min_annotations[20]}]),

                dict(label = incident_type[21],method = 'update',args = [{'visible': d[22]},{'title': incident_type[21]+' per day<br>Sub-total: '+str(sub_total[21]),'annotations':max_annotations[21]+min_annotations[21] }]),

                

                    

            ]),

        )

    ])



layout = Layout(updatemenus=updatemenus,xaxis=dict(range=['2011-01-01','2017-06-01'],type='date',title="Crimes committed over a period of time",),showlegend=False,)



fig = Figure(data=data, layout=layout)

iplot(fig)
sns.set_style("whitegrid")

plt.figure(figsize=(12,10))

sns.countplot(x='city',data=df)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.xlabel("City",size=18)

plt.ylabel("Number of Crimes",size=18)

plt.title("Crime by City",size=20)

plt.show()
plt.figure(figsize=(12,10))

from scipy.stats import norm

# sns.set_palette("Set1", 24, .75)

# sns.color_palette("RdBu", n_colors=24)

sns.despine()

sns.distplot(df['hour_of_day'],kde_kws={"color":"g","lw":4,"label":"KDE Estimation","alpha":0.5},

            hist_kws={"color":"r","alpha":0.5,"label":"Frequency"});

plt.xlim(0,24)

plt.xticks(np.arange(0,24),size=14)

plt.yticks(size=14)

plt.ylabel("Density",rotation=90, size=20)

plt.xlabel("Hour of Day",size=20)

plt.show()
dmap={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

df['day_of_week_numerical']=df['day_of_week'].map(dmap)

df.set_index('day_of_week',inplace=True,drop=True)
a=df['hour_of_day'].groupby(level='day_of_week').value_counts()

b=a.unstack(level=-1)

c = b.reindex(index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

e=c.transpose()
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":14})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":14})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Hour",size=18)

ax0.set_xlabel("Day of Week",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Number of Crimes committed each Day of Week for a given Hour",size=22,loc="right",y=1.05,x=1.1)



plt.show()

a=df['incident_date'].dt.year.groupby(level='day_of_week').value_counts()

b=a.unstack(level=-1)

c = b.reindex(index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

e=c.transpose()
fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":14})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":14})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Year",size=18)

ax0.set_xlabel("Day of Week",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Number of Crimes committed each Day of Week for a given Year",size=22,loc="right",y=1.05,x=1.1)



plt.show()
a=df['incident_date'].dt.month.groupby(level='day_of_week').value_counts()

b=a.unstack(level=-1)

c = b.reindex(index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

d=c.rename(columns={1:'Jan',2:'Feb',3:'Mar',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'})

e=d.transpose()
fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":14})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":14})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Month",size=18)

ax0.set_xlabel("Day of Week",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Number of Crimes commited each Day of Week for a given Month",size=22,loc="right",y=1.05,x=1.1)



plt.show()
a=df['parent_incident_type'].groupby(level='day_of_week').value_counts()

b=a.unstack(level=-1)

c = b.reindex(index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

e=c.transpose()

e=e.replace(np.nan,0)

e=e.astype(int)

day_incident_type=pd.DataFrame([e.idxmax(axis=1),e.max(axis=1)]).transpose()

day_incident_type.columns=['Day','Day_Count']
fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":14})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=pd.DataFrame(e.idxmax(axis=1)),fmt='s',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":14})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0,size=12)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Month",size=18)

ax0.set_xlabel("Day of Week",size=18)

ax1.set_xticklabels(["Max incidents on"],size=16)

ax0.set_title("Number of Crimes committed each Day of Week for a given Month",size=22,loc="right",y=1.05,x=1.1)



plt.show()
df_temp=df.reset_index()

df_temp=df_temp.set_index('hour_of_day')

a=df_temp['parent_incident_type'].groupby(level='hour_of_day').value_counts().unstack(level=-1)

e=a.transpose().replace(np.nan,0).astype(int)

hour_incident_type=pd.DataFrame([e.idxmax(axis=1),e.max(axis=1)]).transpose()

hour_incident_type.columns=['Hour',"Hour_Count"]



df_temp=df.reset_index()

df_temp=df_temp.set_index(df['incident_date'].dt.month)

a=df_temp['parent_incident_type'].groupby(level='incident_date').value_counts().unstack(level=-1)

e=a.transpose().replace(np.nan,0).astype(int)

month_incident_type=pd.DataFrame([e.idxmax(axis=1),e.max(axis=1)]).transpose()

month_incident_type.columns=['Month',"Month_Count"]



df_temp=df.reset_index()

df_temp=df_temp.set_index(df['incident_date'].dt.day)

a=df_temp['parent_incident_type'].groupby(level='incident_date').value_counts().unstack(level=-1)

e=a.transpose().replace(np.nan,0).astype(int)

date_incident_type=pd.DataFrame([e.idxmax(axis=1),e.max(axis=1)]).transpose()

date_incident_type.columns=['Date',"Date_Count"]
j=day_incident_type.join([month_incident_type,hour_incident_type,date_incident_type])
months={1:'Jan',2:'Feb',3:'Mar',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'}

j['Month']=j['Month'].map(months)

j['Day_Count']=j.Day_Count.astype(int)
j.Hour=j.Hour.apply(lambda x: str(x)+":00 to " +str(int(str(x).split(':')[0])+1)+":00" if x!=23 else str(x)+":00 to 0:00")
import warnings

warnings.filterwarnings('ignore')

j.Date=j.Date.astype(str)

for i in range(0,len(j)):

    if(j.Date[i]=='1' or j.Date[i]=='21' or j.Date[i]=='31'):

        j.Date[i]=str(j.Date[i])+"st"

    elif(j.Date[i]=='2' or j.Date[i]=='22'):

        j.Date[i]=str(j.Date[i])+"nd"

    elif(j.Date[i]=='3' or j.Date[i]=='23'):

        j.Date[i]=str(j.Date[i])+"rd"

    else:

        j.Date[i]=str(j.Date[i])+"th"

        
fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1]) 



ax0 = plt.subplot(gs[0])

sns.heatmap(pd.DataFrame(j.Month_Count),annot=pd.DataFrame(j.Month),fmt='s',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":14})





ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(j['Date_Count']),annot=pd.DataFrame(j.Date),fmt='s',linewidths=.5,ax=ax1, cbar=False,annot_kws={"size":14})



ax2=plt.subplot(gs[2],sharey=ax0)

sns.heatmap(pd.DataFrame(j['Day_Count']),annot=pd.DataFrame(j.Day),fmt='s',linewidths=.5,ax=ax2, cbar=False,annot_kws={"size":14})



ax3=plt.subplot(gs[3],sharey=ax0)

sns.heatmap(pd.DataFrame(j['Hour_Count']),annot=pd.DataFrame(j.Hour),fmt='s',linewidths=.5,ax=ax3, cbar=True,annot_kws={"size":14})



plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax2.get_yticklabels(), visible=False)

plt.setp(ax3.get_yticklabels(), visible=False)



plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax2.set_ylabel([]),visible=False)

plt.setp(ax3.set_ylabel([]),visible=False)



plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Type of Incident",size=18)



ax0.set_xticklabels(["Month"],size=16)

ax1.set_xticklabels(["Date"],size=16)

ax2.set_xticklabels(["Day"],size=16)

ax3.set_xticklabels(['Hour'],size=16)

ax0.set_title("Number of Crimes commited each Day of Week for a given Month",size=22,loc="right",y=1.05,x=3)



plt.show()
print("In total {} Traffic incidents took place of those {} took place on Friday".

      format(len(df[df['parent_incident_type']=='Traffic']),

             len(df[(df['parent_incident_type']=='Traffic') &(df['day_of_week_numerical']==5)])))

print("Of the {} Traffic incidents that took place on Friday {} took place between 16:00 to 17:00".format(

        len(df[(df['parent_incident_type']=='Traffic') &(df['day_of_week_numerical']==5)]),

      len(df[(df['parent_incident_type']=='Traffic') &(df['day_of_week_numerical']==5) & (df['hour_of_day']==16)])

     ))