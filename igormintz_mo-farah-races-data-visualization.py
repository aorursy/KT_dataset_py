import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
%matplotlib inline
pd.options.mode.chained_assignment = None
mo_farah=pd.read_excel('../input/mo farah.xlsx')
mo_farah.head(3)
def nametokm(name):
    if name=='10 Kilometres' or name=='10,000 Metres':
        return 10
    elif name=='800 Metres':
        return 0.8
    elif name=='1500 Metres' or name=='1500 Metres Indoor':
        return 1.5
    elif name=='3000 Metres' or name=='3000 Metres Indoor':
        return 3
    elif name=='5000 Metres' or name=='5000 Metres Indoor':
        return 5
    elif name=='Half Marathon':
        return 21
    elif name=='Marathon':
        return 42.195
mo_farah['KM']=mo_farah['EVENT'].apply(nametokm)
mo_farah.head()
def convert(mark):
    return (mark.hour*3600+mark.minute*60+mark.second)/60
     
mo_farah['MINUTES']=mo_farah['RESULT'].apply(convert)
short2long=['800 Metres', '1500 Metres', '1500 Metres Indoor', '3000 Metres', '3000 Metres Indoor', '5000 Metres', '5000 Metres Indoor', '10,000 Metres', '10 Kilometres', 'Half Marathon', 'Marathon'] 
ax=sns.stripplot(x="EVENT", y="MINUTES", data=mo_farah, order=short2long)
plt.rcParams['figure.figsize']=(10,5)
sns.set(font_scale = 1.5)
plt.setp(ax.get_xticklabels(), rotation=-90)
fig = ax.get_figure()
fig.savefig('min per race.png',bbox_inches = 'tight')
ByEventAvg=mo_farah.groupby('EVENT').mean()
ByEventAvg['KM']
ByEventAvg['KM per Hour']=pd.Series()
for i in range(0,len(ByEventAvg.index)):
    ByEventAvg['KM per Hour'][i]=ByEventAvg['KM'][i]/(ByEventAvg['MINUTES'][i]/60)
ByEventAvg.head()
ax=sns.stripplot(x=ByEventAvg.index, y="KM per Hour", data=ByEventAvg, order=short2long)
plt.rcParams['figure.figsize']=(10,5)
sns.set(font_scale = 1.5)
plt.setp(ax.get_xticklabels(), rotation=-90)
fig = ax.get_figure()
fig.savefig('kph.png',bbox_inches = 'tight')
events=pd.DataFrame(columns=['event','number'],data=mo_farah['EVENT'].value_counts())
mo_farah['EVENT'].value_counts().index
events['event']=mo_farah['EVENT'].value_counts().index
events['times raced']=mo_farah['EVENT'].value_counts().values
events
ax=sns.barplot(x='event',y='times raced',data=events)
plt.xticks(rotation=-90)
ax.get_figure()
fig.savefig('times raced.png',bbox_inches = 'tight')
fiveK=mo_farah[mo_farah['EVENT']=='5000 Metres'].sort_values(by='DATE')
threeK=mo_farah[mo_farah['EVENT']=='3000 Metres'].sort_values(by='DATE')
oneandhalf=mo_farah[mo_farah['EVENT']=='1500 Metres'].sort_values(by='DATE')
eight=mo_farah[mo_farah['EVENT']=='800 Metres'].sort_values(by='DATE')
tenk=mo_farah[mo_farah['EVENT']=='10,000 Metres'].sort_values(by='DATE')
tenroad=mo_farah[mo_farah['EVENT']=='10 Kilometres'].sort_values(by='DATE')
halfm=mo_farah[mo_farah['EVENT']=='Half Marathon'].sort_values(by='DATE')
marathon=mo_farah[mo_farah['EVENT']=='Marathon'].sort_values(by='DATE')

fig=plt.figure(figsize=(16,6))
plt.scatter(fiveK['DATE'].tolist(), fiveK['MINUTES'], s=100, label='5,000m', color='m')
plt.scatter(threeK['DATE'].tolist(), threeK['MINUTES'], s=100, label='3,000m', color='k')
plt.scatter(oneandhalf['DATE'].tolist(), oneandhalf['MINUTES'], s=100, label='1,500m', color='c')
plt.scatter(eight['DATE'].tolist(), eight['MINUTES'], s=100, label='800m', color='y')
plt.scatter(tenk['DATE'].tolist(), tenk['MINUTES'], s=100, label='10,000m', color='b')
plt.scatter(tenroad['DATE'].tolist(), tenroad['MINUTES'], s=100, label='10k', color='r')
plt.scatter(halfm['DATE'].tolist(), halfm['MINUTES'], s=100, label='21k', color='gold')
plt.scatter(marathon['DATE'].tolist(), marathon['MINUTES'], s=100, label='Marathon', color='coral')
plt.ylabel('Time in in minutes')
plt.suptitle("Mo Farah's race times")
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=18)
plt.legend()
plt.savefig('all races and times',bbox_inches='tight')
fig, ax=plt.subplots(7,figsize=(16,20))


ax[0].scatter(eight['DATE'].tolist(), eight['MINUTES'], s=100, label='800m', color='y')
ax[0].set_title('800m')
ax[0].set_ylabel('Minutes')
ax[1].scatter(oneandhalf['DATE'].tolist(), oneandhalf['MINUTES'], s=100, label='1,500m', color='c')
ax[1].set_title('1,500m')
ax[1].set_ylabel('Minutes')
ax[2].scatter(threeK['DATE'].tolist(), threeK['MINUTES'], s=100, label='3,000m', color='k')
ax[2].set_title('3,000m')
ax[2].set_ylabel('Minutes')
ax[3].scatter(fiveK['DATE'].tolist(), fiveK['MINUTES'], s=100, label='5,000m', color='m')
ax[3].set_title('5,000m')
ax[3].set_ylabel('Minutes')
ax[4].scatter(tenk['DATE'].tolist(), tenk['MINUTES'], s=100, label='10,000m', color='b')
ax[4].set_title('10,000m')
ax[4].set_ylabel('Minutes')
ax[5].scatter(tenroad['DATE'].tolist(), tenroad['MINUTES'], s=100, label='10k', color='r')
ax[5].set_title('10k road race')
ax[5].set_ylabel('Minutes')
ax[6].scatter(halfm['DATE'].tolist(), halfm['MINUTES'], s=100, label='21k', color='gold')
ax[6].set_title('Half Marathon')
ax[6].set_ylabel('Minutes')
for i in range(7):
    ax[i].xaxis.set_tick_params(labelsize=20)
    ax[i].yaxis.set_tick_params(labelsize=15)



plt.tight_layout()
plt.style.use('seaborn')

plt.savefig('every race own graph',bbox_inches='tight')
fiveK.dropna(inplace=True)
threeK.dropna(inplace=True)
oneandhalf.dropna(inplace=True)
tenk.dropna(inplace=True)
eight.dropna(inplace=True)
tenroad.dropna(inplace=True)
halfm.dropna(inplace=True)
marathon.dropna(inplace=True)

fiveK['PL.'].apply(int)
threeK['PL.'].apply(int)
oneandhalf['PL.'].apply(int)
tenk['PL.'].apply(int)
eight['PL.'].apply(int)
tenroad['PL.'].apply(int)
halfm['PL.'].apply(int)
marathon['PL.'].apply(int)

fig, ax=plt.subplots(7,figsize=(16,25))


ax[0].scatter(eight['DATE'].tolist(), eight['PL.'], s=100, label='800m', color='y')
ax[0].set_title('800m', size=20)
ax[0].set_ylabel('Place', size=20)
ax[1].scatter(oneandhalf['DATE'].tolist(), oneandhalf['PL.'], s=100, label='1,500m', color='c')
ax[1].set_title('1,500m', size=20)
ax[1].set_ylabel('Place', size=20)
ax[2].scatter(threeK['DATE'].tolist(), threeK['PL.'], s=100, label='3,000m', color='k')
ax[2].set_title('3,000m', size=20)
ax[2].set_ylabel('Place', size=20)
ax[3].scatter(fiveK['DATE'].tolist(), fiveK['PL.'], s=100, label='5,000m', color='m')
ax[3].set_title('5,000m', size=20)
ax[3].set_ylabel('Place', size=20)
ax[4].scatter(tenk['DATE'].tolist(), tenk['PL.'], s=100, label='10,000m', color='b')
ax[4].set_title('10,000m', size=20)
ax[4].set_ylabel('Place', size=20)
ax[5].scatter(tenroad['DATE'].tolist(), tenroad['PL.'], s=100, label='10k', color='r')
ax[5].set_title('10k road race', size=20)
ax[5].set_ylabel('Place', size=20)
ax[6].scatter(halfm['DATE'].tolist(), halfm['PL.'], s=100, label='21k', color='gold')
ax[6].set_title('Half Marathon', size=20)
ax[6].set_ylabel('Place', size=20)
for i in range(7):
    ax[i].xaxis.set_tick_params(labelsize=20)
    ax[i].yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
plt.style.use('seaborn')
plt.savefig('places',bbox_inches='tight')
