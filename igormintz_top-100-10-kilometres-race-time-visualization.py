import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import datetime
%matplotlib inline

tenk=pd.read_excel('../input/10k.xlsx')
tenk.head(30)

def convert(mark):
    mins=mark.minute
    secs=mark.second
    micro=mark.microsecond
    return secs*1000000+mins*1000000*60+micro
tenk['MICRO']=tenk['MARK'].apply(convert)
#sns.distplot(a=tenk['MICRO']/60000000,axlabel='minutes',label='distribution of top 100 10k times 2001-2018')
sns.set_context("paper", rc={"axes.labelsize":30})
plt.rcParams['figure.figsize']=(15,6)
sns.set(font_scale = 2)
ax = sns.distplot(kde=False,a=tenk['MICRO']/60000000,axlabel='minutes',label='distribution of top 100 10k times 2001-2018')
fig = ax.get_figure()
fig.savefig('distplot.png',bbox_inches = 'tight')

sns.set_palette("colorblind", n_colors=10)
ax=sns.kdeplot(tenk[tenk['YEAR']==2017]['MICRO']/60000000, label=2017)
ax=sns.kdeplot(tenk[tenk['YEAR']==2016]['MICRO']/60000000, label=2016)
ax=sns.kdeplot(tenk[tenk['YEAR']==2015]['MICRO']/60000000, label=2015)
ax=sns.kdeplot(tenk[tenk['YEAR']==2014]['MICRO']/60000000, label=2014)
ax=sns.kdeplot(tenk[tenk['YEAR']==2013]['MICRO']/60000000, label=2013)
ax=sns.kdeplot(tenk[tenk['YEAR']==2012]['MICRO']/60000000, label=2012)
ax=sns.kdeplot(tenk[tenk['YEAR']==2011]['MICRO']/60000000, label=2011)
ax=sns.kdeplot(tenk[tenk['YEAR']==2010]['MICRO']/60000000, label=2010)
ax=sns.kdeplot(tenk[tenk['YEAR']==2009]['MICRO']/60000000, label=2009)
ax=sns.kdeplot(tenk[tenk['YEAR']==2008]['MICRO']/60000000, label=2008)
ax=sns.kdeplot(tenk[tenk['YEAR']==2007]['MICRO']/60000000, label=2007)
ax=sns.kdeplot(tenk[tenk['YEAR']==2006]['MICRO']/60000000, label=2006)
ax=sns.kdeplot(tenk[tenk['YEAR']==2005]['MICRO']/60000000, label=2005)
ax=sns.kdeplot(tenk[tenk['YEAR']==2004]['MICRO']/60000000, label=2004)
ax=sns.kdeplot(tenk[tenk['YEAR']==2003]['MICRO']/60000000, label=2003)
ax=sns.kdeplot(tenk[tenk['YEAR']==2002]['MICRO']/60000000, label=2002)
ax=sns.kdeplot(tenk[tenk['YEAR']==2001]['MICRO']/60000000, label=2001)
ax.set_title('distribution of top 100 times in differen years')

averages=[]
years=range(2001,2019)
for i in years:
    averages.append(tenk[tenk['YEAR']==i]['MICRO'].mean()/60000000)
ave_dict = dict(zip(years, averages))
ave_ser=pd.Series(ave_dict)
fig=plt.figure()
plt.figure(figsize=(16,6))
plt.plot(ave_ser)
plt.xlabel('year', size=20)
plt.ylabel('time in minutes', size=20)
plt.suptitle("average top 100 10k times per year", position=(0.5, 1.02), size=20)
plt.xticks(years, size=20)
plt.yticks(size=20)
plt.grid(which='major', axis='both')
plt.style.use('fivethirtyeight')
plt.tight_layout()

plt.savefig('average top 100 10k times per year', bbox_inches = 'tight')
ax = sns.barplot(x=tenk['NAT'].value_counts().index, y=tenk['NAT'].value_counts().values).set_title("number of times a runner fron a certain nationallity got to the best 100 the same year")
sns.set(rc={'figure.figsize':(26,8.27)})
sns.set(font_scale = 2)
fig = ax.get_figure()
fig.savefig('by nationality.png',bbox_inches = 'tight')

ken_res=tenk[tenk['NAT']=='KEN']['MICRO']/60000000
ken_year=tenk[tenk['NAT']=='KEN']['YEAR']
ken_df=pd.concat([ken_year, ken_res], axis=1)
eth_res=tenk[tenk['NAT']=='ETH']['MICRO']/60000000
eth_year=tenk[tenk['NAT']=='ETH']['YEAR']
eth_df=pd.concat([eth_year, eth_res], axis=1)
usa_res=tenk[tenk['NAT']=='USA']['MICRO']/60000000
usa_year=tenk[tenk['NAT']=='USA']['YEAR']
usa_df=pd.concat([usa_year, usa_res], axis=1)
jpn_res=tenk[tenk['NAT']=='JPN']['MICRO']/60000000
jpn_year=tenk[tenk['NAT']=='JPN']['YEAR']
jpn_df=pd.concat([jpn_year, jpn_res], axis=1)
uga_res=tenk[tenk['NAT']=='UGA']['MICRO']/60000000
uga_year=tenk[tenk['NAT']=='UGA']['YEAR']
uga_df=pd.concat([uga_year, uga_res], axis=1)
fig=plt.figure(figsize=(16,8))
plt.plot(ken_df.groupby('YEAR').mean(), label='KEN', linewidth=5.0)
plt.plot(eth_df.groupby('YEAR').mean(), label='ETH', linewidth=5.0)
plt.plot(usa_df.groupby('YEAR').mean(), label='USA', linewidth=5.0)
plt.plot(jpn_df.groupby('YEAR').mean(), label='JPN', linewidth=5.0)
plt.plot(uga_df.groupby('YEAR').mean(), label='UGA', linewidth=5.0)
plt.suptitle('average times of 10k races of the countries most frequent at the top 100 times', fontsize=20)
plt.ylabel('time in minutes')
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(prop={'size': 18})
plt.savefig('average times of 10k races of the countries most frequent at the top 100 times')
ax=sns.barplot(x=tenk['COMPETITOR'].value_counts().index[:10], y=tenk['COMPETITOR'].value_counts().values[:10]).set_title("number of times a runner got to the best 100: 2001-2018")
sns.set(rc={'figure.figsize':(26,8.27)})
sns.set(font_scale = 2)
plt.xticks(rotation=45)
fig = ax.get_figure()
fig.savefig('frequent runners.png',bbox_inches = 'tight')
top_10_names=tenk['COMPETITOR'].value_counts().index[:10]
top_10_table=pd.DataFrame(data=None, index=top_10_names, columns=range(2001,2018))
tempo=0
for year in range(2001,2018):
    for runner in top_10_names:
        tempo=tenk[(tenk['COMPETITOR']==runner)&(tenk['YEAR']==year)]['MICRO']/60000000
        tempo=tempo.mean()
        top_10_table.xs(runner)[year]=tempo

top_10_table
colors=['m','k','c','y','b','r','gold','coral','grey','peru']
fig=plt.figure(figsize=(18,8))
for i in range(10):
    plt.plot(top_10_table.columns,top_10_table.iloc[i],label=top_10_table.index[i], color=colors[i])
plt.suptitle('average times of 10k races of the runners most frequent at the top 100 times', fontsize=20)
plt.ylabel('time in minutes')
plt.xticks(years,size=18)
plt.yticks(size=18)
plt.legend(prop={'size': 15})
plt.savefig('average times of 10k races of the runners most frequent at the top 100 times')
top_10_names2=tenk['COMPETITOR'].value_counts().index[:10]
top_10_table2=pd.DataFrame(data=None, index=top_10_names2, columns=range(2001,2018))
tempo=0
for year in range(2001,2018):
    for runner in top_10_names2:
        tempo2=tenk[(tenk['COMPETITOR']==runner)&(tenk['YEAR']==year)]['MICRO']/60000000
        tempo2=tempo2.min(skipna=True)
        top_10_table2.xs(runner)[year]=tempo2
top_10_table2.head()
colors=['m','k','c','y','b','r','gold','coral','grey','peru']
fig=plt.figure(figsize=(18,8))
for i in range(10):
    plt.plot(top_10_table2.columns,top_10_table2.iloc[i],label=top_10_table2.index[i], color=colors[i])
plt.suptitle('fastest times of 10k races of the runners most frequent at the top 100 times', fontsize=20)
plt.ylabel('time in minutes')
plt.xticks(years,size=18)
plt.yticks(size=18)
plt.legend(prop={'size': 15})
plt.savefig('fastest times of 10k races of the runners most frequent at the top 100 times')

#חלוקה לחמישונים
quantiles=pd.DataFrame(index=['first','second', 'third', 'fourth', 'fifth'],columns=range(2001,2018))
q=[0.2, 0.4, 0.6, 0.8, 1]
for i in range(2001,2018):
    temp_q=(tenk[tenk['YEAR']==i]['MICRO']/60000000).quantile(q)
    quantiles.xs('first')[i]=temp_q.values[0]
    quantiles.xs('second')[i]=temp_q.values[1]
    quantiles.xs('third')[i]=temp_q.values[2]
    quantiles.xs('fourth')[i]=temp_q.values[3]
    quantiles.xs('fifth')[i]=temp_q.values[4]
    
quantiles.head()
fig=plt.figure(figsize=(18,5))
plt.plot(range(2001,2018),quantiles.loc['first'],label='first' )
plt.plot(range(2001,2018),quantiles.loc['second'],label='second')
plt.plot(range(2001,2018),quantiles.loc['third'], label='third')
plt.plot(range(2001,2018),quantiles.loc['fourth'], label='fourth')
plt.plot(range(2001,2018),quantiles.loc['fifth'], label='fifth')
plt.text(x=2000.5,y=27.6, s='1st', size=20)
plt.text(x=2000.5,y=27.9, s='2nd', size=20)
plt.text(x=2000.5,y=28.05, s='3rd', size=20)
plt.text(x=2000.5,y=28.15, s='4th', size=20)
plt.text(x=2000.5,y=28.26, s='5th', size=20)
plt.suptitle('top 100 10k running times by quantiles', fontsize=20)
plt.ylabel('time in minutes')
plt.xticks(range(2001,2018),size=18)
plt.yticks(size=18)

plt.savefig('top 100 10k running times by quantiles')
kipngetich=tenk[tenk['COMPETITOR']=='Paul Kipngetich TANUI'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
muchiri=tenk[tenk['COMPETITOR']=='Josphat Muchiri NDAMBIRI'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
martin=tenk[tenk['COMPETITOR']=='Martin Irungu MATHATHI'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
karoki=tenk[tenk['COMPETITOR']=='Bitan KAROKI'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
kipngetich.sort_values(by='DATE',inplace=True)
muchiri.sort_values(by='DATE', inplace=True)
martin.sort_values(by='DATE', inplace=True)
karoki.sort_values(by='DATE', inplace=True)
fig=plt.figure(figsize=(18,10))
plt.plot(kipngetich['DATE'],kipngetich['MICRO']/60000000, label='Kipngetich')
plt.plot(muchiri['DATE'],muchiri['MICRO']/60000000, label='Muchiri')
plt.plot(martin['DATE'],martin['MICRO']/60000000, label='Irungu')
plt.plot(karoki['DATE'],karoki['MICRO']/60000000, label='Karoki')
plt.legend()
best_times=tenk.sort_values(by='MICRO').head(20)
best_times_short=best_times[['MARK','COMPETITOR', 'NAT']]
best_times_short
bekele=tenk[tenk['COMPETITOR']=='Kenenisa BEKELE'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
gebrselassie=tenk[tenk['COMPETITOR']=='Haile GEBRSELASSIE'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)
kemboi=tenk[tenk['COMPETITOR']=='Nicholas KEMBOI'].drop(['RANK','COMPETITOR','DOB','YEAR','MARK','NAT','POS','VENUE'],axis=1)

bekele.sort_values(by='DATE',inplace=True)
gebrselassie.sort_values(by='DATE', inplace=True)
kemboi.sort_values(by='DATE', inplace=True)
fig=plt.figure(figsize=(18,8))
plt.plot(bekele['DATE'],bekele['MICRO']/60000000, label='Bekele')
plt.plot(gebrselassie['DATE'],gebrselassie['MICRO']/60000000, label='Gebrselassie')
plt.plot(kemboi['DATE'],kemboi['MICRO']/60000000, label='Kemboi')
plt.suptitle('fastest runners 10k times-Bekele is king', fontsize=20)
plt.ylabel('time in minutes')
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(prop={'size': 20})
plt.savefig('fastest runners 10k times-Bekele is king')
#temp_tenk=tenk.dropna()
alist=[]
for i in range(0, len(tenk.index)):
            alist.append(tenk.iloc[i]['DATE'].year-tenk.iloc[i]['DOB'].year)
se=pd.Series(alist)
tenk['AGE AT RACE']=se.values
new=tenk.sort_values(by='AGE AT RACE')
normalage=new[new['AGE AT RACE']<96]
normalage['MICRO'].apply(lambda x: x/60000000)
normalage.dropna()
unique_names=normalage['COMPETITOR'].unique()
times=[]
ages=[]
for runner in unique_names:
    last=normalage[normalage['COMPETITOR']==runner].sort_values(by='MICRO')
    times.append(last['MICRO'].iloc[0])
    ages.append(last['AGE AT RACE'].iloc[0])
timesVSages=pd.DataFrame()
timesVSages['names']=unique_names
timesse=pd.Series(times)
agesse=pd.Series(ages)
timesVSages['times']=timesse.values/60000000
timesVSages['ages']=agesse.values
ax=sns.jointplot(y='times',x='ages',data=timesVSages, kind='hex')
#sns.set(rc={'figure.figsize':(26,8.27)})
sns.set(font_scale = 2)
plt.xticks(rotation=45)
ax.fig.set_figwidth(10)
ax.fig.set_figheight(10)
fig.savefig('runnerage.png',bbox_inches = 'tight')
where=tenk.groupby('VENUE').mean()
where.sort_values(by='MICRO', inplace=True)
short_where=where.head(30)
short_where.head()
ax=sns.barplot(x=short_where.index, y=short_where['MICRO']/60000000, data=short_where)
sns.set(rc={'figure.figsize':(26,10)})
sns.set(font_scale = 2)
plt.xticks(rotation=-45)
plt.yticks(range(0,30,2))
plt.ylabel('time')
plt.xlabel('race location')
plt.suptitle('average race times in different tracks (based on top 100 times, 2001-2017)')
fig = ax.get_figure()
fig.savefig('fastest track.png',bbox_inches = 'tight')
