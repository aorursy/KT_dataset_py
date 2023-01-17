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
covid = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
top5mostdeaths = covid[covid['Date']=='4/7/20'].sort_values(by='Deaths', ascending = False).head(5)
figdeaths,ax1 = plt.subplots(dpi=200)
length = np.arange(len(top5mostdeaths))

ax1.bar(length,top5mostdeaths['Deaths'])
ax1.set_xticks(length)
ax1.set_xticklabels(top5mostdeaths['Country/Region'],fontsize=8)
ax1.set_ylabel('Deaths')
ax1.set_xlabel('Countries')
ax1.set_title('Countries with the most Coronavirus deaths')
withouttop5 = covid[covid['Date']=='4/7/20']['Deaths'].sum()-top5mostdeaths['Deaths'].sum()
deaths = list(top5mostdeaths['Deaths'])
deaths.append(withouttop5)
deaths
countrylabelspiechart = list(top5mostdeaths['Country/Region'])
countrylabelspiechart.append('Others')

figdeathpiechart,ax2 = plt.subplots(dpi=200)
ax2.pie(deaths,labels=countrylabelspiechart,autopct='%1.1f%%',labeldistance=1.2,colors=['crimson','blue','green','orange','violet','yellow'],wedgeprops={'linewidth':2,'edgecolor':'black'})
ax2.set_title('Coronovirus Deaths Worldwide')

figworldwidedeaths,ax3 = plt.subplots(dpi=200)
ax3.plot(covid['Date'],covid['Deaths'])
ax3.set_ylabel('Deaths', fontsize=15)
ax3.set_xlabel('Time ->', fontsize = 15)
ax3.set_title('Worldwide Coronavirus Deaths', fontsize = 20)
figtop5,ax5 = plt.subplots(1,4,figsize=(25,5))
width=0.5
top5mostdeaths['NewIndex'] = ['Italy','Spain','US','France','UK']
top5mostdeaths = top5mostdeaths.reindex().set_index('NewIndex')


ax5[0].set_xlim([0,1])
ax5[0].bar(0.5,top5mostdeaths['Deaths'].loc['Italy'],width,label='Death')
ax5[0].bar(0.5,top5mostdeaths['Recovered'].loc['Italy'],width, bottom=top5mostdeaths['Deaths'],label='Recovered')
ax5[0].legend()
ax5[0].set_ylabel('Cases/People')
ax5[0].set_title('Italy',fontsize=20)

ax5[1].set_xlim([0,1])
ax5[1].bar(0.5,top5mostdeaths['Deaths'].loc['Spain'],width,label='Death')
ax5[1].bar(0.5,top5mostdeaths['Recovered'].loc['Spain'],width, bottom=top5mostdeaths['Deaths'],label='Recovered')
ax5[1].legend(loc=1)
ax5[1].set_ylabel('Cases/People')
ax5[1].set_title('Spain',fontsize=20)

ax5[2].set_xlim([0,1])
ax5[2].bar(0.5,top5mostdeaths['Deaths'].loc['US'],width,label='Death')
ax5[2].bar(0.5,top5mostdeaths['Recovered'].loc['US'],width, bottom=top5mostdeaths['Deaths'],label='Recovered')
ax5[2].legend(loc=1)
ax5[2].set_ylabel('Cases/People')
ax5[2].set_title('United States',fontsize=20)

ax5[3].set_xlim([0,1])
ax5[3].bar(0.5,top5mostdeaths['Deaths'].loc['France'],width,label='Death')
ax5[3].bar(0.5,top5mostdeaths['Recovered'].loc['France'],width, bottom=top5mostdeaths['Deaths'],label='Recovered')
ax5[3].legend(loc=1)
ax5[3].set_ylabel('Cases/People')
ax5[3].set_title('France',fontsize=20)
top10countriesdeaths = covid[(covid['Date']=='4/7/20')].groupby('Country/Region').sum()['Deaths'].sort_values(ascending=False).head(10)

figscatter,ax11 = plt.subplots(dpi=200)
ax11.scatter(y=top10countriesdeaths.loc['Italy'],x=47.3,label='Italy')
ax11.scatter(y=top10countriesdeaths.loc['Spain'],x=44.9,label='Spain')
ax11.scatter(y=top10countriesdeaths.loc['US'],x=38.3,label='US')
ax11.scatter(y=top10countriesdeaths.loc['France'],x=42.3,label='France')
ax11.scatter(y=top10countriesdeaths.loc['United Kingdom'],x=40.5,label='United Kingdom')
ax11.scatter(y=top10countriesdeaths.loc['Iran'],x=32.0,label='Iran')
ax11.scatter(y=top10countriesdeaths.loc['China'],x=38.4,label='China')
ax11.scatter(y=top10countriesdeaths.loc['Netherlands'],x=43.3,label='Netherlands')
ax11.scatter(y=top10countriesdeaths.loc['Germany'],x=45.7,label='Germany')
ax11.scatter(y=top10countriesdeaths.loc['Belgium'],x=41.9,label='Belgium')
ax11.legend()
ax11.set_ylabel('Deaths')
ax11.set_xlabel('Median Age')
ax11.set_title('Correlation between Deaths and Median Age of Top 10 countries with the most deaths')
covidindia = pd.read_csv('../input/coronavirus-cases-in-india/Covid cases in India.csv')
indianstatedeaths = covidindia['Deaths']
indianstatenames = covidindia['Name of State / UT']

barcolors = []
import random

#r = lambda: random.randint(0,255)
#for x in range(0,30):
 #   barcolors.append('#%02X%02X%02X' % (r(),r(),r()))
barcolors = ['#A50BE2',
 '#599A9D',
 '#E0228F',
 '#E733ED',
 '#405AEE',
 '#AAAA40',
 '#0BB511',
 '#E050C1',
 '#1D32A5',
 '#24CFCE',
 '#3B193C',
 '#8259B4',
 '#CF07B6',
 '#C0DD72',
 '#B63E3F',
 '#9F066C',
 '#9E7588',
 '#F74924',
 '#01DCCD',
 '#5BF48A',
 '#60C111',
 '#1A58F9',
 '#50DB7A',
 '#0DD170',
 '#7ACBDF',
 '#9BD787',
 '#C4EC5E',
 '#48523D',
 '#5FC53F',
 '#A382D0']

figindianstates,ax6 = plt.subplots(dpi=250)
indianstatelength = np.arange(len(indianstatenames))

ax6.barh(indianstatelength,indianstatedeaths,color=barcolors)
ax6.set_yticks(indianstatelength)
ax6.set_yticklabels(indianstatenames,fontsize=5)
ax6.set_ylabel('States/UT')
ax6.set_xlabel('Deaths')
ax6.set_title('Deaths by State (India)')

figindianstatespiechart,ax7 = plt.subplots(figsize=(8,3),dpi=250)
ax7.pie(covidindia['Deaths'],colors=barcolors,wedgeprops={'linewidth':1,'edgecolor':'black','linestyle':'dashed'})
ax7.legend(indianstatenames,loc=4,prop={'size':4},bbox_to_anchor=(1, 0, 0.5, 1))
ax7.set_title('Indian States Coronavirus Pie Chart', fontsize = 8)
indiancovidinfo = covid[covid['Country/Region']=='India']

figindianplot,ax8 = plt.subplots(1,3,dpi=200,figsize=(15,10))

ax8[0].plot(indiancovidinfo['Date'],indiancovidinfo['Deaths'],'r--')
ax8[0].set_ylabel('Cases',fontsize=10)
ax8[0].set_title('Deaths')

ax8[1].plot(indiancovidinfo['Date'],indiancovidinfo['Recovered'],'b--')
ax8[1].set_xlabel('---------> Date -------->',fontsize = 20)
ax8[1].set_title('Recovered')

ax8[2].plot(indiancovidinfo['Date'],indiancovidinfo['Confirmed'],'g--')
ax8[2].set_title('Confirmed Cases')
figindianmerge,ax9 = plt.subplots(dpi=200)

ax9.plot(indiancovidinfo['Date'],indiancovidinfo['Deaths'],'r--',label='Deaths')
ax9.plot(indiancovidinfo['Date'],indiancovidinfo['Recovered'],'b--',label='Recovered')
ax9.plot(indiancovidinfo['Date'],indiancovidinfo['Confirmed'],'g--',label='Confirmed Cases')
ax9.legend()
ax9.set_ylabel('Deaths')
ax9.set_xlabel('Date -->')

covidindiaconfirmedactions = covidindia[(covidindia['Cured/Discharged/Migrated']>0)|(covidindia['Deaths']>0)]
covidindiaconfirmedactions


figcovidindiaconfirmed,ax10 = plt.subplots(dpi=200)
confirmedactionslength = np.arange(len(covidindiaconfirmedactions))

ax10.barh(confirmedactionslength,covidindiaconfirmedactions['Deaths'],label='Deaths')
ax10.barh(confirmedactionslength,covidindiaconfirmedactions['Cured/Discharged/Migrated'],left=covidindiaconfirmedactions['Deaths'],label='Cured/Discharged/Migrated')
ax10.set_yticks(confirmedactionslength)
ax10.set_yticklabels(covidindiaconfirmedactions['Name of State / UT'],fontsize=5)
ax10.legend(prop={'size':6})
ax10.set_xlabel('Cases/People')
ax10.set_ylabel('States/UTs')