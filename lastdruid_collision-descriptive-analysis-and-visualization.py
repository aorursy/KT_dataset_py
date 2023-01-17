%matplotlib inline
%precision 3

import os
import re
import datetime

import numpy as np
np.set_printoptions(precision=3)
np.random.seed(123)
from numpy.random import randn
from numpy import nan as NA

import pandas as pd
pd.options.display.max_rows = 6

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))

from pandas.plotting import autocorrelation_plot
from pylab import pcolor, show, colorbar, xticks, yticks
from matplotlib.ticker import PercentFormatter

import seaborn as sns
accident = pd.read_csv('../input/NCDB_1999_to_2014.csv', engine = 'python')
#Replace all column head to lower case.
accident.columns = accident.columns.map(lambda x: x[:].lower())
accident
#Check the non-numeric values in date-time columns
accident[accident['c_mnth'].str.contains('[^0-9]')|
         accident['c_wday'].str.contains('[^0-9]')|
         accident['c_hour'].str.contains('[^0-9]')]
#Remove all special values (unknown to us) in date-time columns, prepare for using date series as index.
#Make a copy "df" for further analysis, avoid mess up the original data "accident".

df = accident[:]
df[['c_mnth','c_wday','c_hour']] = df[['c_mnth','c_wday','c_hour']].replace('[^0-9]+',np.nan,regex=True)
df.dropna(axis=0,subset=['c_mnth','c_wday','c_hour'],inplace=True)

df
#Generate the date-time column "date", which could be assigned as index later.
#Notes: 'dfp' using 'PeriodIndex' to generate monthly index, which meke better sense, but very slow.
dfp=df[:]
df['date'] = pd.DatetimeIndex(df['c_year'].map(str) + '-' + df['c_mnth'])
dfp['date'] = pd.PeriodIndex(dfp['c_year'].map(str) + '-' + dfp['c_mnth'], freq='M')

df
#From 'dfp' generating 'df_b', removed non-numeric strings from four other columns.
df_b = dfp[:]
df_b[['c_rcfg','c_wthr','c_rsur','c_traf']] =\
        df_b[['c_rcfg','c_wthr','c_rsur','c_traf']].replace('[^0-9]+',np.nan,regex=True)
df_b.dropna(axis=0,subset=['c_rcfg','c_wthr','c_rsur','c_traf'],inplace=True)

#This line use 'date' as index, meke 'df_b' becoming time series.
df_b = df_b.set_index('date')
df_b
df_b['fatal']=np.where(df_b['c_sev']==1,1,0)
df_b['non_fatal']=np.where(df_b['c_sev']==2,1,0)

df1 = df_b.groupby('date')['fatal','non_fatal'].sum()
df1
#The overall diagram
plot1 = df1.plot(figsize=(15,5),title='Collision overall statistics')
plot1.set_xlabel("Date")
plot1.set_ylabel("Number of collisions");
# Since the dateset period is 15 year, it's diffucult to identify a seasonality pattern in this scale
# Thus, a smaller period was selected to identify on-peak and off-peak of collisions
plot1 = df1[-60:].plot(figsize=(15,5),
                     title='Collision overall statistics (recent 5 year)')
plot1.set_xlabel("Date")
plot1.set_ylabel("Number of collisions");
#Due to a huge difference in number of fatal and non-fatal collisions,
# fatal collisions trend was analyzed in its own scale.
#Overall statistics
plot2 = df1['fatal'].plot(figsize=(15,5),title='Fatal collision statistics')
plot2.set_xlabel("Date")
plot2.set_ylabel("Number of collisions");
# Fatal collision statistics for the recent 5 years
plot1 = df1['fatal'][-60:].plot(figsize=(15,5),title='Fatal collision statistics (recent 5 year)')
plot1.set_xlabel("Date")
plot1.set_ylabel("Number of collisions");
#To confirm the seasonality assumtion, the autocorrelation of fatal and
# non-fatal collisions was performed for the recent 5 years.
plt.figure(figsize=(15,5))
for c in df1.columns:
    autocorrelation_plot(df1[c][-60:],label=c);
sns.regplot('fatal', 'non_fatal', data=df1);
corr = df1.corr()
corr
#Collisions by weekday
by_weekday = df_b.groupby('c_wday')['c_sev'].count()
by_weekday.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plot2 = by_weekday.plot(kind='bar',title='Collisions by week day');
#Collisions by hour
plt.figure(figsize=(15,5))
by_hour = df_b.groupby('c_hour')['c_sev'].count()
plot3 = by_hour.plot(kind='bar',title='Collisions by week hour',color='G');
#Collisions severity by road configuration (c_rcfg field)
rcfg_type = ['Mid-block','At an intersection','Intersection with parking lot entrance/exit',
            'Railroad crossing','Bridge','Tunnel','Passing or climbing lane',
             'Ramp','Traffic circle','Highway express lane',
             'Highway collector lane','Highway transfer lane']

df2 = df_b.groupby(['c_rcfg','c_sev']).size().unstack().fillna(0)
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

#Plot the new dataframe in logarithm, since the gap between numbers are huge.

df2.plot(kind='bar',rot=30,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity by  Road Configuration', ax=ax1)
ax1.grid(axis='both')
ax1.legend(['1.fatal','2.non-fatal'])
ax1.set_xticklabels(rcfg_type,ha='right')
ax1.set_xlabel('Road Configuration')

#Calculate the percentage of fatal injury in all accidents by different road surface type,
# and plot to right axis.
se1 = df2[1]/(df2[1]+df2[2])*100
se1.plot(c='b', style='o--', ax=ax2)

plt.xticks(np.arange(0,10),rcfg_type); 
#Collisions severity  by Weather Condition  (c_wthr field)
wthr_type = ['Clear and sunny','Overcast','Raining','Snowing','Freezing rain, hail','Fog, smog, mist',
            'Strong wind']

df3 = df_b.groupby(['c_wthr','c_sev']).size().unstack().fillna(0)
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

#Plot the new dataframe in logarithm, since the gap between numbers are huge.

df3.plot(kind='bar',rot=30,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity by Weather Condition', ax=ax1)
ax1.grid(axis='both')
ax1.legend(['1.fatal','2.non-fatal'])
ax1.set_xticklabels(wthr_type,ha='right')
ax1.set_xlabel('Weather Condition')

#Calculate the percentage of fatal injury in all accidents by different road surface type,
# and plot to right axis.
se1 = df3[1]/(df3[1]+df3[2])*100
se1.plot(c='b', style='o--', ax=ax2)

plt.xticks(np.arange(0,7),wthr_type); 
#Collision severity grouped by Road Surface (c_rsur field)
rsur_type = ['Dry, normal','Wet','Snow','Slush ,wet snow','Icy','Sand/gravel/dirt','Muddy','Oil','Flooded']

df4 = df_b.groupby(['c_rsur','c_sev']).size().unstack().fillna(0)
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

#Plot the new dataframe in logarithm, since the gap between numbers are huge.

df4.plot(kind='bar',rot=30,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity by  Road Surface', ax=ax1)
ax1.grid(axis='both')
ax1.legend(['1.fatal','2.non-fatal'])
ax1.set_xticklabels(rsur_type,ha='right')
ax1.set_xlabel('Road Surface')

#Calculate the percentage of fatal injury in all accidents by different road surface type,
# and plot to right axis.
se1 = df4[1]/(df4[1]+df4[2])*100
se1.plot(c='b', style='o--', ax=ax2)

plt.xticks(np.arange(0,9),rsur_type); 
#Collisions severity  by Traffic Control (c_traf field)
traf_type = ['Traffic signals fully operational','Traffic signals in flashing mode',
             'Stop sign','Yield sign','Warning sign','Pedestrian crosswalk',
             'Police officer','School guard','School crossing','Reduced speed zone',
             'No passing zone sign','Markings on the road',
             'School bus stopped,signal lights flashing','Railway crossing with signals and gates',
             'Railway crossing with signs only','Control device not specified','No control present']

df5 = df_b.groupby(['c_traf','c_sev']).size().unstack().fillna(0)
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

#Plot the new dataframe in logarithm, since the gap between numbers are huge.

df5.plot(kind='bar',rot=30,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity by Traffic Control', ax=ax1)
ax1.grid(axis='both')
ax1.legend(['1.fatal','2.non-fatal'])
ax1.set_xticklabels(traf_type,ha='right')
ax1.set_xlabel('Traffic Control')

#Calculate the percentage of fatal injury in all accidents by different road surface type,
# and plot to right axis.
se1 = df5[1]/(df5[1]+df5[2])*100
se1.plot(c='b', style='o--', ax=ax2)

plt.xticks(np.arange(0,17),traf_type); 
#Slice the columns I would analyse from original dataset.
df_a = df.loc[:,['date','c_year','c_mnth','c_wday','c_hour',
                'c_sev','c_vehs','c_conf','c_wthr','c_raln',
                'v_type','v_year']]

#Check how many unknown in road alignment.
df_a['c_raln'].value_counts().sort_index()
#Make a slice only focusing on c_raln: road alignment.
df1 = df_a.loc[:,['date','c_raln']]

#Aggregate the events by month into a summarized dataframe.
df2 = df1.groupby(['date','c_raln']).size().unstack()
df2
#Plot the dataframe into graph to visualize the result.
c_raln=['1.Level straight','2.Gradient straight',
        '3.Level curved','4.Gradient curved',
        '5.Top hill','6.Bottom hill',
        'Q.Other','U.Unknown']

df2.iloc[:,:-2].plot(logy=True, style = 'o', figsize = (18,6),
                     title = 'Collisions with Road Alignment')
plt.legend(c_raln)
plt.xlabel('Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');

df3=df1.groupby('c_raln').size().sort_index(ascending=False)

df3.plot.barh(log=True,color='slateblue',figsize=(15,5),
              title='Collisions by Road Alignment')
plt.xlabel('Number of Collisions')
plt.yticks(np.arange(0,8),c_raln[::-1])
plt.ylabel('Road Alignment')
plt.grid(axis='x');

#Use a new dataframe to analyze the relationship between Severity and Road Alignment.
#For c_sev, 1 is fatal, and 2 is non-fatal.

df1 = df_a.loc[:,['c_raln','c_sev']]
df4 = df1.groupby(['c_raln','c_sev']).size().unstack()
#Plot the relationship between Severity and Road Alignment.

#Create a twin y-axis plot area, and define axis parameters.
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

#Plot the new dataframe in logarithm, since the gap between numbers are huge.
df4.plot(kind='bar',rot=0,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity on Road Alignment', ax=ax1)
ax1.set_ylabel('Number of Collisions')
ax1.set_xlabel('Road Alignment')
ax1.grid(axis='y')
ax1.legend(['1.Fatal','2.Non-fatal'])

#Calculate the percentage of fatal injury in all accidents by
# different road alignment, and plot to right axis.
se1 = df4[1]/(df4[1]+df4[2])*100
se1.plot(c='b', style='o--', ax=ax2)
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

plt.xticks(np.arange(0,8),c_raln);

df1 = df_a.loc[:,['c_raln','c_vehs']]
df1.c_vehs.replace('UU',np.nan, inplace=True)
df1.dropna(inplace=True)
df1.c_vehs=df1.c_vehs.astype(int)
df1.c_vehs.value_counts().sort_index()

#Calculate accident portion of each event, considering 58 records that have 57 vehicle involved,
# that should be just one accident, every relevant record should be only about 1/57.
df1['acc']= 1/df1.c_vehs

#Categorize number of involved vehicles into bins.
bins = [0,5,10,20,40,60]
df1['cats'] = pd.cut(df1.c_vehs,bins)

#Summarize the collisions on road alignment and number of involved vehicles.
df5 = df1.groupby(['cats','c_raln'])['acc'].sum().unstack()
df5.dropna(how='all', inplace=True)
df5.fillna(0, inplace=True)
df5
#Plot the result.

df5.plot(logy=True,figsize=(15,5),style='o--',
         title='Number of Involved Vehicles and Road Alignment')
plt.xticks(np.arange(-0.5,5.5),bins)
plt.xlabel('Number of Vehicles Involved in Each Accident')
plt.ylabel('Number of Accidents')
plt.legend(c_raln)
plt.grid(axis='y');
#Prepare the dataset for analyse, replace non-numeric value into number to use in scatter plot.
df1 = df_a.loc[:,['c_raln','c_conf']]
df1.c_raln = df1.c_raln.replace({'Q':7,'U':8}).astype(int)
df1.c_conf = df1.c_conf.replace({'QQ':42,'UU':43,'XX':43}).astype(int)
#Too many collision configuration involved, iterate a list of dateframes to
# separate different collision config.
df6 = []
df6.append( df1[df1.c_conf < 10] )
df6.append( df1[(df1.c_conf > 20)&(df1.c_conf < 30)] )
df6.append( df1[(df1.c_conf > 30)&(df1.c_conf < 40)] )
df6.append( df1[df1.c_conf > 40] )

#Reformat the dataframe, to summarize the collision numbers in different situation,
# and store in another list of dataframe 'df7'

se6 = []; df7 = []
for i in range(0,4):
    se6.append( df6[i].groupby(['c_raln','c_conf']).size() )
    se6[i].name = 'collision'
    df7.append( pd.DataFrame(se6[i]).reset_index() )

df7[3]
#Draw four subplots to show the relation between the road alignment
# and different collision situations.

fig = plt.figure(figsize=(15, 10))
fig.suptitle("The Relation between Road-alignment and Collision-configuration",
             fontsize = 16)

#---------------------- Only one car involved in collision. ---------------------- 
ax1 = fig.add_subplot(2,2,1)
ax1.set_xticks(range(1,9))
ax1.set_xticklabels([1,2,3,4,5,6,'Q','U'])
ax1.set_xlabel("Road Alignment")

ax1.set_yticks(range(1,7))
ax1.set_yticklabels(['Person/animal','Station/tree','Left-roll',
                     'Right-roll','Rollover','Other'])
ax1.set_ylabel("Collision Configure")

ax1.set_title("Single Vehicle in Motion")
ax1.scatter(df7[0].c_raln,df7[0].c_conf,
            df7[0].collision*.01,
            alpha=0.5,color='r')
plt.grid(which='major')

#---------------------- Two car same direction. ---------------------- 
ax2 = fig.add_subplot(2,2,2)
ax2.set_xticks(range(1,9))
ax2.set_xticklabels([1,2,3,4,5,6,'Q','U'])
ax2.set_xlabel("Road Alignment")


ax2.set_ylim([19.5,25.5])
ax2.set_yticks(range(21,26))
ax2.set_yticklabels(['Rear-end','Side-swipe','Left-turn','Right-turn','Other'])
ax2.set_ylabel("Collision Configure")

ax2.set_title("Two Vehicle Same Direction")
ax2.scatter(df7[1].c_raln,df7[1].c_conf,
            df7[1].collision*.01,
            alpha=0.5,color='r')
plt.grid(which='major')

#---------------------- Two car different direction. ---------------------- 
ax3 = fig.add_subplot(2,2,3)
ax3.set_xticks(range(1,9))
ax3.set_xticklabels(c_raln,rotation=30,ha='right')
ax3.set_xlabel("Road Alignment")

ax3.set_ylim([30.5,37])
ax3.set_yticks(range(31,37))
ax3.set_yticklabels(['Head-on','Side-swipe','Left-turn','Right-turn',
                     'Right-angle','Other'])
ax3.set_ylabel("Collision Configure")

ax3.set_title("Two Vehicle Different Direction")
ax3.scatter(df7[2].c_raln,df7[2].c_conf,
            df7[2].collision*.01,
            alpha=0.5,color='r')
plt.grid(which='major')

#---------------------- Other situation. ---------------------- 
ax4 = fig.add_subplot(2,2,4)
ax4.set_xticks(range(1,9))
ax4.set_xticklabels(c_raln,rotation=30,ha='right')
ax4.set_xlabel("Road Alignment")

ax4.set_ylim([40.7,43.3])
ax4.set_yticks(range(41,44))
ax4.set_yticklabels(['Parked car','Other','Unknown'])
ax4.set_ylabel("Collision Configure")

ax4.set_title("Other Situation")
ax4.scatter(df7[3].c_raln,df7[3].c_conf,
            df7[3].collision*.01,
            alpha=0.5,color='r')
plt.grid(which='major');
df1 = df_a.loc[:,['c_mnth','v_type']]

#Aggregate the events by month into a summarized dataframe.
df2 = df1.groupby(['c_mnth','v_type']).size().unstack()
df2.index = df2.index.astype(int)
df2.fillna(0, inplace=True)

v_type = ['1.Light Duty', '5.Cargo <4.5t', '6.Truck <4.5t',
          '7.Truck >4.5t', '8.Road tractor', '9.School bus',
          '10.Small school bus', '11.Urban bus', '14.Motorcycle',
          '16.Off-road', '17.Bicycle', '18.Motorhome', '19.Farm equip.',
          '20.Constru. equip.', '21.Fire engine', '22.Snowmobile',
          '23.Street car', 'N.Not vehicle', 'Q.Others', 'U.Unknown']
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
df2.iloc[:,[0,1,2,3,4,7]].plot(logy=True, figsize = (15,3), style = 'o--',
                               title = 'Urban Vehicle Collisions')
plt.legend(v_type[0:5]+[v_type[7]])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
df2.iloc[:,17:20].plot(logy=True, figsize = (15,3), style = 'o--',
                       title = 'Special Collisions')
plt.legend(v_type[17:20])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
df2.iloc[:,5:7].plot(logy=True, figsize = (15,3), style = 'o--',
                     title = 'School Bus Collisions')
plt.legend(v_type[5:7])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
df2.iloc[:,8:12].plot(logy=True, figsize = (15,3), style = 'o--',
                      title = 'Recreation Vehicle Collisions')
plt.legend(v_type[8:12])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
df2.iloc[:,[13,15]].plot(logy=True, figsize = (15,3), style = 'o--',
                         title = 'Operation Vehicle Collisions')
plt.legend([v_type[13],v_type[15]])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
df2.iloc[:,[12,14,16]].plot(logy=True, figsize = (15,3), style = 'o--',
                            title = 'City Vehicle Collisions')
plt.legend([v_type[12],v_type[14],v_type[16]])
plt.xticks(range(1,13),months)
plt.xlabel('Month of The Year')
plt.ylabel('Number of Collisions')
plt.grid(axis='y');
#Plot the total collisions by different vehicle types (top 10):

df3 = pd.DataFrame(df1.groupby('v_type').size().sort_index())
df3['types'] = v_type
df3 = df3.rename(columns={0:'collisions'}).set_index('types').sort_values(by='collisions')

df3[-10:].plot.barh(log=True,color='slateblue',figsize=(15,5),
                    title='Collisions by Vehicle Type',legend=None)
plt.xlabel('Number of Collisions')
plt.ylabel('Vehicle Type')
plt.grid(axis='x');
#Use a new dataframe to analyze the relationship between Severity and Road Alignment.
#For c_sev, 1 is fatal, and 2 is non-fatal.

df1 = df_a.loc[:,['v_type','c_sev']]
df4 = df1.groupby(['v_type','c_sev']).size().unstack().fillna(0)

#Plot the relationship between Severity and Vehicle Type.

#Create a twin y-axis plot area, and define axis parameters.
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

#Plot the new dataframe in logarithm, since the gap between numbers are huge.
df4.plot(kind='bar',rot=30,log=True,color=['red','coral'],figsize=(15,5),
         title='Severity on Vehicle Type', ax=ax1)
ax1.set_xticklabels(v_type,ha='right')
ax1.set_xlabel('Vehicle Type')
ax1.set_ylabel('Number of Collisions')
ax1.grid(axis='both')
ax1.legend(['1.Fatal','2.Non-fatal'])

#Calculate the percentage of fatal injury in all accidents by different road alignment,
# and plot to right axis.
se1 = df4[1]/(df4[1]+df4[2])*100
se1.plot(c='b', style='o--', ax=ax2)
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')

plt.xticks(np.arange(0,20),v_type);

#Prepare the dataset and clean it.
df1 = df_a.loc[:,['c_year','v_year']]

df1.v_year.replace({'NNNN':NA,'UUUU':NA,'XXXX':NA},inplace=True)
df1.dropna(inplace=True)
df1 = df1.astype(int)

#Aggregate the collisions by year into a summarized dataframe (model year as index).
df2 = df1.groupby(['v_year','c_year']).size().unstack()
df2.fillna(0, inplace=True)
df2 = df2.astype(int)
df2
#Number of collisions grouped by vehicle model year range (12 ranges as columns)
bins = list(range(1900,2021,10))
df2['cats'] = pd.cut(df2.index,bins,labels=bins[:-1])
df3 = df2.groupby('cats').sum().T
df3
#Plot result of most involved model years:
se4=df3.sum().sort_values()
se4[-5:].plot.barh(log=True,color='slateblue',figsize=(15,5),
                   title='Collisions by Vehicle Model Year')
plt.xlabel('Number of Collisions')
plt.grid(axis='x');
#Plot the trending of model year involved in collision.
df3.iloc[:,7:].plot(logy=True,style='o--',figsize=(15,5),
                    title='Vehicle Model Year Involved in Collisions Over Time')
plt.xlabel('Collision Year')
plt.ylabel('Number of Collisions')
plt.legend(title='Model Year')
plt.grid(axis='y');
#Generating working data set 'df_d' by keeping the using elements
df_d = df.loc[:,['c_hour','c_vehs','c_wday',
                 'c_conf','p_sex','p_age','p_isev','p_user']]
df_d
# define 0 as men; 1 as women
df_d.p_sex.replace({'M':0,'F':1},inplace=True)
# Clean the missing variable by droping the row which contains any missing variables
df_d.replace('[^0-9]+',np.nan,regex=True,inplace=True)
df_d.dropna(inplace=True)
num = df_d.groupby(['p_sex']).count()
print(num)
df1 = df_d.loc[:,['p_sex','c_hour']].groupby(['c_hour','p_sex']).size().unstack()
df1
df1.plot(rot=0,color=['g','r'],figsize=(15,5),
         title='Person sex & Collision hour')
plt.legend(['0.Male','1.Female'])
plt.xticks(range(0,24),range(0,24))
plt.grid();
#8 o'clock:
df1.iloc[8,]
df2 = df_d.loc[:,['p_sex','p_isev']].groupby(['p_sex','p_isev']).size().unstack()
df2
df2.plot(kind='bar',rot=0, color=['g','orange','r'], figsize=(8,5),
         title='Person sex & Medical treatment required')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,2),['0.Male','1.Female'])
plt.grid(axis='y');
#Same result in logarithm scale.
df2.plot(kind='bar',rot=0, color=['g','orange','r'], figsize=(8,5),log= True,
         title='Person sex & Medical treatment required (in log)')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,2),['0.Male','1.Female'])
plt.grid(axis='y');
#Prepare sub-dataset
df3 = df_d.loc[:,['p_sex','c_conf']].groupby(['c_conf','p_sex']).size().unstack()
df3
#Prepare labels for x-axis
x_label=['01.SV Hit movible','02.SV Hit stationary','03.SV Ran off left shoulder',
         '04.SV Ran off right shoulder','05.SV Rollover on roadway',
         '06.SV Others','21.2V SD Rear','22.2V SD side-swipe',
         '23.2V SD Lf','24.2V SD Rt','25.2V Others','31.2V DD Head',
         '32.2V DD side-swipe','33.2V DD Lf','34.2V DD Rt','35.2V DD Right angle',
         '36.2V DD other','41.Hit parked vehicle']
len(x_label)
df3.plot(style = 'o--', figsize = (15,5),
         title='Person sex & Type of accident')
plt.legend(['Male','Female'])
plt.xticks(range(0,18),x_label,rotation=30,ha='right');
plt.grid();
#Prepare sub-dataset, manage NA value
df4 = df_d.loc[:,['p_age','c_hour']].astype(int).groupby(['p_age','c_hour']).size().unstack()
df4.fillna(0, inplace=True)
#Bin ages into 10 buckets.
bins = list(range(0,101,10))
df4['age'] = pd.cut(df4.index,bins,labels=bins[:-1])
df5 = df4.groupby('age').sum()
df5 = df5.T
df5
df5.plot(rot=0,figsize=(15,5), title='Person age & Collision hour')
#plt.legend(['0.Male','1.Female'])
plt.xticks(range(0,24),range(0,24))
plt.grid();
df6 = df_d.loc[:,['p_age','c_wday']].astype(int).groupby(['c_wday','p_age']).size().unstack().T
df6
bins = list(range(0,101,10))
df6['age'] = pd.cut(df6.index,bins,labels=bins[:-1])
df7 = df6.groupby('age').sum()
df7 = df7.T
df7
df7.plot(rot=0,figsize=(8,5),title='Person age & Days of the week',style='o--')
plt.xticks(range(1,8),['Mon','Tue','Wed','Thu','Fri','Sat','Sun']);
#plt.grid(axis='y');
df8 = df_d.loc[:,['p_age','p_isev']].astype(int).groupby(['p_age','p_isev']).size().unstack()
df8.fillna(0, inplace=True)
df8
bins = list(range(0,101,10))
df8['age'] = pd.cut(df8.index,bins,labels=bins[:-1])
df9 = df8.groupby('age').sum()
df9 
df9.plot(kind='bar',rot=0, color=['g','orange','r'],figsize=(15,5),
         title='Person age & Medical treatment required')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,11),['(0, 10]',' (10, 20]',' (20, 30]',' (30, 40]',
                        ' (40, 50]',' (50, 60]',' (60, 70]',' (70, 80]',
                        ' (80, 90]',' (90, 100]'])
plt.grid(axis='y');
df9.plot(kind='bar',rot=0, color=['g','orange','r'],figsize=(15,5),log= True,
         title='Person age & Medical treatment required (in log)')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,11),['(0, 10]',' (10, 20]',' (20, 30]',' (30, 40]',
                        ' (40, 50]',' (50, 60]',' (60, 70]',' (70, 80]',
                        ' (80, 90]',' (90, 100]'])
plt.grid(axis='y');
#Verifying result using(20,30] age group
df_d[(df_d.p_age.astype(int)> 20) & (df_d.p_age.astype(int)<= 30)].p_isev.value_counts()
#Verifying result using (50,60] age group
df_d[(df_d.p_age.astype(int)> 50) & (df_d.p_age.astype(int)<= 60)].p_isev.value_counts()
df10 = df_d.loc[:,['p_user','p_isev']].astype(int).groupby(['p_user','p_isev']).size().unstack()
df10.fillna(0, inplace=True)
df10
df10.plot(kind='bar',rot=0,color=['g','orange','r'],figsize=(8,5),
          title='Road user class & Medical treatment required ')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,5),['Driver','Passenger','Pedestrian',
                       'Bicyclist', 'Motorcyclist'])
plt.grid(axis='y');
df10.plot(kind='bar',rot=0,color=['g','orange','r'],figsize=(8,5),
          log = True, title='Road user class & Medical treatment required (in log)')
plt.legend(['1.No Injury','2.Injury','3.Fatality'])
plt.xticks(range(0,5),['Driver','Passenger','Pedestrian',
                       'Bicyclist', 'Motorcyclist'])
plt.grid(axis='y');
df11 = df_d.loc[:,['p_user','c_conf']].groupby(['c_conf','p_user']).size().unstack()
df11.fillna(0,inplace=True)
df11
df11.plot(style = 'o--', logy = True, figsize = (15,5),
          title='Road user class & Collision configration' )
plt.legend(['Diver','Passenger','Pedestrian',
            'Bicyclist', 'Motorcyclist'])
plt.xticks(range(0,18),x_label,rotation=30,ha='right')
plt.grid();
#Prepare dataset 'df_c'
df_c = dfp.loc[:,['date','c_year','c_mnth','c_wday',
                  'c_hour','c_sev','c_vehs','c_conf','p_psn','p_safe']]
df_c[['c_sev','c_vehs','c_conf','p_psn','p_safe']] =\
        df_c[['c_sev','c_vehs','c_conf','p_psn','p_safe']].replace('^([A-Za-z])+$', np.nan, regex=True)
df_c = df_c.dropna()

#This process is to eliminate the '0' preceding a single digit number. ex. 01 --> 1, converting str objects to int
df_c[['c_vehs','c_conf','p_safe']] = df_c[['c_vehs','c_conf','p_safe']].fillna(-1).astype(int)
df_c
#Number of vehicles invovled in collision over time
df1 = df_c.loc[:,['date','c_vehs']]
#Total number of vehicles involved in collision 
df2 = df1.groupby('c_vehs').size().sort_index()
df2.plot(style='x',logy=True, figsize=(10,6),
         title='Total number of vehicles involved in Collision')
plt.grid(axis='x');
plt.xlabel('Number of vehicles involved in each collision')
plt.ylabel('Number of collisions');
#Visualize the result
vehs = df1.groupby(['date','c_vehs']).size().unstack()
vehs.plot(style = 'o', figsize=(18,9),
                  title = 'Number of vehicles involved in Collision')
plt.grid(axis='y')
plt.xlabel('Date')
plt.legend(ncol=3)
plt.ylabel('Number of collisions'); 
#Selected c_vehs 1-5 to view the top 5 most frequent number of vehicles involved in collisions 
a = vehs.iloc[:,0:5]
a.plot(style='o--',figsize=(12,6),
       title='Number of vehicles involved in collision over time')
plt.xlabel('Date')
plt.ylabel('Number of collisions');
#Total number of vehicles involved in collision in regards to weekday
df3 = df_c.loc[:,['c_wday','c_vehs']]
weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
w_day = df3.groupby('c_wday')['c_vehs'].size()

#Visualize the result
w_day.plot(kind='barh', color='c',figsize=(10,6),rot =0,
           title='Total Number of vehicles involved in collision by Weekday')
plt.yticks(range(0,8),weekday)
plt.ylabel('Weekday');
#Relationship of number of vehicles involved in collision (1-57 vehicles) on a specific weekday 
w_day1 = df3.groupby(['c_wday','c_vehs']).size().unstack()

w_day1.plot(style = 'o--', figsize=(16,7),
            title = 'Number of vehicles involved in Collision')
plt.legend(ncol=3,loc=7)
plt.xticks(range(0,8),weekday)
plt.xlabel('Week day')
plt.ylabel('Number of collisions');
b = w_day1.iloc[:,0:5]
b.plot(figsize=(7,4),style='o--')
plt.xlabel('Weekday')
plt.ylabel('Number of collisions')
plt.xticks(range(0,8),weekday);
#Slice the data with just Number of vehicles involved in collision & hour
df4 = df_c.loc[:,['c_hour','c_vehs']]
df4
#Total Number of vehicles involved in collision by hour
hour = df4.groupby('c_hour')['c_vehs'].size()
hour.plot(kind='bar',color='G',rot=0,
          title='Total Number of vehicles involved in collision by hour')
plt.xlabel('Hour')
plt.ylabel('Number of collisions');
#Number of vehicles involved in collisions at different hours
hour1 = df4.groupby(['c_hour','c_vehs']).size().unstack()
hour1.plot(figsize=(16,8),style='o-',alpha=0.9)
plt.xticks(range(0,24),range(0,24))
plt.xlabel('Hour')
plt.ylabel('Number of collisions')
plt.legend(ncol=3);
#Slice the data into only visualizing c_vehs=1-5 
c = hour1.iloc[:,0:5]
c.plot(style='o--',figsize=(7,4))
plt.xlabel('Hour')
plt.ylabel('Number of collisions')
plt.xticks(range(0,24),range(0,24));
#Relationship between number of vehicles involved in collision and Collision severity

df5 = df_c.loc[:,['c_vehs','c_sev']]
vehs_sev = df5.groupby(['c_vehs','c_sev']).size().unstack()
vehs_sev.index = vehs_sev.index.astype(str)
vehs_sev
#Plot the new dataframe in logarithm, since the gap between numbers are huge.
vehs_sev.plot(logy=True,style = 'o', figsize=(15,5),
              title = 'Number of vehicles involved in Collision & Collision Severity')
plt.grid(axis='y')
plt.legend(['1.fatal','2.non-fatal'])
plt.xlabel('Number of vehicles involved in Collision') 
plt.ylabel('Number of collisions');
#Total Number of vehicles involved in collision & Collision configuration
conf = ['1. Hit a moving object','2. Hit a stationary object',
        '3.Ran off left shoulder','4.Ran off Right shoulder','5.Rollover on a Roadway',
        '6.Other single vehicle collision', '21. Rear End collision', '22. Side Swipe',
        '23. 1 vehicle passing to the left of the other/left turn conflict',
        '24. 1 vehicle passing to the right of the other/or right turn conflict',
        '25. Other 2 vehicle-same direction of travel','31.Head-on collision',
        '32.Approaching side-swipe','33.Left turn across opposing traffic',
        '34.Right turn,including turning conflicts','35.Right angle collision',
        '36.Any other two-vehicle','41.Hit a parked motor vehicle']

vehs_conf = df_c.groupby('c_conf')['c_vehs'].count().sort_index(ascending=True)

vehs_conf.plot(figsize=(10,6), kind='barh', rot=0)
plt.xlabel('Total Number of vehicles involved in collision')
plt.ylabel('Collision Configuration')
plt.yticks(np.arange(0,18),conf);
#Number of vehicles involved in each collision configuration
df6 = df_c.loc[:,['c_vehs','c_conf']].groupby(['c_conf','c_vehs']).size().unstack().fillna(0)
df6.index = df6.index.astype(str)
df6
#Visualize the result
df6.plot(style = 'o-',figsize =(15,9),title='Number of Vehicles & Collision Configuration')
plt.xticks(np.arange(0,18),conf,rotation=30,ha='right')
plt.grid()
plt.xlabel('Collision configuration')
plt.ylabel('Number of collision')
plt.legend(ncol=3);
#Select 1-4 number of vehicles involved in collision to
# visualize its relationship with collision configuration
d = df6.iloc[:,0:4]
d.plot(style='o--')
plt.xlabel('Collision configuration')
plt.xticks(np.arange(0,18),conf,rotation=50,ha='right');
#Analyze the relationship between collision configuration over date
df7 = df_c.loc[:,['date','c_conf']]
configuration = df7.groupby(['date','c_conf']).size().unstack()
configuration
configuration.plot(style='',figsize=(15,7))
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Number of collision');
#Select top 5 frequest configuration collision and plot the data
confi = ['1. Hit a moving object','6.Other single vehicle collision',
         '21. Rear End collision','31.Head-on collision',
         '33.Left turn across opposing traffic',
         '35.Right angle collision','36.Any other two-vehicle']
e = configuration[[1,6,21,31,33,35,36]]
e.plot(figsize=(12,6),style='o--',alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Number of Collision')
plt.legend(confi);
#Analyze the relationship between collision configuration and collision severity
df8 = df_c.loc[:,['c_conf','c_sev']]
conf_sev = df8.groupby(['c_conf','c_sev']).size().unstack()
conf_sev.index = conf_sev.index.astype(str)
conf_sev
#Visualize the results
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

conf_sev.plot(kind='bar',rot=30,log=True,style='o--',figsize=(12,6),ax=ax1)
ax1.set_ylabel('Number of Collisions')
ax1.set_xlabel('Collision configuration')
ax1.set_xticklabels(conf,ha='right')
ax1.grid(axis='y')
ax1.legend(['1.Fatal','2.Non-fatal'])

se1 = conf_sev[1]/(conf_sev[1]+conf_sev[2])*100
se1.plot(c='b', style='o--', ax=ax2)
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')
plt.xticks(np.arange(0,18),conf,rotation=50,ha='right')
plt.xlim(-0.5,17.5);
#Analyze the relationship between person position and collision severity
position = ['11.Driver','12. Front row, center','13.Front row: right outboard',
            '21.Second row:left outboard','22.Second row:center',
            '23.Second row:right outboard','31.Third row:left outboard',
            '32.Third row:center','33.Third row:right outboard',
            '96.Unknown occupant','97.Sitting on someone’s lap',
            '98.Outside passenger compartment','99.Pedestrian']
df9 = df_c.loc[:,['p_psn','c_sev']]
psn_sev = df9.groupby(['p_psn','c_sev']).size().unstack()
psn_sev
#Visualize the result
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

psn_sev.plot(kind='bar',rot=30,log=True,style='o--',figsize=(10,6),ax=ax1)
ax1.set_ylabel('Number of Collisions')
ax1.set_xlabel('Person Position')
ax1.set_xticklabels(position,ha='right')
ax1.grid(axis='y')
ax1.legend(['1.Fatal','2.Non-fatal'])

se1 = psn_sev[1]/(psn_sev[1]+psn_sev[2])*100
se1.plot(c='b', style='o--', ax=ax2)
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')
plt.xticks(np.arange(0,13),position,rotation=40,ha='right')
plt.xlim(-0.5,12.5);
#Relationship between Person position and collision configuration
df10 = df_c.loc[:,['p_psn','c_conf']]
psn_conf = df10.groupby(['c_conf','p_psn']).size().unstack().fillna(0)
psn_conf.index = psn_conf.index.astype(str)
psn_conf
#Visualize the result
psn = ['11.Driver','13.Front row: right outboard','21.Second row: left outboard',
       '23.Second row:right outboard','99.Pedestrian']
f = psn_conf.iloc[:,[0,2,3,5,12]]
f.plot(logy=True,style='o--',figsize=(12,6))
plt.legend(psn)
plt.xlabel('Collision configuration')
plt.xticks(np.arange(0,18),conf,rotation=40,ha='right');
#Relationship between safety device used over time
df11 = df_c.loc[:,['date','p_safe']]
saf_date = df11.groupby(['date','p_safe']).size().unstack()
saf_date
#Visualize the result
#13. No safety device equipped => eg. buses
safety = ['1.No safety device used','2.Safety device used','9.Helmet',
          '10.Reflective clothing','11. Helmet & reflective clothing',
          '12.Other safey device','13.No safety device equipped']
saf_date.plot(logy=True, style='o--',figsize=(10,6))
plt.legend(safety, loc=4)
plt.ylabel('Number of collision')
plt.xlabel('Date');
#Relationship between safety device used and collision severity
df12 = df_c.loc[:,['p_safe','c_sev']]
saf_sev = df12.groupby(['p_safe','c_sev']).size().unstack().fillna(0)
saf_sev.index = saf_sev.index.astype(str)
saf_sev
#Visualize the result
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()

saf_sev.plot(kind='bar',rot=30,log=True,style='o--',figsize=(10,6),ax=ax1)
ax1.set_ylabel('Number of Collisions')
ax1.set_xlabel('Saftey device used')
ax1.set_xticklabels(safety,ha='right')
ax1.grid(axis='y')
ax1.legend(['1.Fatal','2.Non-fatal'])

se1 = saf_sev[1]/(saf_sev[1]+saf_sev[2])*100
se1.plot(c='b', style='o--', ax=ax2)
ax2.set_ylabel('Fatal Percentage (%)')
ax2.spines['right'].set_color('b')
ax2.yaxis.label.set_color('b')
ax2.tick_params(axis='y',colors='b')
plt.xticks(range(0,7),safety)
plt.xlim(-0.5,6.5);