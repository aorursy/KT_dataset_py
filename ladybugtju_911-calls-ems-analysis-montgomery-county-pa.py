import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.head(5)
# Our dataset counts 99,492 calls and presents the following information for each of them 

df.info()
# We count 110 general case types

df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
# We report 3 main categories 

df['Reason'].value_counts()
# Pie chart

# EMS cases represent 49.1% of the total cases to study, followed by 35.9% of traffic and 15% of fire

labels = 'EMS', 'Traffic', 'Fire'

sizes = df['Reason'].value_counts()

explode = (0.1, 0, 0)  



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
EMS = pd.DataFrame(df[df['Reason'] == 'EMS'])

EMS.head(3)
# Looking at emergency types

EMS['EMSCases'] = EMS['title'].apply(lambda title: title.split(':')[1])

EMS.head(3)
# Type count

# We report 68 emergency types

EMS['EMSCases'] .nunique()
# General overview 

sns.countplot(x='EMSCases',data=EMS,palette='viridis')
# Most common cases plotting 

EMS['EMSCases'].value_counts()[:20].plot(kind='barh')
# Top 5 EMS cases

EMS['EMSCases'].value_counts()[:5]
# Creating a new dataframe for deeper analysis of the top 5 EMS cases

Top5 =pd.DataFrame(EMS[(EMS['EMSCases']== ' RESPIRATORY EMERGENCY')|(EMS['EMSCases']== ' CARDIAC EMERGENCY') | (EMS['EMSCases']== ' FALL VICTIM')| (EMS['EMSCases']== ' VEHICLE ACCIDENT') | (EMS['EMSCases']== ' SUBJECT IN PAIN')])

Top5.set_index('EMSCases',inplace=True)

Top5.drop(labels=['e','Reason'],axis=1,inplace=True)

Top5.head(3)
# Tranforming timeStamp Data 

Top5['timeStamp'] = pd.to_datetime(Top5['timeStamp'])
Top5['Year'] = Top5['timeStamp'].apply(lambda time: time.year)

Top5['Month'] = Top5['timeStamp'].apply(lambda time: time.month)

Top5['Day of Week'] = Top5['timeStamp'].apply(lambda time: time.dayofweek)

Top5['Hour'] = Top5['timeStamp'].apply(lambda time: time.hour)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
Top5['Day of Week'] = Top5['Day of Week'].map(dmap)
# Ordering our data by timeStamp

Top5.sort_values(['timeStamp'],ascending=[1])

Top5.head(3)
# TimeStamp Range : December 10, 2015 - August 24, 2016

Top5['timeStamp'][[0,-1]] 
# Comfirming min

Top5['timeStamp'].min()
# Plotting cases by year

# 2015 Data seems not to be introducing significant bias for general plotting 

sns.countplot(x='Year',data=Top5,hue='title',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Sharp decrease of cases by February

Top5.groupby('Month').count()['title'].plot()
# General decrease in February for all EMS Cases with considerable decline in FALL VICTIMS 

# This could also be interpreted as an increase in January since subsequent months keep steady at February level with a light increase in July 

sns.countplot(x='Month',data=Top5,hue='title',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# The general plotting shows less calls on the weekends

Top5.groupby('Day of Week').count()['title'].plot()
# A general decline of vehicle accidents calls during the weekends

# Respiratory emergency cases seem to be at the same level during the week with a peak on Mondays and Tuesdays

# Cardiac emergency cases are the highest with a remarkable decline on weekends

sns.countplot(x='Day of Week',data=Top5,hue='title',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Cases peak hours are from 10am to 3pm

# We notice sharp decrease from 3pm to 3am and a reverse increase from 5am to 10am

Top5.groupby('Hour').count()['title'].plot()
# Respiratory emergencies seem to peak from 4pm to 7pm

# Hightest Cardiac Cases are noticed from 8am to 3pm

# Fall victim cases are fairly noticeable and follow general trends with a sharp increase 

#      from 7am to 11am reporting counts up to 300 calls followed by a slow decrease

# Vehicle accidents peak at 2 and 3pm to surpass 250 calls. 

#   Highest rates seem to follow traffic trends and general congestion from 8am to 9pm. 

 

sns.countplot(x='Hour',data=Top5,hue='title',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Analysing the busiest towns

# The hights towns exceed 1000 cases for our study period 

Top5['twp'].value_counts().head(5)
# A big different between our population and the top five towns calls case count

[Top5['twp'].value_counts().mean(), Top5['twp'].value_counts().head(5).mean()]
twp =pd.DataFrame(Top5[(Top5['twp']== 'NORRISTOWN')|(Top5['twp']== 'LOWER MERION') | (Top5['twp']== 'ABINGTON')| (Top5['twp']== 'POTTSTOWN') | (Top5['twp']== 'UPPER MERION')])

twp = twp.sort_values(['twp', 'Year', 'Month'],ascending=[1, 0 ,0])

twp.head(3)
# Call regarding subjets in pain are among the lowest in our selection except for Norristown and Pottstown

# Respiratory emergencies are the most reported with a disproportionately high level in Norristown

# Cardiac cases seem to be low in Upper Merion. This trait seems to hold while comparing data againts other towns or EMS case types in the same region

sns.countplot(y='twp',data=twp,hue='title',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Tuesday seems to the busiest day of the week

# Monday and Tuesday between 8am to 2pm are calls peak times 

# Thursday and Friday seem to second the beginning of the week with busy hours from 10am to 5pm

# Weekends look mostly quite

dayHour = Top5.groupby(by=['Day of Week','Hour']).count()['title'].unstack()

plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
# A better viz with Clustermaps

sns.clustermap(dayHour,cmap='viridis')
#  Day of Week and Month analysis 

#  The busiest days of the year seem to be Fridays mainly in the month of July 

dayMonth = Top5.groupby(by=['Day of Week','Month']).count()['title'].unstack()

plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')
# Following July, January seems to be the second busiest month of the year with a peak on Friday

# In March lots of attention should be paid on Tuesdays and in June, on Thursdays

sns.clustermap(dayMonth,cmap='viridis')
# twp data confirms the conclusion that Fridays of July are the busiest days of the week

# Averages are milder than the main population one, leading us to expect some regions to have unexpected peaks

dayMonth = twp.groupby(by=['Day of Week','Month']).count()['title'].unstack()

plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')
# As noticed above, July and January seem to be fairly close with the highest count for January on Friday and unexpectedly Sunday  

# The month of June seem to fairly busy with sharp declines on Weekends

sns.clustermap(dayMonth,cmap='viridis')