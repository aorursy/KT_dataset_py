import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, date

import gc

%matplotlib inline

sns.set()
df = pd.read_csv('../input/vancouver-crime-report/crime_records.csv')

df.replace([np.inf,-np.inf],np.nan,inplace=True)

df.fillna(0,inplace=True)

df['HOUR'] = df['HOUR'].astype('int')

df['MINUTE'] = df['MINUTE'].astype('int')

df['date'] = df.apply(lambda row: datetime(year=row.YEAR,month=row.MONTH,day=row.DAY,hour=row.HOUR,minute=row.MINUTE),axis=1)

df['weekday'] = df['date'].apply(lambda x: x.weekday())

df = df[df.date<pd.Timestamp('2019-07-01')]

df.index = df['date']
def prepare_df(crime_type, df, percentage=True,years=[2011,2012,2013]):

    newdf1 = df[(df.TYPE==crime_type)&(df.YEAR.isin(years))]

    newdf2 = df[(df.TYPE==crime_type)&(df.YEAR.isin([2017,2018,2019]))]

    if percentage==True:

        newdf1 = (newdf1.groupby(['NEIGHBOURHOOD']).size()/len(newdf1)).sort_values(ascending=True).reset_index().rename(columns={0: 'years 2011, 2012 and 2013'})

        newdf2 = (newdf2.groupby(['NEIGHBOURHOOD']).size()/len(newdf2)).sort_values(ascending=True).reset_index().rename(columns={0: 'years 2017, 2018 and 2019'})

    else:

        newdf1 = (newdf1.groupby(['NEIGHBOURHOOD']).size()).sort_values(ascending=True).reset_index().rename(columns={0: 'years 2011, 2012 and 2013'})

        newdf2 = (newdf2.groupby(['NEIGHBOURHOOD']).size()).sort_values(ascending=True).reset_index().rename(columns={0: 'years 2017, 2018 and 2019'})  

    newdf = pd.merge(newdf2,newdf1,on=['NEIGHBOURHOOD'],how='left')

    newdf.index = newdf['NEIGHBOURHOOD']

    newdf.drop(['NEIGHBOURHOOD'],axis=1,inplace=True)

    return newdf 



def make_heatmap(df,color='black',crime_type='Other Theft', years=[2019],comments=''): 

    df_clean = df.copy()

    df_clean = df_clean[df_clean.TYPE==crime_type]

    df_clean = df_clean[df_clean.YEAR.isin(years)]

    weekdays = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

    heatmap = pd.pivot_table(df_clean[~((df_clean.HOUR==0)&(df_clean.MINUTE==0))], values='TYPE', index=['HOUR'],columns=['weekday'], aggfunc='count')

    heatmap.fillna(0,inplace=True)

    ylabels = [weekdays[i] for i in heatmap.columns.values]

    xlabels = heatmap.index.values

    values = heatmap.values.T



    fig, ax = plt.subplots(figsize=(15,5))

    ax.xaxis.tick_top()

    ax.tick_params(axis=u'both', which=u'both',length=0)

    im = ax.imshow(values,cmap='Reds',vmin=0)

    ax.set_xticks(np.arange(len(xlabels)))

    ax.set_yticks(np.arange(len(ylabels)))

    ax.set_xticklabels(xlabels)

    ax.set_yticklabels(ylabels)

    plt.setp(ax.get_xticklabels(), ha="right",

             rotation_mode="anchor")

    for i in range(len(ylabels)):

        for j in range(len(xlabels)):

            text = ax.text(j, i, int(values[i, j]),

                           ha="center", va="center", color="white")

    ax.grid(False)



    fig.tight_layout()

    text = ''

    for y in years:

        text += str(y)+", "

    plt.title(f'Total number of "{crime_type}" incidents in years: {text[:-2]} (weekday by hour of the day)',fontsize=12,color=color)

    plt.subplots_adjust(left=0.38)

    plt.gcf().text(0.995, 0.5, comments, fontsize=12)

    plt.show()

    del df_clean

    gc.collect()
crime_type = 'Theft from Vehicle'

plt.figure(figsize=(10,5))

plt.rcParams['axes.facecolor'] = 'white'

df[df.TYPE==crime_type].groupby(pd.Grouper(freq='BQ')).size().rolling(4).mean().plot(color='blue',alpha=0.5,linestyle='--', label='moving average 12 months')

df[df.TYPE==crime_type].groupby(pd.Grouper(freq='BQ')).size().plot(color='black',alpha=0.9, label = f'number "{crime_type}" of incidents')

plt.title(f'{crime_type} per Quarter',fontsize=14)

plt.legend(frameon=False)

plt.xlabel('')

plt.xlim(pd.Timestamp('2003-01-01'), pd.Timestamp('2019-06-30'))

plt.axvspan(pd.Timestamp('2011-01-01'), pd.Timestamp('2013-12-31'), facecolor='green', alpha=0.1)

plt.axvspan(pd.Timestamp('2017-01-01'), pd.Timestamp('2019-06-30'), facecolor='red', alpha=0.1)

plt.ylim(400)

textstr = f'"{crime_type}" numbers were decreasing, from 2003 to ~2012,'

textstr2 = 'from 2013 to 2019 there is a clear increase in the number of this'

textstr3 = 'type of crime. There is a seasonal influence, but observing '

textstr4 = 'the "moving average" curve it is clear the trend of increase.'

plt.gcf().text(0.92, 0.82, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.79, textstr2, fontsize=10.5)

plt.gcf().text(0.92, 0.76, textstr3, fontsize=10.5)

plt.gcf().text(0.92, 0.73, textstr4, fontsize=10.5)



textstr = 'In the graphs below there will be a comparison between the total'

textstr2 = 'number of occurrences in the timespan between'

textstr21 = '2011 to 2013'

textstr22 = 'and'

textstr3 = '2017 to 2019 (June 30).'

plt.gcf().text(0.92, 0.52, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.49, textstr2, fontsize=10.5)

plt.gcf().text(1.235, 0.49, textstr21, fontsize=10.5, color='green',alpha=0.8)

plt.gcf().text(1.325, 0.49, textstr22, fontsize=10.5)

plt.gcf().text(0.92, 0.46, textstr3, fontsize=10.5, color='red', alpha=0.8)

plt.show()
newdf = prepare_df(crime_type = crime_type, df=df, percentage=True)

plt.figure(figsize=(15,5.5))

ax = plt.subplot(121)

y = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:].index

v1 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,0]

v2 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,1]

ind = np.arange(len(y))

width=0.4

ax.barh(ind+0.4,v1,width,color='red', alpha=0.25)

ax.barh(ind,v2,width, color ='green', alpha=0.25)

ax.set(yticks=ind + width/2, yticklabels=y, ylim=[2*width - 1, len(y)])

ax.get_xaxis().set_visible(False)

for i in ax.patches:

    ax.text(i.get_width()+0.001, i.get_y()+.09, \

            str((round((i.get_width())*100,1)))+'%', fontsize=10,color='black')

plt.ylabel('')

plt.title(f'{crime_type} in percentage')



newdf = prepare_df(crime_type = crime_type, df=df, percentage=False)

ax1 = plt.subplot(122)

y = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:].index

v1 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,0]

v2 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,1]

ind = np.arange(len(y))

width=0.4

ax1.barh(ind+0.4,v1,width,label='years 2017, 2018 and 2019',color='red', alpha=0.25)

ax1.barh(ind,v2,width,label='years 2011, 2012 and 2013', color ='green', alpha=0.25)

ax1.set(yticks=ind + width/2, yticklabels=y, ylim=[2*width - 1, len(y)])

ax1.get_xaxis().set_visible(False)

ax1.get_yaxis().set_visible(False)

for i in ax1.patches:

    ax1.text(i.get_width()+0.001, i.get_y()+.09, \

            str((round((i.get_width()),1))), fontsize=10,color='black')

plt.legend(frameon=False)

plt.ylabel('')

plt.title(f'{crime_type} in percentage')

plt.subplots_adjust(left=0.4)



textstr = 'CBD more than double the number of "Theft from Vehicle"'

textstr2 = ''

plt.gcf().text(0.92, 0.82, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.79, textstr2, fontsize=10.5)



textstr = 'All neighbourhoods in the top 10 increased the number'

textstr2 = ' of occurrences.'

plt.gcf().text(0.92, 0.578, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.548, textstr2, fontsize=10.5)



textstr = 'Notice that even with a smaller timespan of 2.5 years'

textstr2 = '2017, 2018 and 2019 (until 30 June) have a greater '

textstr3 = 'number of occurrences than 2003, 2004 and 2005,'

textstr4 = 'confirming the increase in "Theft from Vehicle".'

plt.gcf().text(0.92, 0.218, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.188, textstr2, fontsize=10.5)

plt.gcf().text(0.92, 0.158, textstr3, fontsize=10.5)

plt.gcf().text(0.92, 0.128, textstr4, fontsize=10.5)



textstr = 'Top 10 neighbourhoods in number of incidents'

plt.gcf().text(0.5, 0.98, textstr, fontsize=12)

plt.show()
comments = 'Incidents more likely to happen in the evening, \nafter ~17.'

make_heatmap(df,color='green',crime_type=crime_type, years =[2011,2012,2013],comments=comments)

comments = f'No big change in the time when \n"{crime_type}" likely to occur, still\n concentrated in the evening.'

make_heatmap(df,color='red',crime_type=crime_type, years =[2017,2018,2019],comments=comments)
crime_type = 'Other Theft'

plt.figure(figsize=(10,5))

plt.rcParams['axes.facecolor'] = 'white'

#df[df.TYPE==crime_type].groupby(pd.Grouper(freq='BQ')).size().rolling(4).mean().plot(color='black',linestyle='--', label='rolling average 12 months')

df[df.TYPE==crime_type].groupby(pd.Grouper(freq='BQ')).size().plot(color='black',alpha=0.9, label = f'Number of "{crime_type}" incidents')

plt.title(f'{crime_type} per Quarter')

plt.legend(frameon=False)

plt.xlabel('')

plt.xlim(pd.Timestamp('2003-01-01'), pd.Timestamp('2019-06-30'))

plt.axvspan(pd.Timestamp('2003-01-01'), pd.Timestamp('2005-12-31'), facecolor='green', alpha=0.1)

plt.axvspan(pd.Timestamp('2017-01-01'), pd.Timestamp('2019-06-30'), facecolor='red', alpha=0.1)

plt.ylim(400)

textstr = f'"{crime_type}" numbers are increasing since 2003, numbers stabilized'

textstr2 = 'from ~2011 to ~2013 and in more recent years.'

#textstr3 = 'type of crime.'

plt.gcf().text(0.92, 0.82, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.79, textstr2, fontsize=10.5)

#plt.gcf().text(0.92, 0.76, textstr3, fontsize=10.5)



textstr = 'In the graphs below there will be a comparison between the total'

textstr2 = 'number of occurrences in the timespan between'

textstr21 = '2003 to 2005'

textstr22 = 'and'

textstr3 = '2017 to 2019 (June 30).'

plt.gcf().text(0.92, 0.62, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.59, textstr2, fontsize=10.5)

plt.gcf().text(1.235, 0.59, textstr21, fontsize=10.5, color='green',alpha=0.8)

plt.gcf().text(1.325, 0.59, textstr22, fontsize=10.5)

plt.gcf().text(0.92, 0.56, textstr3, fontsize=10.5, color='red', alpha=0.8)

plt.show()
newdf = prepare_df(crime_type = crime_type, df=df, percentage=True,years=[2003,2004,2005])

plt.figure(figsize=(15,5.5))

ax = plt.subplot(121)

y = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:].index

v1 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,0]

v2 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,1]

ind = np.arange(len(y))

width=0.4

ax.barh(ind+0.4,v1,width,color='red', alpha=0.25)

ax.barh(ind,v2,width, color ='green', alpha=0.25)

ax.set(yticks=ind + width/2, yticklabels=y, ylim=[2*width - 1, len(y)])

ax.get_xaxis().set_visible(False)

for i in ax.patches:

    ax.text(i.get_width()+0.001, i.get_y()+.09, \

            str((round((i.get_width())*100,1)))+'%', fontsize=10,color='black')

plt.ylabel('')

plt.title(f'{crime_type} in percentage')



newdf = prepare_df(crime_type = crime_type, df=df, percentage=False, years=[2003,2004,2005])

ax1 = plt.subplot(122)

y = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:].index

v1 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,0]

v2 = newdf.sort_values(by=['years 2017, 2018 and 2019'],ascending=True).iloc[-10:,1]

ind = np.arange(len(y))

width=0.4

ax1.barh(ind+0.4,v1,width,label='years 2017, 2018 and 2019',color='red', alpha=0.25)

ax1.barh(ind,v2,width,label='years 2003, 2004 and 2005', color ='green', alpha=0.25)

ax1.set(yticks=ind + width/2, yticklabels=y, ylim=[2*width - 1, len(y)])

ax1.get_xaxis().set_visible(False)

ax1.get_yaxis().set_visible(False)

for i in ax1.patches:

    ax1.text(i.get_width()+0.001, i.get_y()+.09, \

            str((round((i.get_width()),1))), fontsize=10,color='black')

plt.legend(frameon=False)

plt.ylabel('')

plt.title(f'{crime_type} in percentage')

plt.subplots_adjust(left=0.4)



textstr = 'Top 10 neighbourhoods in number of incidents'

plt.gcf().text(0.5, 0.98, textstr, fontsize=12)



textstr = 'CBD concentrated the most of "Other Theft"'

textstr2 = 'and still increased the number of occurrences.'

plt.gcf().text(0.92, 0.82, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.79, textstr2, fontsize=10.5)



textstr = 'Kensington-Cedar Cottage was the only neighbourhood'

textstr2 = 'in the top 10 to decrease number of "Other Theft".'

plt.gcf().text(0.92, 0.378, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.348, textstr2, fontsize=10.5)



textstr = 'Note that even with a smaller timespan of 2.5 years'

textstr2 = '2017, 2018 and 2019 (until 30 June) have a greater '

textstr3 = 'number of occurrences than 2011, 2012 and 2013,'

textstr4 = 'confirming the increase in "Other Theft".'

plt.gcf().text(0.92, 0.218, textstr, fontsize=10.5)

plt.gcf().text(0.92, 0.188, textstr2, fontsize=10.5)

plt.gcf().text(0.92, 0.158, textstr3, fontsize=10.5)

plt.gcf().text(0.92, 0.128, textstr4, fontsize=10.5)

plt.show()
comments = 'Incidents more likely to happen in the afternoon, \nbetween ~12 to ~19.'

make_heatmap(df,color='green',crime_type=crime_type, years =[2011,2012,2013],comments=comments)

comments = f'No big change in the time when "{crime_type}" \nlikely to occur, still concentrated in afternoon.'

make_heatmap(df,color='red',crime_type=crime_type, years =[2017,2018,2019],comments=comments)