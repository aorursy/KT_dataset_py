# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import json

import datetime as dt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('../input/pietsmiet-tv-thumbnails-labeled/PietSmietTV_labeled.xlsx')

df = df.drop(['json', 'index'], axis=1).fillna(0).rename(columns={'Brammen()':'Brammen', 'Chris()':'Chris', 'Jay()':'Jay', 'Piet()':'Piet', 'Sep()':'Sep'})

df['Sum'] = np.sum(df[['Brammen', 'Chris', 'Jay', 'Piet', 'Sep']], axis=1)

df['Paths'] = '../input/pietsmiettv-json-paths/' + df['Paths'].str.replace('.jpg', '.info.json')

df
#Moving Average

df['Average'] = 0.00

i = 7

alpha = 0.4



while i < len(df['Sum']):

    df['Average'][i] = np.mean(df['Sum'][i-7:i])

    i +=1



#Views

view_count = []

for path in df['Paths']:

    with open(path) as json_file: 

        data = json.load(json_file)

    view_count.append(data['view_count'])

    

df['Views'] = view_count



#Auf 10 Minuten gestreckt

duration = []

for path in df['Paths']:

    with open(path) as json_file: 

        data = json.load(json_file)

    duration.append(data['duration'])

    

df['duration (Min)'] = duration

df['duration (Min)'] = (df['duration (Min)'] / 60).astype(int)
df['Paths'] = df['Paths'].str.replace('../input/pietsmiettv-json-paths/', '')

df
n = df[df['Sum'] > 0].count()[0]

print('In {} Thumbnails ist mindestens einer der Jungs zu sehen. Das entspricht {}% aller Thumbnails.'.format(n, round(n/df.count()[0]*100, 2)))



m = df[(df['Sum'] > 0)&(df['date'] > '01.01.2018')].count()[0]

print('Seit Anfang 2018 enthielten {} Thumbnails mindestens einen der Jungs. Das entspricht {}%.'.format(m, round(m/df[(df['date'] > '01.01.2018')].count()[0]*100, 2)))



print(df['Sum'].value_counts())
plt.figure(figsize=(25,10))

plt.rcParams.update({'font.size': 20})

fig = sns.lineplot(x=df['date'], y=df['Average'], color='darkgreen', ci=None)

plt.ylabel('Anzahl der Schnittchen auf Thumbnails (geglättet)')

plt.xlabel('Datum')



plt.axvline(dt.datetime(2015, 10, 7), color='red')



#plt.savefig('Summe_Histogramm_insgesamt.jpg')
plt.figure(figsize=(15,10))

plt.rcParams.update({'font.size': 14})

sns.lineplot(x=df[df['date']<'01.01.2018']['date'], y=df['Average'], color='darkgreen', ci=None)

plt.fill_between(df[df['date']<'01.01.2018']['date'].values, df[df['date']<'01.01.2018']['Average'].values, alpha=0.5, color='darkgreen')



plt.ylabel('Anzahl der Schnittchen auf Thumbnails (geglättet)')

plt.xlabel('Datum')



#plt.savefig('Summe_Histogramm_vor_Aufbruch.jpg')
plt.figure(figsize=(20,10))

sns.lineplot(x=df[df['date']>'01.01.2018']['date'], y=df['Average'], color='darkgreen', ci=None)

plt.fill_between(df[df['date']>'01.01.2018']['date'].values, df[df['date']>'01.01.2018']['Average'].values, alpha=0.5, color='darkgreen')

plt.ylabel('Anzahl der Schnittchen auf Thumbnails (geglättet)')

plt.xlabel('Datum')



plt.axvline(dt.datetime(2019, 11, 1), color='red')

#plt.savefig('Summe_Histogramm_nach_Aufbruch.jpg')
df[(df['date']>'01.01.2018')&(df['date']>'01.11.2019')].groupby('Sum').count()
#Erscheinungen im Zeitverlauf

pre_Aufbruch_ratio = round(df[(df['date']<'07.10.2015')&(df['Sum']>0)]['Sum'].count() / df[df['date']<'07.10.2015']['Sum'].count() *100, 2)

post_Aufbruch_ratio = round(df[(df['date']>'07.10.2015')&(df['Sum']>0)]['Sum'].count() / df[df['date']>'07.10.2015']['Sum'].count()*100, 2)



print('Vor #Aufbruch war in {}% der Thumbnails mindestens einer der Jungs zu sehen. Seit #Aufbruch sind es {}%'.format(pre_Aufbruch_ratio, post_Aufbruch_ratio))



print('\nEinige Monate vor #Aufbruch wurde der Kanal PietSmietTV bis auf weiteres stillgelegt.')

print('Vielleicht ist die Schließung des Kanals ein Indiz dafür wie schwierig diese Zeit für alle Beteiligten war \n'

      'und sie sich in einer Phase der Neuorientierung befanden.')





pre_daily_ratio = round(df[(df['date']<'01.11.2019')&(df['Sum']>0)]['Sum'].count() / df[df['date']<'01.11.2019']['Sum'].count() *100, 2)

post_daily_ratio = round(df[(df['date']>'01.11.2019')&(df['Sum']>0)]['Sum'].count() / df[df['date']>'01.11.2019']['Sum'].count()*100, 2)



print('\nBevor die Daily-reacts begannen waren auf {}% der Thumbnails mindestens einer der Jungs. Seit den Daily Reacts sind es {}%'.format(pre_daily_ratio, post_daily_ratio))
#Schnittchen auf Thumnails VOR der Pause

x = df[df['date']<'07.10.2015'][['date', 'Sum']].set_index('date')



plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(5,5))

sns.violinplot(data=x, color='darkred')

plt.yticks([0,1,2,3,4,5])

plt.ylabel('Schnittchen auf Thumbnails')

plt.xlabel('n = ' + str(x.shape[0]))



#plt.savefig('Gesichter_Verteilung_vor_Aufbruch.jpg')
#Schnittchen auf Thumnails NACH der Pause

y = df[df['date']>'07.10.2015'][['date', 'Sum']].set_index('date')

sns.violinplot(data=y, color='darkgreen')

plt.yticks([0,1,2,3,4,5])

plt.ylabel('Schnittchen auf Thumbnails')

plt.xlabel('n = ' + str(y.shape[0]))



#plt.savefig('Gesichter_Verteilung_nach_Aufbruch.jpg')
#Schnittchen auf Thumnails SEIT den Daily Reacts

z = df[df['date']>'01.11.2019'][['date', 'Sum']].set_index('date')

sns.violinplot(data=z, color='darkblue')

plt.yticks([0,1,2,3,4,5])

plt.ylabel('Schnittchen auf Thumbnails')

plt.xlabel('n = ' + str(z.shape[0]))



#plt.savefig('Gesichter_Verteilung_seit_Daily.jpg')
#Wer war wie oft auf Thumbnails (insgesamt)?



app_df = df[['Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'date']].set_index('date')

#app_df = app_df[app_df.index > '01.11.2019']



pre_break_app = np.sum(app_df)



plt.figure(figsize=(8,8))

sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Set1')

plt.ylabel('Erscheinungen auf Thumbnails')



splot = sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Greens_r')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    

#plt.savefig('Erscheinungen_pro_Person.jpg')
#Wer war wie oft auf Thumbnails (seit #Aufbruch)?



app_df = df[['Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'date']].set_index('date')

app_df = app_df[app_df.index > '01.11.2017']



pre_break_app = np.sum(app_df)



plt.figure(figsize=(8,8))

sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Greens_r')

plt.ylabel('Erscheinungen auf Thumbnails')



splot = sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Greens_r')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    

#plt.savefig('Erscheinungen_pro_Person_nach_Aufbruch.jpg')
#Wer war wie oft auf Thumbnails (seit Daily Reacts)?



app_df = df[['Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'date']].set_index('date')

app_df = app_df[app_df.index > '01.11.2019']



pre_break_app = np.sum(app_df)



plt.figure(figsize=(8,8))

sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Greens_r')

plt.ylabel('Erscheinungen auf Thumbnails')



splot = sns.barplot(x=pre_break_app.index, y=pre_break_app.values, palette='Greens_r')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    

#plt.savefig('Erscheinungen_pro_Person_seit_Daily.jpg')
#Appearances over time, seit #Aufbruch

#No insights, abort.



app_over_time = app_df[app_df.index > '01.11.2019'].resample('M').sum()



plt.figure(figsize=(30,10))

sns.lineplot(data=app_over_time, palette='CMRmap_r')
#Kommen Jungs nur vor wenn eh alle drauf sind?

#zsm steht für zusammen.



zsm_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum']]



zsm_sep = round(zsm_df[(zsm_df['Sep']==1) & (zsm_df['Sum']==1)].count()[0] / zsm_df[zsm_df['Sep']==1].count()[0] *100, 2)

print('Wenn Sep auf einem Thumbnail ist, ist er zu {}% alleine darauf'.format(zsm_sep))



zsm_brammen = round(zsm_df[(zsm_df['Brammen']==1) & (zsm_df['Sum']==1)].count()[0] / zsm_df[zsm_df['Brammen']==1].count()[0] *100, 2)

print('Wenn Brammen auf einem Thumbnail ist, ist er zu {}% alleine darauf'.format(zsm_brammen))



zsm_chris = round(zsm_df[(zsm_df['Chris']==1) & (zsm_df['Sum']==1)].count()[0] / zsm_df[zsm_df['Chris']==1].count()[0] *100, 2)

print('Wenn Chris auf einem Thumbnail ist, ist er zu {}% alleine darauf'.format(zsm_chris))



zsm_piet = round(zsm_df[(zsm_df['Piet']==1) & (zsm_df['Sum']==1)].count()[0] / zsm_df[zsm_df['Piet']==1].count()[0] *100, 2)

print('Wenn Piet auf einem Thumbnail ist, ist er zu {}% alleine darauf'.format(zsm_piet))



zsm_jay = round(zsm_df[(zsm_df['Jay']==1) & (zsm_df['Sum']==1)].count()[0] / zsm_df[zsm_df['Jay']==1].count()[0] *100, 2)

print('Wenn Jay auf einem Thumbnail ist, ist er zu {}% alleine darauf'.format(zsm_jay))
for col in ['Brammen', 'Chris', 'Jay', 'Piet', 'Sep']:

    for column in ['Brammen', 'Chris', 'Jay', 'Piet', 'Sep']:

        x = round(zsm_df[(zsm_df[col]==1) & (zsm_df[column]==1)].count()[0] / zsm_df[zsm_df[col]==1].count()[0] *100, 2)

        print('Wenn {} im Thumbnail ist, ist zu {}% auch {} zu sehen.'.format(col, x, column))

    print('')

        
Heatmap_df = df[['Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'date']].set_index('date')

Heatmap_df
#Wer mit wem (insgesamt)

heatmap_df = pd.DataFrame(columns=['Brammen', 'Chris', 'Jay', 'Piet', 'Sep'])

for col in Heatmap_df.columns:

    for column in Heatmap_df.columns:

        heatmap_df.loc[column,col] = Heatmap_df[(Heatmap_df[col]==1)&(Heatmap_df[column]==1)].count()[0]



heatmap_df[heatmap_df.columns] = heatmap_df[heatmap_df.columns].astype(int)



heatmap_df

plt.figure(figsize=(8,6))

sns.heatmap(data=heatmap_df, annot=True, fmt='g',cmap='Greens')

plt.title('Wie oft wer mit wem zu sehen ist')



#plt.savefig('Wer_mit_wem.jpg')
#Wer mit wem (seit #Aufbruch)

heatmap_df = pd.DataFrame(columns=['Brammen', 'Chris', 'Jay', 'Piet', 'Sep'])

for col in Heatmap_df.columns:

    for column in Heatmap_df.columns:

        heatmap_df.loc[column,col] = Heatmap_df[(Heatmap_df[col]==1)&(Heatmap_df[column]==1)&(Heatmap_df.index>'01.11.2017')].count()[0]



heatmap_df[heatmap_df.columns] = heatmap_df[heatmap_df.columns].astype(int)



heatmap_df

plt.figure(figsize=(8,6))

sns.heatmap(data=heatmap_df, annot=True, fmt='g',cmap='Greens')

plt.title('Wie oft wer mit wem zu sehen ist')



#plt.savefig('Wer_mit_wem_nach_Aufbruch.jpg')
#Wer mit wem (seit Daily Reacts)

heatmap_df = pd.DataFrame(columns=['Brammen', 'Chris', 'Jay', 'Piet', 'Sep'])

for col in Heatmap_df.columns:

    for column in Heatmap_df.columns:

        heatmap_df.loc[column,col] = Heatmap_df[(Heatmap_df[col]==1)&(Heatmap_df[column]==1)&(Heatmap_df.index>'01.11.2019')].count()[0]



heatmap_df[heatmap_df.columns] = heatmap_df[heatmap_df.columns].astype(int)



heatmap_df

plt.figure(figsize=(8,6))

sns.heatmap(data=heatmap_df, annot=True, fmt='g',cmap='Greens')

plt.title('Wie oft wer mit wem zu sehen ist')



plt.savefig('Wer_mit_wem_seit_Daily.jpg')
#Sep nur drauf, wenn alle drauf sind?.



sep_alle = round(zsm_df[(zsm_df['Sep']==1) & (zsm_df['Sum']==1) & (zsm_df['date']>'01.11.2019')].count()[0] / zsm_df[(zsm_df['Sep']==1) & (zsm_df['date']>'01.11.2019')].count()[0] *100, 2)

print('Wenn Sep auf einem Thumbnail ist, sind zu {}% alle darauf'.format(sep_alle))



#Klickzahlen nach Anzahl derJungs auf dem Thumbnail (insgesamt)



view_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'Views']]#[df['date']>'01.01.2018']



view_per_app = view_df.sort_values(by='Views', ascending=False)

view_per_app

plt.figure(figsize=(15,8))

sns.swarmplot(x="Sum", y="Views", data=view_per_app)

plt.yticks(range(0, 1700000, 200000))

plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))



#plt.savefig('Klickzahlen_pro_Anzahl.jpg')
#Klickzahlen nach Anzahl derJungs auf dem Thumbnail (seit #Aufbruch)

view_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'Views']][df['date']>'01.01.2018']



view_per_app = view_df.sort_values(by='Views', ascending=False)

view_per_app

plt.figure(figsize=(15,8))

sns.swarmplot(x="Sum", y="Views", data=view_per_app)

plt.yticks(range(0, 1700000, 200000))

plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))



#plt.savefig('Klickzahlen_pro_Anzahl_seit_Aufbruch.jpg')
#Klickzahlen nach Anzahl derJungs auf dem Thumbnail (seit Daily Reacts)

view_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'Views']][df['date']>'01.11.2019']



view_per_app = view_df.sort_values(by='Views', ascending=False)

view_per_app

plt.figure(figsize=(15,8))

sns.swarmplot(x="Sum", y="Views", data=view_per_app)

plt.yticks(range(0, 1700000, 200000))

plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))



#plt.savefig('Klickzahlen_pro_Anzahl_seit_Daily.jpg')
brammen_views = view_df[view_df['Brammen']==1]['Views'].mean()

print('Wenn Brammen im Thumbnail ist, hat das Video im Schnitt {} Views.'.format(round(brammen_views)))



chris_views = view_df[view_df['Chris']==1]['Views'].mean()

print('Wenn Chris im Thumbnail ist, hat das Video im Schnitt {} Views.'.format(round(chris_views)))



jay_views = view_df[view_df['Jay']==1]['Views'].mean()

print('Wenn Jay im Thumbnail ist, hat das Video im Schnitt {} Views.'.format(round(jay_views)))



piet_views = view_df[view_df['Piet']==1]['Views'].mean()

print('Wenn Piet im Thumbnail ist, hat das Video im Schnitt {} Views.'.format(round(piet_views)))



sep_views = view_df[view_df['Sep']==1]['Views'].mean()

print('Wenn Sep im Thumbnail ist, hat das Video im Schnitt {} Views.'.format(round(sep_views)))
#Beste Videos und wer drauf war

view_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'Views']]

view_df.sort_values(by='Views', ascending=False)
dur_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'duration (Min)']]



plt.figure(figsize=(12,5))

sns.swarmplot(x="Sum", y="duration (Min)", data=dur_df)



#plt.savefig('Dauer_pro_Anzahl.jpg')
dur_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'duration (Min)']][df['date']>'01.01.2018']



plt.figure(figsize=(12,5))

sns.swarmplot(x="Sum", y="duration (Min)", data=dur_df)



#plt.savefig('Dauer_pro_Anzahl_seit_Aufbruch.jpg')
dur_df = df[['date', 'Brammen', 'Chris', 'Jay', 'Piet', 'Sep', 'Sum', 'duration (Min)']][df['date']>'01.11.2019']



plt.figure(figsize=(12,5))

sns.swarmplot(x="Sum", y="duration (Min)", data=dur_df)



#plt.savefig('Dauer_pro_Anzahl_seit_Daily.jpg')
brammen_dur = dur_df[dur_df['Brammen']==1]['duration (Min)']

print('Wenn Brammen im Thumbnail ist, geht das Video im Schnitt {} Min.'.format(round(brammen_dur.mean())))



chris_dur = dur_df[dur_df['Chris']==1]['duration (Min)']

print('Wenn Chris im Thumbnail ist, geht das Video im Schnitt {} Min.'.format(round(chris_dur.mean())))



jay_dur = dur_df[dur_df['Jay']==1]['duration (Min)']

print('Wenn Jay im Thumbnail ist, geht das Video im Schnitt {} Min.'.format(round(jay_dur.mean())))



piet_dur = dur_df[dur_df['Piet']==1]['duration (Min)']

print('Wenn Piet im Thumbnail ist, geht das Video im Schnitt {} Min.'.format(round(piet_dur.mean())))



sep_dur = dur_df[dur_df['Sep']==1]['duration (Min)']

print('Wenn Sep im Thumbnail ist, geht das Video im Schnitt {} Min.'.format(round(sep_dur.mean())))
gestreckt = dur_df[dur_df['duration (Min)']==10].sum()



plt.figure(figsize=(8,6))

sns.barplot(x=gestreckt.index[0:5], y=gestreckt.values[0:5], palette='Greens_r')

plt.ylabel('Anzahl der 10-Minuten Videos')



#plt.savefig('10_Minuten_gestreckt.jpg')
df['Paths'] = df['Paths'].str.replace('.info.json','')

df = df.set_index('Paths')

df
df_test = pd.read_csv('../input/pietsmiet-tv-thumbnails/Dataframe_jpg_only3.csv')

df_test['json'] = df_test['json'].str.replace('/Users/Frank/Desktop/PietSmietTV Thumbnails/', '').str.replace('.info.json', '')

df_test = df_test[['json', 'keys']].set_index('keys')

df_test
df_final = df.join(df_test, how='outer')

df_final
df_final = df_final.rename(columns={'json':'index'})

df_final = df_final.set_index('index')

df_final
df_final.to_excel('PietSmietTV Thumbnail-Analysis.xlsx')