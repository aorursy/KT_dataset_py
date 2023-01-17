#import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt #plotting, math, stats

%matplotlib inline

import seaborn as sns #plotting, regressions

import datetime

import os

import functools

from scipy import stats

#Import Data



#Confirmed Cases per country

df = pd.read_csv('/kaggle/input/cases-and-deaths-per-country/cases.csv', sep=';')

df = df.fillna(value=0)



measuresdf = pd.read_csv ('/kaggle/input/covid19-national-responses-dataset/COVID 19 Containment measures data.csv')

countermeasuresdf = pd.read_csv ('/kaggle/input/covid19-national-responses-dataset/countermeasures_db_johnshopkins_2020_03_30.csv')

#Organization of the entities to be compared

AR=df.loc[df['Country']== 'Argentina']

US=df.loc[df['Code']== 'USA']

BR=df.loc[df['Country']== 'Brazil']

IT=df.loc[df['Country']== 'Italy']

SP=df.loc[df['Country']== 'Spain']

CL=df.loc[df['Country']== 'Chile']

BO=df.loc[df['Country']== 'Bolivia']

PY=df.loc[df['Country']== 'Paraguay']

CH=df.loc[df['Country']== 'China']

CO=df.loc[df['Country']== 'Colombia']

CR=df.loc[df['Code']== 'CRI']

EC=df.loc[df['Country']== 'Ecuador']

ES=df.loc[df['Country']== 'Estonia']

FR=df.loc[df['Country']== 'France']

GU=df.loc[df['Country']== 'Guatemala']

MX=df.loc[df['Country']== 'Mexico']

PE=df.loc[df['Country']== 'Peru']

UK=df.loc[df['Country']== 'United Kingdom']

UY=df.loc[df['Country']== 'Uruguay']

SK=df.loc[df['Country']== 'South Korea']

World=df.loc[df['Country']== 'World']
data = [['South Korea', SK['Day'].max(),measuresdf.loc[measuresdf['Country']== 'South Korea'].Country.count(),SK['Cases'].max(), SK['Deaths'].max(),SK['Deaths'].max()/SK['Cases'].max()*100,SK['Daily_Cases'].mean(),SK['Daily_Cases'].median(),SK['Daily_Cases'].std(),SK['Daily_Fatalities'].mean(),SK['Daily_Fatalities'].median(),SK['Daily_Fatalities'].std()],

        ['Argentina', AR['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Argentina'].Country.count(),AR['Cases'].max(), AR['Deaths'].max(),AR['Deaths'].max()/AR['Cases'].max()*100,AR['Daily_Cases'].mean(),AR['Daily_Cases'].median(),AR['Daily_Cases'].std(),AR['Daily_Fatalities'].mean(),AR['Daily_Fatalities'].median(),AR['Daily_Fatalities'].std()],

        ['United Kingdom', UK['Day'].max(),measuresdf.loc[measuresdf['Country']== 'United Kingdom'].Country.count(),UK['Cases'].max(), UK['Deaths'].max(),UK['Deaths'].max()/UK['Cases'].max()*100,UK['Daily_Cases'].mean(),UK['Daily_Cases'].median(),UK['Daily_Cases'].std(),UK['Daily_Fatalities'].mean(),UK['Daily_Fatalities'].median(),UK['Daily_Fatalities'].std()],

        ['Chile', CL['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Chile'].Country.count(),CL['Cases'].max(), CL['Deaths'].max(),CL['Deaths'].max()/CL['Cases'].max()*100,CL['Daily_Cases'].mean(),CL['Daily_Cases'].median(),CL['Daily_Cases'].std(),CL['Daily_Fatalities'].mean(),CL['Daily_Fatalities'].median(),CL['Daily_Fatalities'].std()],

        ['Italy', IT['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Italy'].Country.count(),IT['Cases'].max(), IT['Deaths'].max(),IT['Deaths'].max()/IT['Cases'].max()*100,IT['Daily_Cases'].mean(),IT['Daily_Cases'].median(),IT['Daily_Cases'].std(),IT['Daily_Fatalities'].mean(),IT['Daily_Fatalities'].median(),IT['Daily_Fatalities'].std()],

        ['Spain', SP['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Spain'].Country.count(),SP['Cases'].max(), SP['Deaths'].max(),SP['Deaths'].max()/SP['Cases'].max()*100,SP['Daily_Cases'].mean(),SP['Daily_Cases'].median(),SP['Daily_Cases'].std(),SP['Daily_Fatalities'].mean(),SP['Daily_Fatalities'].median(),SP['Daily_Fatalities'].std()],

        ['China',CH['Day'].max(),measuresdf.loc[measuresdf['Country']== 'China'].Country.count(),CH['Cases'].max(), CH['Deaths'].max(),CH['Deaths'].max()/CH['Cases'].max()*100,CH['Daily_Cases'].mean(),CH['Daily_Cases'].median(),CH['Daily_Cases'].std(),CH['Daily_Fatalities'].mean(),CH['Daily_Fatalities'].median(),CH['Daily_Fatalities'].std()],

        ['Brazil', BR['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Brazil'].Country.count(),BR['Cases'].max(), BR['Deaths'].max(),BR['Deaths'].max()/BR['Cases'].max()*100,BR['Daily_Cases'].mean(),BR['Daily_Cases'].median(),BR['Daily_Cases'].std(),BR['Daily_Fatalities'].mean(),BR['Daily_Fatalities'].median(),BR['Daily_Fatalities'].std()],

        ['World', World['Day'].max(),measuresdf.loc[measuresdf['Country']== 'World'].Country.count(),World['Cases'].max(), World['Deaths'].max(),World['Deaths'].max()/World['Cases'].max()*100,World['Daily_Cases'].mean(),World['Daily_Cases'].median(),World['Daily_Cases'].std(),World['Daily_Fatalities'].mean(),World['Daily_Fatalities'].median(),World['Daily_Fatalities'].std()],

        ['United States', US['Day'].max(),measuresdf.loc[measuresdf['Country']== 'United States'].Country.count(),US['Cases'].max(), US['Deaths'].max(),US['Deaths'].max()/US['Cases'].max()*100,US['Daily_Cases'].mean(),US['Daily_Cases'].median(),US['Daily_Cases'].std(),US['Daily_Fatalities'].mean(),US['Daily_Fatalities'].median(),US['Daily_Fatalities'].std()]]



df = pd.DataFrame(data, columns = ['Country', 'Days','Measures','Cases', 'Deaths', 'Death ratio','Cases Mean','Cases Median','Cases Std','Deaths Mean','Deaths Median','Deaths Std']) 



sns.set(style="whitegrid")

plt.figure(figsize=(25,15))

plt.title('Cases and fatalities per country') # Title

df = df.sort_values(by=['Cases'])



ax1 = sns.barplot(x='Country', y='Cases',label='Cases per country', data=df)



ax2 = ax1.twinx()

ax2.tick_params(axis='y')

ax2 = sns.lineplot(x='Country', y='Deaths',marker='*', label='Fatalities per country', data=df)



plt.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

df['Death ratio'] = df['Death ratio'].map('{:,.2f}%'.format)

df['Cases Mean'] = df['Cases Mean'].map('{:,.2f}'.format)

df['Deaths Mean'] = df['Deaths Mean'].map('{:,.2f}'.format)

df['Cases Median'] = df['Cases Median'].map('{:,.2f}'.format)

df['Deaths Median'] = df['Deaths Median'].map('{:,.2f}'.format)

df['Cases Std'] = df['Cases Std'].map('{:,.2f}'.format)

df['Deaths Std'] = df['Deaths Std'].map('{:,.2f}'.format)

df['Cases'] = df['Cases'].map('{:,}'.format)

df['Deaths'] = df['Deaths'].map('{:,}'.format)

df
x = AR

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();

x = SK

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = CL

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = BR

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = CH

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = UK

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = SP

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
x = IT

plt.figure(figsize=(25,15))

plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title

sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);

sns.kdeplot(x['Daily_Cases'])

plt.legend();
Countries=pd.concat([World]) 

Countries=Countries.sort_values(by=['Day','Date'], ascending=[True,True])



plt.figure(figsize=(25,15))

plt.title('World details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([SK]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('South Korea details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CH]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('China details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([US]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('US details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CL]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Chile details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([UY]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Uruguay details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)





legend = plt.legend()

legend.texts[0].set_text("")

legend.texts[1].set_text("Daily Cases")

legend.texts[2].set_text("")

legend.texts[3].set_text("Daily Fatalities")





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([World]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('World details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([SK]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('South Korea details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CH]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('China details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CL]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Chile details') # Title

#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)

sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)





plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
# Concatenate dataframes 

Countries = pd.concat([World, SK, CH,US,UK, SP, IT]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('World, South Korea, China, UK and US reported COVID19 infections') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
# Concatenate dataframes 

Countries=pd.concat([BR, US, SP, IT]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Brazil, Spain, Italy and US reported COVID19 infections') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 infections') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 fatalities') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()

Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 fatalities') # Title

sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([SK,AR,IT, SP]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('South Korea, Argentina, Italia and Spain reported COVID19 infections') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([SK,AR,IT, SP]) 

Countries = Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('South Korea, Argentina, Italia and Spain reported COVID19 fatalities') # Title

sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR,CL, UY]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina, Chile and Uruguay reported COVID19 infection') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

#sns.barplot(x="Day", y="Deaths", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([AR,CL, UY]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Argentina, Chile and Uruguay reported COVID19 fatalities') # Title

sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)

#sns.barplot(x="Day", y="Deaths", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CL,AR,BO, PY, UY]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Chile, Argentina, Bolivia, Paraguay and Uruguay reported COVID19 infections') # Title

sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()
Countries=pd.concat([CL,AR,BO, PY, UY]) 

Countries=Countries.sort_values(by=['Day'], ascending=True)



plt.figure(figsize=(25,15))

plt.title('Chile, Argentina, Bolivia, Paraguay and Uruguay reported COVID19 deaths') # Title

sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)

plt.xticks(Countries.Day.unique(), rotation=90)

plt.show()