import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime, timedelta,date

import matplotlib.ticker as mticker
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_covid19.head()
df_covid19['confirmed_rank'] = df_covid19['Confirmed'].rank(ascending=False)
df_covid19[df_covid19['confirmed_rank'] == 1][['Country_Region','Confirmed','confirmed_rank']]
df_covid19['Death_Rate'] = np.round(df_covid19['Deaths']/df_covid19['Confirmed']*100,2)

df_covid19['Recovery_Rate'] = np.round(df_covid19['Recovered']/df_covid19['Confirmed']*100,2)
df_covid19[df_covid19['confirmed_rank'] == 1][['Country_Region','Confirmed','confirmed_rank','Deaths','Recovered','Death_Rate','Recovery_Rate']]
df_covid19.isnull().sum()
df_covid19.drop(['People_Tested','People_Hospitalized'], axis = 1,inplace = True)
df_covid19.isnull().sum()
df_covid19.sort_values(by = 'Confirmed', ascending = False,inplace = True)

plt.figure(figsize=(12,8))

bar = sns.barplot(x='Confirmed',y='Country_Region',data=df_covid19[df_covid19['confirmed_rank'] <= 10],palette='OrRd_r')

plt.ylabel("Countries",fontsize = 14)

plt.xlabel("Number of confirmed cases", fontsize = 14)

plt.title('Top 10 most affected countries',fontsize=20)
df_covid19.sort_values(by = 'Confirmed', ascending = False,inplace = True)

plt.figure(figsize=(12,12))

bar = sns.barplot(x='Confirmed',y='Country_Region',data = df_covid19[(df_covid19['confirmed_rank'] > 10) 

                  & (df_covid19['confirmed_rank'] <= 60)] ,palette='OrRd_r')

plt.ylabel("Countries",fontsize = 14)

plt.xlabel("Number of confirmed cases", fontsize = 14)

plt.title('Next 50 most affected countries',fontsize=20)
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(12, 15))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Country_Region", data=df_covid19[(df_covid19['Confirmed'] > 10000) & (df_covid19['Country_Region'] != 'US')],

            label="Total Cases", color="b")





sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Country_Region", data=df_covid19[(df_covid19['Confirmed'] > 10000)  & (df_covid19['Country_Region'] != 'US')],

                                                               label="Recovered", color="b")



ax.legend(ncol=2, loc="lower right", frameon=True)

#ax.set(xlim=(0, 24), ylabel="",

       #xlabel="Automobile collisions per billion miles")

sns.despine(left=True, bottom=True)

plt.title('Recovered cases from total cases',fontsize=20)
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(12, 15))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Country_Region", data=df_covid19[(df_covid19['Confirmed'] > 10000) & (df_covid19['Country_Region'] != 'US')],

            label="Total Cases", color="b")





sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Country_Region", data=df_covid19[(df_covid19['Confirmed'] > 10000)  & (df_covid19['Country_Region'] != 'US')],

                                                               label="Recovered", color="b")





sns.set_color_codes("dark")

sns.barplot(x="Deaths", y="Country_Region", data=df_covid19[(df_covid19['Confirmed'] > 10000)  & (df_covid19['Country_Region'] != 'US')],

                                                               label="Deaths", color="r")



ax.legend(ncol=3, loc="lower right", frameon=True)

#ax.set(xlim=(0, 24), ylabel="",

       #xlabel="Automobile collisions per billion miles")

sns.despine(left=True, bottom=True)

plt.title('Deaths and Recovery from Total Cases',fontsize=20)
df_covid19.sort_values(by = 'Death_Rate', ascending = False,inplace = True)

plt.figure(figsize=(14,10))

bar = sns.barplot(x='Death_Rate',y='Country_Region',data=df_covid19[df_covid19['Confirmed'] > 10000].head(50),palette='OrRd_r')

plt.ylabel("Countries",fontsize = 14)

plt.xlabel("Death rate per 100 confirmed cases", fontsize = 14)

plt.title('Top 50 countries by Death Rate (Minimum 10000 confirmed cases)',fontsize=20)
df_covid19.sort_values(by = 'Recovery_Rate', ascending = False,inplace = True)

plt.figure(figsize=(14,10))

bar = sns.barplot(x='Recovery_Rate',y='Country_Region',data=df_covid19[df_covid19['Confirmed'] > 10000].head(50),palette='Greens_r')

plt.ylabel("Countries",fontsize = 14)

plt.xlabel("Recovery rate per 100 confirmed cases", fontsize = 14)

plt.title('Top 50 countries by Recovery Rate (Minimum 10000 confirmed cases)',fontsize=20)
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_confirmed1 = df_confirmed.drop(['Province/State','Lat','Long'],axis = 1)
df_confirmed1.head()
new_sum_row = list(df_confirmed1.sum(axis=0))

new_sum_row[0] = 'Total'

df_confirmed2 = df_confirmed1.append(pd.Series(new_sum_row, index=df_confirmed1.columns ), ignore_index=True)

tran = df_confirmed2.set_index('Country/Region').transpose()

tran.reset_index(inplace = True)

tran.rename(columns = {'index':'Date'}, inplace = True) 

tran['new_date'] = pd.to_datetime(tran['Date'])

tran['new_date1']=tran['new_date'].apply(lambda x: x.strftime("%d %b"))

# there are duplicate columns for country

tran1 = tran.groupby(tran.columns, axis=1).sum()
tran1.tail()
import matplotlib.ticker as mticker



fig, ax = plt.subplots(figsize=(12, 8))

style = dict(size=12, color='gray')

plt.plot(tran1['new_date1'],tran1['Total'],color='red',marker = 'o')

plt.xticks(rotation=45)

ax.grid(False)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

plt.yticks(np.arange(0, tran1['Total'].max(),200000))

#ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

#ax.xaxis.set_minor_formatter(mticker.NullFormatter())

ax.text('17 Mar',300000, "Exponential increase started", **style,ha='right',va = 'top')

plt.title('Increase in Number of Confirmed Cases Worldwide',fontsize=20)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))





axes[0,0].tick_params('x', labelrotation=45)

axes[0,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,0].plot(tran1['new_date1'],tran1['Italy'],color='red',marker = 'o')

axes[0,0].set_title('Increase in Number of Confirmed Cases in Italy',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[0,1].tick_params('x', labelrotation=45)

axes[0,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,1].plot(tran1['new_date1'],tran1['Spain'],color='red',marker = 'o')

axes[0,1].set_title('Increase in Number of Confirmed Cases in Spain',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



axes[1,0].tick_params('x', labelrotation=45)

axes[1,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,0].plot(tran1['new_date1'],tran1['United Kingdom'],color='red',marker = 'o')

axes[1,0].set_title('Increase in Number of Confirmed Cases in UK',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)





axes[1,1].tick_params('x', labelrotation=45)

axes[1,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,1].plot(tran1['new_date1'],tran1['Germany'],color='red',marker = 'o')

axes[1,1].set_title('Increase in Number of Confirmed Cases in Germany',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



#ax.text('17 Mar',300000, "Exponential increase started", **style,ha='right',va = 'top')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))





axes[0,0].tick_params('x', labelrotation=45)

axes[0,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,0].plot(tran1['new_date1'],tran1['US'],color='red',marker = 'o')

axes[0,0].set_title('Increase in Number of Confirmed Cases in US',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[0,1].tick_params('x', labelrotation=45)

axes[0,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,1].plot(tran1['new_date1'],tran1['Canada'],color='red',marker = 'o')

axes[0,1].set_title('Increase in Number of Confirmed Cases in Canada',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



axes[1,0].tick_params('x', labelrotation=45)

axes[1,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,0].plot(tran1['new_date1'],tran1['Chile'],color='red',marker = 'o')

axes[1,0].set_title('Increase in Number of Confirmed Cases in Chile',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)





axes[1,1].tick_params('x', labelrotation=45)

axes[1,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,1].plot(tran1['new_date1'],tran1['Brazil'],color='red',marker = 'o')

axes[1,1].set_title('Increase in Number of Confirmed Cases in Brazil',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)
import matplotlib.ticker as mticker



fig, ax = plt.subplots(figsize=(12, 8))



style = dict(size=12, color='gray')

plt.plot(tran1['new_date1'],tran1['India'],color='red',marker = 'o')

plt.xticks(rotation=45)

#ax.grid(False)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

#plt.yticks(np.arange(0, tran1['Total'].max(),200000))

#ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

#ax.xaxis.set_minor_formatter(mticker.NullFormatter())

#ax.text('17 Mar',300000, "Exponential increase started", **style,ha='right',va = 'top')

plt.title('Increase in Number of Confirmed Cases in India',fontsize=20)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))





axes[0].tick_params('x', labelrotation=45)

axes[0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0].plot(tran1['new_date1'],tran1['China'],color='red',marker = 'o')

axes[0].set_title('Increase in Number of Confirmed Cases in China',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[1].tick_params('x', labelrotation=45)

axes[1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1].plot(tran1['new_date1'],tran1['Korea, South'],color='red',marker = 'o')

axes[1].set_title('Increase in Number of Confirmed Cases in Sount Korea',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)

#https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons/notebook

# Retriving Dataset

df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])

df_table['new_date1']=df_table['Last_Update'].apply(lambda x: x.strftime("%d %b"))
US_confirmed = df_table[(df_table['Country_Region'] == 'US') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

US_confirmed1 = US_confirmed.groupby('Last_Update', axis=0).sum().reset_index()

US_confirmed1['new_date1']=US_confirmed1['Last_Update'].apply(lambda x: x.strftime("%d %b"))

US_confirmed1.sort_values('Confirmed',inplace = True)

US_confirmed1['Day'] = US_confirmed1['Confirmed'].rank(method = 'first')

US_confirmed_fin = US_confirmed1[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_US'})
Canada_confirmed = df_table[(df_table['Country_Region'] == 'Canada') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Canada_confirmed.sort_values('Confirmed',inplace = True)

Canada_confirmed['Day'] = Canada_confirmed['Confirmed'].rank(method = 'first')

Canada_confirmed_fin = pd.DataFrame(Canada_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Canada'}))

Canada_confirmed_fin.reset_index(drop = True, inplace = True)
Italy_confirmed = df_table[(df_table['Country_Region'] == 'Italy') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Italy_confirmed.sort_values('Confirmed',inplace = True)

Italy_confirmed['Day'] = Italy_confirmed['Confirmed'].rank(method = 'first')

Italy_confirmed_fin = pd.DataFrame(Italy_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Italy'}))

Italy_confirmed_fin.reset_index(drop = True, inplace = True)
Spain_confirmed = df_table[(df_table['Country_Region'] == 'Spain') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Spain_confirmed.sort_values('Confirmed',inplace = True)

Spain_confirmed['Day'] = Spain_confirmed['Confirmed'].rank(method = 'first')

Spain_confirmed_fin = pd.DataFrame(Spain_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Spain'}))

Spain_confirmed_fin.reset_index(drop = True, inplace = True)
China_confirmed = df_table[(df_table['Country_Region'] == 'China') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

China_confirmed.sort_values('Confirmed',inplace = True)

China_confirmed['Day'] = China_confirmed['Confirmed'].rank(method = 'first')

China_confirmed_fin = pd.DataFrame(China_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_China'}))

China_confirmed_fin.reset_index(drop = True, inplace = True)
Japan_confirmed = df_table[(df_table['Country_Region'] == 'Japan') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Japan_confirmed.sort_values('Confirmed',inplace = True)

Japan_confirmed['Day'] = Japan_confirmed['Confirmed'].rank(method = 'first')

Japan_confirmed_fin = pd.DataFrame(Japan_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Japan'}))

Japan_confirmed_fin.reset_index(drop = True, inplace = True)
India_confirmed = df_table[(df_table['Country_Region'] == 'India') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

India_confirmed.sort_values('Confirmed',inplace = True)

India_confirmed['Day'] = India_confirmed['Confirmed'].rank(method = 'first')

India_confirmed_fin = pd.DataFrame(India_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_India'}))

India_confirmed_fin.reset_index(drop = True, inplace = True)
Germany_confirmed = df_table[(df_table['Country_Region'] == 'Germany') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Germany_confirmed.sort_values('Confirmed',inplace = True)

Germany_confirmed['Day'] = Germany_confirmed['Confirmed'].rank(method = 'first')

Germany_confirmed_fin = pd.DataFrame(Germany_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Germany'}))

Germany_confirmed_fin.reset_index(drop = True, inplace = True)
Belgium_confirmed = df_table[(df_table['Country_Region'] == 'Belgium') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Belgium_confirmed.sort_values('Confirmed',inplace = True)

Belgium_confirmed['Day'] = Belgium_confirmed['Confirmed'].rank(method = 'first')

Belgium_confirmed_fin = pd.DataFrame(Belgium_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Belgium'}))

Belgium_confirmed_fin.reset_index(drop = True, inplace = True)
United_Kingdom_confirmed = df_table[(df_table['Country_Region'] == 'United Kingdom') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

United_Kingdom_confirmed.sort_values('Confirmed',inplace = True)

United_Kingdom_confirmed['Day'] = United_Kingdom_confirmed['Confirmed'].rank(method = 'first')

United_Kingdom_confirmed_fin = pd.DataFrame(United_Kingdom_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_United_Kingdom'}))

United_Kingdom_confirmed_fin.reset_index(drop = True, inplace = True)
Austria_confirmed = df_table[(df_table['Country_Region'] == 'Austria') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Austria_confirmed.sort_values('Confirmed',inplace = True)

Austria_confirmed['Day'] = Austria_confirmed['Confirmed'].rank(method = 'first')

Austria_confirmed_fin = pd.DataFrame(Austria_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Austria'}))

Austria_confirmed_fin.reset_index(drop = True, inplace = True)
Switzerland_confirmed = df_table[(df_table['Country_Region'] == 'Switzerland') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Switzerland_confirmed.sort_values('Confirmed',inplace = True)

Switzerland_confirmed['Day'] = Switzerland_confirmed['Confirmed'].rank(method = 'first')

Switzerland_confirmed_fin = pd.DataFrame(Switzerland_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Switzerland'}))

Switzerland_confirmed_fin.reset_index(drop = True, inplace = True)
Peru_confirmed = df_table[(df_table['Country_Region'] == 'Peru') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Peru_confirmed.sort_values('Confirmed',inplace = True)

Peru_confirmed['Day'] = Peru_confirmed['Confirmed'].rank(method = 'first')

Peru_confirmed_fin = pd.DataFrame(Peru_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Peru'}))

Peru_confirmed_fin.reset_index(drop = True, inplace = True)
Chile_confirmed = df_table[(df_table['Country_Region'] == 'Chile') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Chile_confirmed.sort_values('Confirmed',inplace = True)

Chile_confirmed['Day'] = Chile_confirmed['Confirmed'].rank(method = 'first')

Chile_confirmed_fin = pd.DataFrame(Chile_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Chile'}))

Chile_confirmed_fin.reset_index(drop = True, inplace = True)
South_Korea_confirmed = df_table[(df_table['Country_Region'] == 'Korea, South') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

South_Korea_confirmed.sort_values('Confirmed',inplace = True)

South_Korea_confirmed['Day'] = South_Korea_confirmed['Confirmed'].rank(method = 'first')

South_Korea_confirmed_fin = pd.DataFrame(South_Korea_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_South_Korea'}))

South_Korea_confirmed_fin.reset_index(drop = True, inplace = True)
Brazil_confirmed = df_table[(df_table['Country_Region'] == 'Brazil') & (df_table['Confirmed'] > 0)][['Country_Region','Confirmed','Last_Update','new_date1']]

Brazil_confirmed.sort_values('Confirmed',inplace = True)

Brazil_confirmed['Day'] = Brazil_confirmed['Confirmed'].rank(method = 'first')

Brazil_confirmed_fin = pd.DataFrame(Brazil_confirmed[['Day','Confirmed']].rename(columns = {'Confirmed':'Confirmed_Brazil'}))
data_frames = [US_confirmed_fin, Canada_confirmed_fin, Italy_confirmed_fin,Spain_confirmed_fin,China_confirmed_fin,Japan_confirmed_fin,

              India_confirmed_fin,Germany_confirmed_fin,Belgium_confirmed_fin,United_Kingdom_confirmed_fin,

               Austria_confirmed_fin,Switzerland_confirmed_fin,Peru_confirmed_fin,Chile_confirmed_fin,South_Korea_confirmed_fin,

              Brazil_confirmed_fin]
from functools import reduce

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Day'],

                                            how='outer'), data_frames)
df_merged.head()
#df_merged1 = df_merged[df_merged['Day'] <= 42]



fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Americas - Cases comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Confirmed Cases",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(df_merged['Confirmed_US'], label = 'US' )

plt.plot(df_merged['Confirmed_Canada'], label = 'Canada')

plt.plot(df_merged['Confirmed_Peru'], label = 'Peru')

plt.plot(df_merged['Confirmed_Chile'], label = 'Chile')

plt.plot(df_merged['Confirmed_Brazil'], label = 'Brazil')





plt.legend(loc = "upper left")



plt.yscale('log')
fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Europe - Cases comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Confirmed Cases",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(df_merged['Confirmed_Italy'],label = 'Italy' )

plt.plot(df_merged['Confirmed_Spain'],label = 'Spain' )

plt.plot(df_merged['Confirmed_Germany'],label = 'Germany' )

plt.plot(df_merged['Confirmed_Belgium'],label = 'Belgium' )

plt.plot(df_merged['Confirmed_United_Kingdom'],label = 'UK' )

plt.plot(df_merged['Confirmed_Austria'],label = 'Austria' )

plt.plot(df_merged['Confirmed_Switzerland'],label = 'Switzerland' )





plt.legend(loc = "upper left")



plt.yscale('log')
fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Asia - Cases comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Confirmed Cases",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(df_merged['Confirmed_China'],label = 'China' )

plt.plot(df_merged['Confirmed_South_Korea'], label = 'South Korea')

plt.plot(df_merged['Confirmed_India'],label = 'India' )

plt.plot(df_merged['Confirmed_Japan'],label = 'Japan' )





plt.legend(loc = "upper left")



plt.yscale('log')
from scipy.stats import linregress

from math import log2, ceil, sqrt
#https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/



US_confirmed2 = US_confirmed1[US_confirmed1['Confirmed'] >= 100]





df = US_confirmed2[['Last_Update','Confirmed']].rename(columns = {'Last_Update':'Date','Confirmed':'Count'})



df.reset_index(drop = True, inplace = True)



# Extend the table with base-2 log of the case count and the day number

df['Log2Count'] = np.log2(df.Count)

firstDate = df.Date[0]

df['Days'] = (df.Date - firstDate).dt.days



# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'US COVID-19 growth'

plt.title(title)

plt.ylabel('Confirmed case count')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
#https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/



Canada_confirmed1 = Canada_confirmed[Canada_confirmed['Confirmed'] >= 100]



df = Canada_confirmed1[['Last_Update','Confirmed']].rename(columns = {'Last_Update':'Date','Confirmed':'Count'})





df.reset_index(drop = True, inplace = True)







# Extend the table with base-2 log of the case count and the day number

df['Log2Count'] = np.log2(df.Count)

firstDate = df.Date[0]

df['Days'] = (df.Date - firstDate).dt.days



# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, '--', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Canada COVID-19 growth'

plt.title(title)

plt.ylabel('Confirmed case count')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
#https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/



India_confirmed1 = India_confirmed[India_confirmed['Confirmed'] >= 100]



df = India_confirmed1[['Last_Update','Confirmed']].rename(columns = {'Last_Update':'Date','Confirmed':'Count'})





df.reset_index(drop = True, inplace = True)







# Extend the table with base-2 log of the case count and the day number

df['Log2Count'] = np.log2(df.Count)

firstDate = df.Date[0]

df['Days'] = (df.Date - firstDate).dt.days



# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, '--', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'India COVID-19 growth'

plt.title(title)

plt.ylabel('Confirmed case count')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
#https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/



Italy_confirmed1 = Italy_confirmed[Italy_confirmed['Confirmed'] >= 100]



df = Italy_confirmed1[['Last_Update','Confirmed']].rename(columns = {'Last_Update':'Date','Confirmed':'Count'})





df.reset_index(drop = True, inplace = True)







# Extend the table with base-2 log of the case count and the day number

df['Log2Count'] = np.log2(df.Count)

firstDate = df.Date[0]

df['Days'] = (df.Date - firstDate).dt.days



# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, '--', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Italy COVID-19 growth'

plt.title(title)

plt.ylabel('Confirmed case count')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
#https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/



Spain_confirmed1 = Spain_confirmed[Spain_confirmed['Confirmed'] >= 100]



df = Spain_confirmed1[['Last_Update','Confirmed']].rename(columns = {'Last_Update':'Date','Confirmed':'Count'})





df.reset_index(drop = True, inplace = True)







# Extend the table with base-2 log of the case count and the day number

df['Log2Count'] = np.log2(df.Count)

firstDate = df.Date[0]

df['Days'] = (df.Date - firstDate).dt.days



# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, '--', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Spain COVID-19 growth'

plt.title(title)

plt.ylabel('Confirmed case count')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
df = df_merged
df.apply(lambda x: linregress(df.Day, x), result_type='expand').rename(index={0: 'slope', 1: 

                                                                                  'intercept', 2: 'rvalue', 3:

                                                                                  'p-value', 4:'stderr'})
cols=list(df.columns)
del cols[0]
np.array(df[df['Confirmed_Canada']>100]['Confirmed_Canada'])

new_df=pd.DataFrame(np.array(df[df['Confirmed_Canada']>100]['Confirmed_Canada']),columns=['Confirmed_Canada'])
new_df.shape
country=[]

slope=[]

i=cols[0]

booleanC = True

newVlaye= 0.0

if df[df[i]>100][i].count()>0:

  new_df=pd.DataFrame(np.array(df[df[i]>100][i]),columns=[i])

  country.append(i)

  # if booleanC:

  newVlaye = new_df.apply(lambda x: 1/(linregress(new_df.index, np.array(np.log2(x))).slope))

  print(newVlaye.values)

    # booleanC = False

  # print(new_df.apply(lambda x: 1/(linregress(new_df.index, np.array(np.log2(x))).slope)))

#del new_df
country=[]

slope=[]

for i in cols:

  if df[df[i]>100][i].count()>0:

    new_df=pd.DataFrame(np.array(df[df[i]>100][i]),columns=[i])

    country.append(i)

    slope.append(new_df.apply(lambda x: 1/(linregress(new_df.index, np.array(np.log2(x))).slope)).values)

    del new_df
new_slope=[]

for i in range(0,len(slope)):

  new_slope.append(slope[i][0])
ziplist=list(zip(country,new_slope))
df1=pd.DataFrame(ziplist,columns=['country','slope'])

df1['new_country'] = df1['country'].str.split(pat = '_',expand = True)[1]

df1['new_country'] = df1.new_country.apply(lambda x : 'UK' if x == 'United' else x)

df1['new_country'] = df1.new_country.apply(lambda x : 'South_Korea' if x == 'South' else x)

df1['slope'] = round(df1['slope'],2)
df1.sort_values(by = 'slope', ascending = True,inplace = True)

plt.figure(figsize=(16,8))

bar = sns.barplot(x='new_country',y='slope',data = df1[(df1['country'] != 'Confirmed_China')

                                                          & (df1['country'] != 'Confirmed_South_Korea')], palette='OrRd_r')

plt.yticks(np.arange(0,10,0.5))

#bar.set_xticklabels(bar.get_xticklabels(), rotation = 90)

plt.ylabel("Doubling Rate (in days)",fontsize = 14)

plt.xlabel("Countries", fontsize = 14,)

plt.title('Doubling rate across countries',fontsize=20)
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_deaths1 = df_deaths.drop(['Province/State','Lat','Long'],axis = 1)
df_deaths1.head()
d_new_sum_row = list(df_deaths1.sum(axis=0))

d_new_sum_row[0] = 'Total'

df_deaths2 = df_deaths1.append(pd.Series(d_new_sum_row, index=df_deaths1.columns ), ignore_index=True)

d_tran = df_deaths2.set_index('Country/Region').transpose()

d_tran.reset_index(inplace = True)

d_tran.rename(columns = {'index':'Date'}, inplace = True) 

d_tran['new_date'] = pd.to_datetime(d_tran['Date'])

d_tran['new_date1']=d_tran['new_date'].apply(lambda x: x.strftime("%d %b"))

# there are duplicate columns for country

d_tran1 = d_tran.groupby(d_tran.columns, axis=1).sum()
fig, ax = plt.subplots(figsize=(12, 8))

style = dict(size=12, color='gray')

plt.plot(d_tran1['new_date1'],d_tran1['Total'],color='red',marker = 'o')

plt.xticks(rotation=45)

ax.grid(False)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

plt.yticks(np.arange(0, d_tran1['Total'].max(),20000))

#ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

#ax.xaxis.set_minor_formatter(mticker.NullFormatter())

ax.text('22 Mar',30000, "Exponential increase started", **style,ha='right',va = 'top')

plt.title('Increase in Number of Deaths Worldwide',fontsize=20)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))





axes[0,0].tick_params('x', labelrotation=45)

axes[0,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,0].plot(d_tran1['new_date1'],d_tran1['Italy'],color='red',marker = 'o')

axes[0,0].set_title('Increase in Number of Deaths in Italy',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[0,1].tick_params('x', labelrotation=45)

axes[0,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,1].plot(d_tran1['new_date1'],d_tran1['Spain'],color='red',marker = 'o')

axes[0,1].set_title('Increase in Number of Deaths in Spain',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



axes[1,0].tick_params('x', labelrotation=45)

axes[1,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,0].plot(d_tran1['new_date1'],d_tran1['United Kingdom'],color='red',marker = 'o')

axes[1,0].set_title('Increase in Number of Deaths in UK',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)





axes[1,1].tick_params('x', labelrotation=45)

axes[1,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,1].plot(d_tran1['new_date1'],d_tran1['Germany'],color='red',marker = 'o')

axes[1,1].set_title('Increase in Number of Deaths in Germany',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



#ax.text('17 Mar',300000, "Exponential increase started", **style,ha='right',va = 'top')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))





axes[0,0].tick_params('x', labelrotation=45)

axes[0,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,0].plot(d_tran1['new_date1'],d_tran1['US'],color='red',marker = 'o')

axes[0,0].set_title('Increase in Number of Deaths in US',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[0,1].tick_params('x', labelrotation=45)

axes[0,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0,1].plot(d_tran1['new_date1'],d_tran1['Canada'],color='red',marker = 'o')

axes[0,1].set_title('Increase in Number of Deaths in Canada',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)



axes[1,0].tick_params('x', labelrotation=45)

axes[1,0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,0].plot(d_tran1['new_date1'],d_tran1['Chile'],color='red',marker = 'o')

axes[1,0].set_title('Increase in Number of Deaths in Chile',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)





axes[1,1].tick_params('x', labelrotation=45)

axes[1,1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1,1].plot(d_tran1['new_date1'],d_tran1['Brazil'],color='red',marker = 'o')

axes[1,1].set_title('Increase in Number of Deaths in Brazil',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)
fig, ax = plt.subplots(figsize=(12, 8))



style = dict(size=12, color='gray')

plt.plot(d_tran1['new_date1'],d_tran1['India'],color='red',marker = 'o')

plt.xticks(rotation=45)

#ax.grid(False)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

#plt.yticks(np.arange(0, tran1['Total'].max(),200000))

#ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

#ax.xaxis.set_minor_formatter(mticker.NullFormatter())

#ax.text('17 Mar',300000, "Exponential increase started", **style,ha='right',va = 'top')

plt.title('Increase in Number of Deaths in India',fontsize=20)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))





axes[0].tick_params('x', labelrotation=45)

axes[0].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[0].plot(d_tran1['new_date1'],d_tran1['China'],color='red',marker = 'o')

axes[0].set_title('Increase in Number of Deaths in China',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,0].grid(False)





axes[1].tick_params('x', labelrotation=45)

axes[1].xaxis.set_major_locator(mticker.MultipleLocator(5))

axes[1].plot(d_tran1['new_date1'],d_tran1['Korea, South'],color='red',marker = 'o')

axes[1].set_title('Increase in Number of Deaths in Sount Korea',fontsize=12)

#axes[0,0].set_yticks(np.arange(0, tran['Italy'].max(),20000))

#axes[0,1].grid(False)
d_US_confirmed = d_tran1[d_tran1['US'] > 0][['US','new_date1','Date']]

d_US_confirmed.sort_values('US',inplace = True)

d_US_confirmed['Day'] = d_US_confirmed['US'].rank(method = 'first')

d_US_confirmed.reset_index(drop = True,inplace = True)

d_US_confirmed_fin = d_US_confirmed[['Day','US']]
d_Canada_confirmed = d_tran1[d_tran1['Canada'] > 0][['Canada','new_date1','Date']]

d_Canada_confirmed.sort_values('Canada',inplace = True)

d_Canada_confirmed['Day'] = d_Canada_confirmed['Canada'].rank(method = 'first')

d_Canada_confirmed.reset_index(drop = True,inplace = True)

d_Canada_confirmed_fin = d_Canada_confirmed[['Day','Canada']]
d_Italy_confirmed = d_tran1[d_tran1['Italy'] > 0][['Italy','new_date1','Date']]

d_Italy_confirmed.sort_values('Italy',inplace = True)

d_Italy_confirmed['Day'] = d_Italy_confirmed['Italy'].rank(method = 'first')

d_Italy_confirmed.reset_index(drop = True,inplace = True)

d_Italy_confirmed_fin = d_Italy_confirmed[['Day','Italy']]
d_Spain_confirmed = d_tran1[d_tran1['Spain'] > 0][['Spain','new_date1','Date']]

d_Spain_confirmed.sort_values('Spain',inplace = True)

d_Spain_confirmed['Day'] = d_Spain_confirmed['Spain'].rank(method = 'first')

d_Spain_confirmed.reset_index(drop = True,inplace = True)

d_Spain_confirmed_fin = d_Spain_confirmed[['Day','Spain']]
d_UK_confirmed = d_tran1[d_tran1['United Kingdom'] > 0][['United Kingdom','new_date1','Date']]

d_UK_confirmed.sort_values('United Kingdom',inplace = True)

d_UK_confirmed['Day'] = d_UK_confirmed['United Kingdom'].rank(method = 'first')

d_UK_confirmed.reset_index(drop = True,inplace = True)

d_UK_confirmed_fin = d_UK_confirmed[['Day','United Kingdom']]
d_Germany_confirmed = d_tran1[d_tran1['Germany'] > 0][['Germany','new_date1','Date']]

d_Germany_confirmed.sort_values('Germany',inplace = True)

d_Germany_confirmed['Day'] = d_Germany_confirmed['Germany'].rank(method = 'first')

d_Germany_confirmed.reset_index(drop = True,inplace = True)

d_Germany_confirmed_fin = d_Germany_confirmed[['Day','Germany']]
d_India_confirmed = d_tran1[d_tran1['India'] > 0][['India','new_date1','Date']]

d_India_confirmed.sort_values('India',inplace = True)

d_India_confirmed['Day'] = d_India_confirmed['India'].rank(method = 'first')

d_India_confirmed.reset_index(drop = True,inplace = True)

d_India_confirmed_fin = d_India_confirmed[['Day','India']]
d_Japan_confirmed = d_tran1[d_tran1['Japan'] > 0][['Japan','new_date1','Date']]

d_Japan_confirmed.sort_values('Japan',inplace = True)

d_Japan_confirmed['Day'] = d_Japan_confirmed['Japan'].rank(method = 'first')

d_Japan_confirmed.reset_index(drop = True,inplace = True)

d_Japan_confirmed_fin = d_Japan_confirmed[['Day','Japan']]
d_Belgium_confirmed = d_tran1[d_tran1['Belgium'] > 0][['Belgium','new_date1','Date']]

d_Belgium_confirmed.sort_values('Belgium',inplace = True)

d_Belgium_confirmed['Day'] = d_Belgium_confirmed['Belgium'].rank(method = 'first')

d_Belgium_confirmed.reset_index(drop = True,inplace = True)

d_Belgium_confirmed_fin = d_Belgium_confirmed[['Day','Belgium']]
d_Switzerland_confirmed = d_tran1[d_tran1['Switzerland'] > 0][['Switzerland','new_date1','Date']]

d_Switzerland_confirmed.sort_values('Switzerland',inplace = True)

d_Switzerland_confirmed['Day'] = d_Switzerland_confirmed['Switzerland'].rank(method = 'first')

d_Switzerland_confirmed.reset_index(drop = True,inplace = True)

d_Switzerland_confirmed_fin = d_Switzerland_confirmed[['Day','Switzerland']]
d_China_confirmed = d_tran1[d_tran1['China'] > 0][['China','new_date1','Date']]

d_China_confirmed.sort_values('China',inplace = True)

d_China_confirmed['Day'] = d_China_confirmed['China'].rank(method = 'first')

d_China_confirmed.reset_index(drop = True,inplace = True)

d_China_confirmed_fin = d_China_confirmed[['Day','China']]
d_Brazil_confirmed = d_tran1[d_tran1['Brazil'] > 0][['Brazil','new_date1','Date']]

d_Brazil_confirmed.sort_values('Brazil',inplace = True)

d_Brazil_confirmed['Day'] = d_Brazil_confirmed['Brazil'].rank(method = 'first')

d_Brazil_confirmed.reset_index(drop = True,inplace = True)

d_Brazil_confirmed_fin = d_Brazil_confirmed[['Day','Brazil']]
d_Peru_confirmed = d_tran1[d_tran1['Peru'] > 0][['Peru','new_date1','Date']]

d_Peru_confirmed.sort_values('Peru',inplace = True)

d_Peru_confirmed['Day'] = d_Peru_confirmed['Peru'].rank(method = 'first')

d_Peru_confirmed.reset_index(drop = True,inplace = True)

d_Peru_confirmed_fin = d_Peru_confirmed[['Day','Peru']]
d_Chile_confirmed = d_tran1[d_tran1['Chile'] > 0][['Chile','new_date1','Date']]

d_Chile_confirmed.sort_values('Chile',inplace = True)

d_Chile_confirmed['Day'] = d_Chile_confirmed['Chile'].rank(method = 'first')

d_Chile_confirmed.reset_index(drop = True,inplace = True)

d_Chile_confirmed_fin = d_Chile_confirmed[['Day','Chile']]
d_South_Korea_confirmed = d_tran1[d_tran1['Korea, South'] > 0][['Korea, South','new_date1','Date']]

d_South_Korea_confirmed.sort_values('Korea, South',inplace = True)

d_South_Korea_confirmed['Day'] = d_South_Korea_confirmed['Korea, South'].rank(method = 'first')

d_South_Korea_confirmed.reset_index(drop = True,inplace = True)

d_South_Korea_confirmed_fin = d_South_Korea_confirmed[['Day','Korea, South']]
d_Austria_confirmed = d_tran1[d_tran1['Austria'] > 0][['Austria','new_date1','Date']]

d_Austria_confirmed.sort_values('Austria',inplace = True)

d_Austria_confirmed['Day'] = d_Austria_confirmed['Austria'].rank(method = 'first')

d_Austria_confirmed.reset_index(drop = True,inplace = True)

d_Austria_confirmed_fin = d_Austria_confirmed[['Day','Austria']]
d_data_frames = [d_US_confirmed_fin, d_Canada_confirmed_fin, d_Italy_confirmed_fin,d_Spain_confirmed_fin,d_China_confirmed_fin,d_Japan_confirmed_fin,

              d_India_confirmed_fin,d_Germany_confirmed_fin,d_Belgium_confirmed_fin,d_UK_confirmed_fin,

              d_Austria_confirmed_fin,d_Switzerland_confirmed_fin,d_Peru_confirmed_fin,d_Chile_confirmed_fin,d_South_Korea_confirmed_fin,

              d_Brazil_confirmed_fin]
from functools import reduce

d_df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Day'],

                                            how='outer'), d_data_frames)
d_df_merged.head()
#df_merged1 = df_merged[df_merged['Day'] <= 42]



fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Americas - Deaths comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Deaths",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(d_df_merged['US'], label = 'US' )

plt.plot(d_df_merged['Canada'], label = 'Canada')

plt.plot(d_df_merged['Peru'], label = 'Peru')

plt.plot(d_df_merged['Chile'], label = 'Chile')

plt.plot(d_df_merged['Brazil'], label = 'Brazil')





plt.legend(loc = "upper left")



plt.yscale('log')
#df_merged1 = df_merged[df_merged['Day'] <= 42]



fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Europe - Deaths comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Deaths",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(d_df_merged['Spain'], label = 'Spain' )

plt.plot(d_df_merged['Italy'], label = 'Italy')

plt.plot(d_df_merged['United Kingdom'], label = 'UK')

plt.plot(d_df_merged['Germany'], label = 'Germany')

plt.plot(d_df_merged['Belgium'], label = 'Belgium')

plt.plot(d_df_merged['Austria'], label = 'Austria')

plt.plot(d_df_merged['Switzerland'], label = 'Switzerland')



plt.legend(loc = "upper left")



plt.yscale('log')
#df_merged1 = df_merged[df_merged['Day'] <= 42]



fig, ax = plt.subplots(figsize=(14, 10))



plt.title('Asia - Deaths comparison for different countries',fontsize=20)

plt.xlabel("Days",fontsize=14)

plt.ylabel("Number of Deaths",fontsize=14)

plt.grid(True)

ax.xaxis.set_major_locator(mticker.MultipleLocator(5))





plt.plot(d_df_merged['China'], label = 'China' )

plt.plot(d_df_merged['India'], label = 'India')

plt.plot(d_df_merged['Japan'], label = 'Japan')

plt.plot(d_df_merged['Korea, South'], label = 'South Korea')





plt.legend(loc = "upper left")



plt.yscale('log')
d_US_confirmed2 = d_US_confirmed[d_US_confirmed['US'] >= 100]



df = d_US_confirmed2[['new_date1','Date','US']].rename(columns = {'US':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'US COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
d_Canada_confirmed2 = d_Canada_confirmed[d_Canada_confirmed['Canada'] >= 100]



df = d_Canada_confirmed2[['new_date1','Date','Canada']].rename(columns = {'Canada':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Canada COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
d_India_confirmed2 = d_India_confirmed[d_India_confirmed['India'] >= 100]



df = d_India_confirmed2[['new_date1','Date','India']].rename(columns = {'India':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'India COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-0.65), dText, fontsize=11)
d_Spain_confirmed2 = d_Spain_confirmed[d_Spain_confirmed['Spain'] >= 100]



df = d_Spain_confirmed2[['new_date1','Date','Spain']].rename(columns = {'Spain':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Spain COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
d_Italy_confirmed2 = d_Italy_confirmed[d_Italy_confirmed['Italy'] >= 100]



df = d_Italy_confirmed2[['new_date1','Date','Italy']].rename(columns = {'Italy':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Italy COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
d_Germany_confirmed2 = d_Germany_confirmed[d_Germany_confirmed['Germany'] >= 100]



df = d_Germany_confirmed2[['new_date1','Date','Germany']].rename(columns = {'Germany':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'Germany COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)
d_UK_confirmed2 = d_UK_confirmed[d_UK_confirmed['United Kingdom'] >= 100]



df = d_UK_confirmed2[['new_date1','Date','United Kingdom']].rename(columns = {'United Kingdom':'Count'})



df.reset_index(drop = True, inplace = True)



df['Log2Count'] = np.log2(df.Count)



df['Date'] = pd.to_datetime(df['Date'])



firstDate = df.Date[0]



df['Days'] = (df.Date - firstDate).dt.days





# Linear regression of Log2Count on Days

lr = linregress(df.Days, df.Log2Count)

df['Fit'] = 2**(lr.intercept + lr.slope*df.Days)



# Doubling time

doubling = 1/lr.slope

dText = f'Count doubles every {doubling:.1f} days'





# Plot the data and the fit

fig, ax = plt.subplots(figsize=(12, 8))

plt.yscale('log', basey=2)

ax.plot(df.Days, df.Count, 'o', color='#d95f02', lw=2)

ax.plot(df.Days, df.Fit, '--', color='#7570b3', lw=2)





# Ticks and grid

yStart = 100

coordMax = log2(df.Count.max()/yStart)

expMax = ceil(coordMax)

yAdd = .4142*yStart*2**expMax if expMax - coordMax < .1 else 0

plt.ylim(ymin=yStart, ymax=yStart*2**expMax + yAdd)

majors = np.array([ yStart*2**i for i in range(expMax+1) ])

ax.set_yticks(majors)

ax.set_yticklabels(majors)





dateTickFreq = int(doubling)

dMax = df.Days.max()

#xAdd = 2 if df.Days.max() % dateTickFreq else 1



plt.xlim(xmin=-1, xmax=dMax + 1)

#ax.set_xticks([ x for x in range(0, dMax+xAdd, dateTickFreq) ])

#dates = [ (firstDate.date() + timedelta(days=x)).strftime('%b %-d') for x in range(0, dMax+xAdd, dateTickFreq) ]



#ax.set_xticklabels(dates)





ax.grid(linewidth=.5, which='major', color='#dddddd', linestyle='-')







# Title and labels

title = 'UK COVID-19 growth'

plt.title(title)

plt.ylabel('Number of Deaths')





# Add doubling text

plt.text(1, yStart*2**(expMax-1.65), dText, fontsize=11)