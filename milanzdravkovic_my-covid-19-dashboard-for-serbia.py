import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
dfconfirmed=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

dfdeaths=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

dfrecovered=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
import collections



def stripZeros(df):

    count=0

    for x in df.values[0]:

        count+=1

        if(x!=0):

            return count-1, df.iloc[:,count-1:]



dfcountry= collections.defaultdict(dict)



dfcountry['rs']['population']=6963764

dfcountry['bg']['population']=6951482

dfcountry['ro']['population']=19405156

dfcountry['hr']['population']=4076246

dfcountry['si']['population']=2094060

dfcountry['it']['population']=60238522

dfcountry['gr']['population']=10724599

dfcountry['se']['population']=10338368

dfcountry['es']['population']=47100396

dfcountry['de']['population']=83149300

dfcountry['sk']['population']=5456362



dfcountry['rs']['day'], dfcountry['rs']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Serbia'].iloc[:,4:])

dfcountry['bg']['day'], dfcountry['bg']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Bulgaria'].iloc[:,4:])

dfcountry['ro']['day'], dfcountry['ro']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Romania'].iloc[:,4:])

dfcountry['hr']['day'], dfcountry['hr']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Croatia'].iloc[:,4:])

dfcountry['si']['day'], dfcountry['si']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Slovenia'].iloc[:,4:])

dfcountry['it']['day'], dfcountry['it']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Italy'].iloc[:,4:])

dfcountry['gr']['day'], dfcountry['gr']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Greece'].iloc[:,4:])

dfcountry['se']['day'], dfcountry['se']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Sweden'].iloc[:,4:])

dfcountry['es']['day'], dfcountry['es']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Spain'].iloc[:,4:])

dfcountry['de']['day'], dfcountry['de']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Germany'].iloc[:,4:])

dfcountry['sk']['day'], dfcountry['sk']['confirmed']=stripZeros(dfconfirmed.loc[dfconfirmed['Country/Region']=='Slovakia'].iloc[:,4:])



dfcountry['rs']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Serbia'].iloc[:,4+dfcountry['rs']['day']:]

dfcountry['bg']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Bulgaria'].iloc[:,4+dfcountry['bg']['day']:]

dfcountry['ro']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Romania'].iloc[:,4+dfcountry['ro']['day']:]

dfcountry['hr']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Croatia'].iloc[:,4+dfcountry['hr']['day']:]

dfcountry['si']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Slovenia'].iloc[:,4+dfcountry['si']['day']:]

dfcountry['it']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Italy'].iloc[:,4+dfcountry['it']['day']:]

dfcountry['gr']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Greece'].iloc[:,4+dfcountry['gr']['day']:]

dfcountry['se']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Sweden'].iloc[:,4+dfcountry['se']['day']:]

dfcountry['es']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Spain'].iloc[:,4+dfcountry['es']['day']:]

dfcountry['de']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Germany'].iloc[:,4+dfcountry['de']['day']:]

dfcountry['sk']['deaths']=dfdeaths.loc[dfdeaths['Country/Region']=='Germany'].iloc[:,4+dfcountry['sk']['day']:]



dfcountry['rs']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Serbia'].iloc[:,4+dfcountry['rs']['day']:]

dfcountry['bg']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Bulgaria'].iloc[:,4+dfcountry['bg']['day']:]

dfcountry['ro']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Romania'].iloc[:,4+dfcountry['ro']['day']:]

dfcountry['hr']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Croatia'].iloc[:,4+dfcountry['hr']['day']:]

dfcountry['si']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Slovenia'].iloc[:,4+dfcountry['si']['day']:]

dfcountry['it']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Italy'].iloc[:,4+dfcountry['it']['day']:]

dfcountry['gr']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Greece'].iloc[:,4+dfcountry['gr']['day']:]

dfcountry['se']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Sweden'].iloc[:,4+dfcountry['se']['day']:]

dfcountry['es']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Spain'].iloc[:,4+dfcountry['es']['day']:]

dfcountry['de']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Germany'].iloc[:,4+dfcountry['de']['day']:]

dfcountry['sk']['recovered']=dfrecovered.loc[dfrecovered['Country/Region']=='Germany'].iloc[:,4+dfcountry['sk']['day']:]
dfstemp=pd.read_excel('https://data.gov.rs/sr/datasets/r/a57afd99-85d5-4f8e-9740-5363e6ceb40e')

dfstemp.head(15)
dfsarr=dfstemp['Vrednost'].values

dfsarr=dfsarr.reshape(-1,17)

dfs=pd.DataFrame(data=dfsarr, columns=['RESP_TOTAL','HOSP_TOTAL','POZITOVNO_DAN','POZ_UKUPNO','TEST_DAN','TEST_UKUPNO','UMRLI_DAN','UMM_DAN','UMZ_DAN','UMR_UKUPNO','PROS_GOD_DAN','PROC1','PROC2','PROC3','PROC4','IZLECENO_UKUPNO','PROC5'])
plt.rcParams["figure.figsize"] = (14,10)

plt.rcParams.update({'font.size': 16})

plt.yscale('log') 



plt.title('Number of confirmed cases in selected countries since day 1')

plt.plot(dfcountry['rs']['confirmed'].values[0],linewidth=4.0)

plt.plot(dfcountry['bg']['confirmed'].values[0])

plt.plot(dfcountry['hr']['confirmed'].values[0])

plt.plot(dfcountry['si']['confirmed'].values[0])

plt.plot(dfcountry['gr']['confirmed'].values[0])

plt.plot(dfcountry['ro']['confirmed'].values[0])

plt.plot(dfcountry['it']['confirmed'].values[0])

plt.plot(dfcountry['se']['confirmed'].values[0])

plt.plot(dfcountry['es']['confirmed'].values[0])

plt.plot(dfcountry['de']['confirmed'].values[0])

plt.plot(dfcountry['sk']['confirmed'].values[0])

plt.xlabel('Days since first occurence')

plt.ylabel('Number of confirmed cases (log)')

plt.legend(['RS','BG','HR','SI','GR','RO','IT','SE','ES','DE','SK'])

plt.grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
plt.rcParams["figure.figsize"] = (14,10)

plt.rcParams.update({'font.size': 16})



fig, ax = plt.subplots(2, 2, figsize=(17, 10))

plt.subplots_adjust(wspace=0.2, hspace=0.8)



ax[0,0].set_title('Number of recovered confirmed cases\n in selected countries (region) since day 1')

ax[0,0].plot(dfcountry['rs']['recovered'].values[0],linewidth=4.0)

ax[0,0].plot(dfcountry['bg']['recovered'].values[0])

ax[0,0].plot(dfcountry['hr']['recovered'].values[0])

ax[0,0].plot(dfcountry['si']['recovered'].values[0])

ax[0,0].plot(dfcountry['gr']['recovered'].values[0])

ax[0,0].plot(dfcountry['ro']['recovered'].values[0])

ax[0,0].set_xlabel('Days since first occurence')

ax[0,0].set_ylabel('Number of recovered cases')

ax[0,0].legend(['RS','BG','HR','SI','GR','RO'])

ax[0,0].grid(color='gray', ls = '-.', lw = 0.2)





ax[0,1].set_title('Number of recovered confirmed cases\n in selected countries since day 1')

ax[0,1].plot(dfcountry['it']['recovered'].values[0])

ax[0,1].plot(dfcountry['se']['recovered'].values[0])

ax[0,1].plot(dfcountry['es']['recovered'].values[0])

ax[0,1].plot(dfcountry['de']['recovered'].values[0])

ax[0,1].set_xlabel('Days since first occurence')

ax[0,1].set_ylabel('Number of recovered cases')

ax[0,1].legend(['IT','SE','ES','DE'])

ax[0,1].grid(color='gray', ls = '-.', lw = 0.2)



ax[1,0].set_title('Number of recovered confirmed cases\n in selected countries (region) since day 1\n per 100,000 population')

ax[1,0].plot(100000*dfcountry['rs']['recovered'].values[0]/dfcountry['rs']['population'],linewidth=4.0)

ax[1,0].plot(100000*dfcountry['bg']['recovered'].values[0]/dfcountry['bg']['population'])

ax[1,0].plot(100000*dfcountry['hr']['recovered'].values[0]/dfcountry['hr']['population'])

ax[1,0].plot(100000*dfcountry['si']['recovered'].values[0]/dfcountry['si']['population'])

ax[1,0].plot(100000*dfcountry['gr']['recovered'].values[0]/dfcountry['gr']['population'])

ax[1,0].plot(100000*dfcountry['ro']['recovered'].values[0]/dfcountry['ro']['population'])

ax[1,0].set_xlabel('Days since first occurence')

ax[1,0].set_ylabel('Number of recovered cases')

ax[1,0].legend(['RS','BG','HR','SI','GR','RO'])

ax[1,0].grid(color='gray', ls = '-.', lw = 0.2)



ax[1,1].set_title('Number of recovered confirmed cases\n in selected countries since day 1\n per 100,000 population')

ax[1,1].plot(100000*dfcountry['it']['recovered'].values[0]/dfcountry['it']['population'])

ax[1,1].plot(100000*dfcountry['se']['recovered'].values[0]/dfcountry['se']['population'])

ax[1,1].plot(100000*dfcountry['es']['recovered'].values[0]/dfcountry['es']['population'])

ax[1,1].plot(100000*dfcountry['de']['recovered'].values[0]/dfcountry['de']['population'])

ax[1,1].set_xlabel('Days since first occurence')

ax[1,1].set_ylabel('Number of recovered cases')

ax[1,1].legend(['IT','SE','ES','DE'])

ax[1,1].grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
plt.rcParams["figure.figsize"] = (14,10)

plt.rcParams.update({'font.size': 16})



fig, ax = plt.subplots(1, 2, figsize=(17, 5))

plt.subplots_adjust(wspace=0.2, hspace=0.8)



ax[0].set_title('Number of deaths per 100,000 population\n of selected countries (region) since day 1')

ax[0].plot(100000*dfcountry['rs']['deaths'].values[0]/dfcountry['rs']['population'],linewidth=4.0)

ax[0].plot(100000*dfcountry['bg']['deaths'].values[0]/dfcountry['bg']['population'])

ax[0].plot(100000*dfcountry['hr']['deaths'].values[0]/dfcountry['hr']['population'])

ax[0].plot(100000*dfcountry['si']['deaths'].values[0]/dfcountry['si']['population'])

ax[0].plot(100000*dfcountry['gr']['deaths'].values[0]/dfcountry['gr']['population'])

ax[0].plot(100000*dfcountry['ro']['deaths'].values[0]/dfcountry['ro']['population'])

ax[0].set_xlabel('Days since first occurence')

ax[0].set_ylabel('Mortality rate')

ax[0].legend(['RS','BG','HR','SI','GR','RO'])

ax[0].grid(color='gray', ls = '-.', lw = 0.2)



ax[1].set_title('Number of deaths per 100,000 population\n of selected countries since day 1')

ax[1].plot(100000*dfcountry['it']['deaths'].values[0]/dfcountry['it']['population'])

ax[1].plot(100000*dfcountry['se']['deaths'].values[0]/dfcountry['se']['population'])

ax[1].plot(100000*dfcountry['es']['deaths'].values[0]/dfcountry['es']['population'])

ax[1].plot(100000*dfcountry['de']['deaths'].values[0]/dfcountry['de']['population'])

ax[1].set_xlabel('Days since first occurence')

ax[1].set_ylabel('Mortality rate')

ax[1].legend(['IT','SE','ES','DE'])

ax[1].grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
plt.rcParams["figure.figsize"] = (14,10)

plt.rcParams.update({'font.size': 16})



plt.title('Rate of number of deaths vs confirmed cases in\n selected countries since day 1 (%)')

plt.plot(100*dfcountry['rs']['deaths'].values[0]/dfcountry['rs']['confirmed'].values[0],linewidth=4.0)

plt.plot(100*dfcountry['bg']['deaths'].values[0]/dfcountry['bg']['confirmed'].values[0])

plt.plot(100*dfcountry['hr']['deaths'].values[0]/dfcountry['hr']['confirmed'].values[0])

plt.plot(100*dfcountry['si']['deaths'].values[0]/dfcountry['si']['confirmed'].values[0])

plt.plot(100*dfcountry['gr']['deaths'].values[0]/dfcountry['gr']['confirmed'].values[0])

plt.plot(100*dfcountry['ro']['deaths'].values[0]/dfcountry['ro']['confirmed'].values[0])

plt.plot(100*dfcountry['it']['deaths'].values[0]/dfcountry['it']['confirmed'].values[0])

plt.plot(100*dfcountry['se']['deaths'].values[0]/dfcountry['se']['confirmed'].values[0])

plt.plot(100*dfcountry['es']['deaths'].values[0]/dfcountry['es']['confirmed'].values[0])

plt.plot(100*dfcountry['de']['deaths'].values[0]/dfcountry['de']['confirmed'].values[0])

plt.xlabel('Days since first occurence')

plt.ylabel('Number of deaths vs confirmed cases (%)')

plt.legend(['RS','BG','HR','SI','GR','RO','IT','SE','ES','DE'])

plt.grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
plt.rcParams.update({'font.size': 13})



fig, ax = plt.subplots(1, 2, figsize=(17, 5))

plt.subplots_adjust(wspace=0.2, hspace=0.8)

s=dfs['HOSP_TOTAL'].values.size



ax[0].plot(dfs['HOSP_TOTAL'])

ax[0].set_title('Number of hospitalized confirmed cases in RS')

ax[0].set_xlabel('Days since first occurence')

ax[1].set_xlabel('Days since first occurence')

ax[0].set_ylabel('Number of patients')

ax[1].set_ylabel('Number of patients')

ax[1].plot(dfs['RESP_TOTAL'])

ax[1].set_title('Number of confirmed cases in ICU in RS')

ax[1].axvspan(s-2, s-1, color='red', alpha=0.3)

ax[1].axvspan(s-3, s-2, color='red', alpha=0.2)

ax[1].axvspan(s-4, s-3, color='red', alpha=0.1)

ax[0].axvspan(s-2, s-1, color='red', alpha=0.3)

ax[0].axvspan(s-3, s-2, color='red', alpha=0.2)

ax[0].axvspan(s-4, s-3, color='red', alpha=0.1)

ax[0].grid(color='gray', ls = '-.', lw = 0.2)

ax[1].grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
s=dfs['POZ_UKUPNO'].values.size

plt.rcParams["figure.figsize"] = (13,9)

plt.rcParams.update({'font.size': 14})

plt.plot(100*dfs['RESP_TOTAL'].values/dfs['HOSP_TOTAL'].values)

plt.plot(100*dfs['RESP_TOTAL'].values/dfs['POZ_UKUPNO'].values)

plt.title('Rate of ICU vs hospitalized/confirmed cases in RS (%)')

plt.xlabel('Days since first occurence')

plt.ylabel('Rate of ICU vs hospitalized vs confirmed cases (%)')

plt.axvspan(s-2, s-1, color='red', alpha=0.3)

plt.axvspan(s-3, s-2, color='red', alpha=0.2)

plt.axvspan(s-4, s-3, color='red', alpha=0.1)

plt.legend(['ICU vs hospitalized','ICU vs confirmed'])

plt.grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
s=dfs['POZ_UKUPNO'].values.size

plt.rcParams["figure.figsize"] = (13,9)

plt.rcParams.update({'font.size': 14})

plt.plot(100*dfs['POZ_UKUPNO'].values/dfs['TEST_UKUPNO'].values)

plt.plot(100*dfs['POZITOVNO_DAN'].values/dfs['TEST_DAN'].values)

plt.title('Rate of confirmed vs tested cases in RS (%)')

plt.xlabel('Days since first occurence')

plt.ylabel('Rate of confirmed vs tested cases (%)')

plt.axvspan(s-2, s-1, color='red', alpha=0.3)

plt.axvspan(s-3, s-2, color='red', alpha=0.2)

plt.axvspan(s-4, s-3, color='red', alpha=0.1)

plt.legend(['Cumulative','Daily'])

plt.grid(color='gray', ls = '-.', lw = 0.2)



plt.show()
dfsztemp=pd.read_excel('https://data.gov.rs/sr/datasets/r/1977833a-36f2-445e-916a-ef51b217d3a0')

dfsztemp.head()
plt.rcParams["figure.figsize"] = (18,8)

plt.rcParams.update({'font.size': 15})



plt.title('Distribution of confirmed cases across country in RS (top 30)')

plt.xlabel('City')

plt.ylabel('Number of confirmed cases')



dfgar=dfsztemp.groupby('Место (МУП)').count().sort_values('Пол', ascending=False)['Пол']

bars = plt.bar(dfgar.index[0:30], dfgar.values[0:30], color='y')

plt.grid(color='gray', ls = '-.', lw = 0.2)

plt.xticks(rotation=90)

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x(), yval + 5, yval, fontsize=12)



plt.show()
dfgop=dfsztemp.groupby('Општина (МУП)').count().sort_values('Пол', ascending=False)['Пол']

dfz = pd.DataFrame({'Mesto':dfgop.index, 'Zarazeni':dfgop.values})

dfo=pd.read_csv('../input/opstine.csv', header=None, names=['Mesto', 'Stanovnika'])

dfcombined = pd.merge(dfz, dfo, on='Mesto', how='inner')

dfcombined['Zarazenipo1000']=round(1000*dfcombined['Zarazeni']/dfcombined['Stanovnika'],1)

del dfcombined['Zarazeni']

del dfcombined['Stanovnika']

dfcombined=dfcombined.sort_values('Zarazenipo1000', ascending=False)

dfcombined.head()
plt.rcParams["figure.figsize"] = (18,8)

plt.rcParams.update({'font.size': 15})



plt.title('Distribution of confirmed cases in municipalities in RS per 1000 capita (2008 census data)')

plt.xlabel('Municipality')

plt.ylabel('Number of confirmed cases per 1000')



bars1 = plt.bar(dfcombined['Mesto'].values[0:30], dfcombined['Zarazenipo1000'].values[0:30], color='r')

plt.grid(color='gray', ls = '-.', lw = 0.2)

plt.xticks(rotation=90)



plt.show()