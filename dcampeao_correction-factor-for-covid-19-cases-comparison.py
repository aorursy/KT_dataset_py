import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data['CurrentCases'] = data['Confirmed']-data['Deaths']-data['Recovered']

data.head()
# Get a few contries to compare

data_fr = data[data['Country/Region']=='France'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_br = data[data['Country/Region']=='Brazil'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_it = data[data['Country/Region']=='Italy'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_ca = data[data['Country/Region']=='Canada'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_jp = data[data['Country/Region']=='Japan'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_us = data[data['Country/Region']=='US'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_gr = data[data['Country/Region']=='Germany'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_uk = data[data['Country/Region']=='UK'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_kr = data[data['Country/Region']=='South Korea'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()

data_sp = data[data['Country/Region']=='Spain'].groupby(['Country/Region','ObservationDate'], as_index=False).sum()
# Adjusting the date of the first case to be the same for all countries

data_fr = data_fr[data_fr.Confirmed != 0]

data_br = data_br[data_br.Confirmed != 0]

data_it = data_it[data_it.Confirmed != 0]

data_ca = data_ca[data_ca.Confirmed != 0]

data_jp = data_jp[data_jp.Confirmed != 0]

data_us = data_us[data_us.Confirmed != 0]

data_gr = data_gr[data_gr.Confirmed != 0]

data_uk = data_uk[data_uk.Confirmed != 0]

data_kr = data_kr[data_kr.Confirmed != 0]

data_sp = data_sp[data_sp.Confirmed != 0]
# Number of tests from https://en.wikipedia.org/wiki/COVID-19_testing as of 25-Mar-2020 (note: some of this numbers are out-of-date.)

tests_fr = 36747 #As of 15-Mar. Too out-of-date. I'll remove France from the analysis

tests_br = 45708

tests_it = 324445

tests_ca = 142154

tests_jp = 24430

tests_us = 472820

tests_gr = 167000

tests_uk = 90436

tests_kr = 364942

tests_sp = 355000



#Population / 1000

pop_br = 209300

pop_it = 60480

pop_ca = 37590

pop_jp = 126800

pop_us = 327200

pop_gr = 82790

pop_uk = 66440

pop_kr = 51470

pop_sp = 46660
# Correction factor. I've used the simplest one: Number of tests per day of disease. 

# We can add population, age, number of hospital beds, etc.

correction_fr = tests_fr/data_fr.size

correction_br = tests_br/data_br.size

correction_it = tests_it/data_it.size

correction_ca = tests_ca/data_ca.size

correction_jp = tests_jp/data_jp.size

correction_us = tests_us/data_us.size

correction_gr = tests_gr/data_gr.size

correction_uk = tests_uk/data_uk.size

correction_kr = tests_kr/data_kr.size

correction_sp = tests_sp/data_sp.size



normalizing_factor = max(correction_br,correction_it,correction_ca,correction_jp,correction_us,correction_gr,correction_uk,correction_kr)
plt.figure(num=None, figsize=(18, 15), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 1)



#plt.plot(data_fr.index,(data_fr['CurrentCases']), label='Cases France')

plt.plot(data_br.index,(data_br['CurrentCases']), label='Cases Brazil')

plt.plot(data_it.index,(data_it['CurrentCases']), label='Cases Italy')

plt.plot(data_ca.index,(data_ca['CurrentCases']), label='Cases Canada')

plt.plot(data_jp.index,(data_jp['CurrentCases']), label='Cases Japan')

plt.plot(data_us.index,(data_us['CurrentCases']), label='Cases US')

plt.plot(data_gr.index,(data_gr['CurrentCases']), label='Cases Germany')

plt.plot(data_uk.index,(data_uk['CurrentCases']), label='Cases UK')

plt.plot(data_kr.index,(data_kr['CurrentCases']), label='Cases Korea')

plt.plot(data_sp.index,(data_sp['CurrentCases']), label='Cases Spain')

plt.grid()

plt.legend(loc='upper left')

plt.title('Without Correction')

plt.xlabel('Days')

plt.ylabel('Cases')

plt.show()



plt.figure(num=None, figsize=(18, 15), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(3, 1, 2)

#plt.plot(data_fr.index,(data_fr['CurrentCases']/correction_fr*normalizing_factor), label='Cases France')

plt.plot(data_br.index,(data_br['CurrentCases']/correction_br*normalizing_factor), label='Cases Brazil')

plt.plot(data_it.index,(data_it['CurrentCases']/correction_it*normalizing_factor), label='Cases Italy')

plt.plot(data_ca.index,(data_ca['CurrentCases']/correction_ca*normalizing_factor), label='Cases Canada')

plt.plot(data_jp.index,(data_jp['CurrentCases']/correction_jp*normalizing_factor), label='Cases Japan')

plt.plot(data_us.index,(data_us['CurrentCases']/correction_us*normalizing_factor), label='Cases US')

plt.plot(data_gr.index,(data_gr['CurrentCases']/correction_gr*normalizing_factor), label='Cases Germany')

plt.plot(data_uk.index,(data_uk['CurrentCases']/correction_uk*normalizing_factor), label='Cases UK')

plt.plot(data_kr.index,(data_kr['CurrentCases']/correction_kr*normalizing_factor), label='Cases Korea')

plt.plot(data_sp.index,(data_sp['CurrentCases']/correction_sp*normalizing_factor), label='Cases Spain')

plt.xticks(rotation='vertical')

plt.grid()

plt.legend(loc='upper left')

plt.title('Reconstructed Cases using Germany as Reference with Correction ')

plt.xlabel('Days')

plt.ylabel('Estimated Cases')

plt.show()
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 1)

#plt.plot(data_fr.index,np.log(data_fr['CurrentCases']), label='Cases France')

plt.plot(data_br.index,np.log(data_br['CurrentCases']), label='Cases Brazil')

plt.plot(data_it.index,np.log(data_it['CurrentCases']), label='Cases Italy')

plt.plot(data_ca.index,np.log(data_ca['CurrentCases']), label='Cases Canada')

plt.plot(data_jp.index,np.log(data_jp['CurrentCases']), label='Cases Japan')

plt.plot(data_us.index,np.log(data_us['CurrentCases']), label='Cases US')

plt.plot(data_gr.index,np.log(data_gr['CurrentCases']), label='Cases Germany')

plt.plot(data_uk.index,np.log(data_uk['CurrentCases']), label='Cases UK')

plt.plot(data_kr.index,np.log(data_kr['CurrentCases']), label='Cases Korea')

plt.plot(data_sp.index,np.log(data_sp['CurrentCases']), label='Cases Spain')

plt.xticks(rotation='vertical')

plt.title('Without Correction')

plt.xlabel('Days')

plt.ylabel('log(Cases)')

plt.grid()

plt.legend()

plt.show()



plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2, 1, 2)

#plt.plot(data_fr.index,np.log(data_fr['CurrentCases']/normalizing_factor), label='Cases France')

plt.plot(data_br.index,np.log(data_br['CurrentCases']/normalizing_factor), label='Cases Brazil')

plt.plot(data_it.index,np.log(data_it['CurrentCases']/normalizing_factor), label='Cases Italy')

plt.plot(data_ca.index,np.log(data_ca['CurrentCases']/normalizing_factor), label='Cases Canada')

plt.plot(data_jp.index,np.log(data_jp['CurrentCases']/normalizing_factor), label='Cases Japan')

plt.plot(data_us.index,np.log(data_us['CurrentCases']/normalizing_factor), label='Cases US')

plt.plot(data_gr.index,np.log(data_gr['CurrentCases']/normalizing_factor), label='Cases Germany')

plt.plot(data_uk.index,np.log(data_uk['CurrentCases']/normalizing_factor), label='Cases UK')

plt.plot(data_kr.index,np.log(data_kr['CurrentCases']/normalizing_factor), label='Cases Korea')

plt.plot(data_sp.index,np.log(data_sp['CurrentCases']/normalizing_factor), label='Cases Spain')

plt.xticks(rotation='vertical')

plt.title('With Correction')

plt.xlabel('Days')

plt.ylabel('Normalized log(Cases)')

plt.grid()

plt.legend()

plt.show()
df_normalized = pd.DataFrame()

df_normalized['Countries'] = ['Brazil','Italy','Canada','Japan','US','Germany','UK','Korea','Spain']

df_normalized['Deaths'] = [data_br['Deaths'].iloc[-1],data_it['Deaths'].iloc[-1],data_ca['Deaths'].iloc[-1],data_jp['Deaths'].iloc[-1],

                           data_us['Deaths'].iloc[-1],data_gr['Deaths'].iloc[-1],data_uk['Deaths'].iloc[-1],data_kr['Deaths'].iloc[-1],

                           data_sp['Deaths'].iloc[-1]]



df_normalized['Official Cases'] = [data_br['Confirmed'].iloc[-1],data_it['Confirmed'].iloc[-1],data_ca['Confirmed'].iloc[-1],

                                   data_jp['Confirmed'].iloc[-1],data_us['Confirmed'].iloc[-1],data_gr['Confirmed'].iloc[-1],

                                   data_uk['Confirmed'].iloc[-1],data_kr['Confirmed'].iloc[-1],data_sp['Confirmed'].iloc[-1]]



df_normalized['Number of Tests'] = [tests_br,tests_it,tests_ca,tests_jp,tests_us,tests_gr,tests_uk,tests_kr,tests_sp]



df_normalized['Tests / 1000 Population'] = [tests_br/pop_br,tests_it/pop_it,tests_ca/pop_ca,tests_jp/pop_jp,tests_us/pop_us,tests_gr/pop_gr,tests_uk/pop_uk,tests_kr/pop_kr,tests_sp/pop_sp]



df_normalized['Normalized Cases'] = [data_br['Confirmed'].iloc[-1]/correction_br*normalizing_factor,data_it['Confirmed'].iloc[-1]/correction_it*normalizing_factor,data_ca['Confirmed'].iloc[-1]/correction_ca*normalizing_factor,

                                     data_jp['Confirmed'].iloc[-1]/correction_jp*normalizing_factor,data_us['Confirmed'].iloc[-1]/correction_it*normalizing_factor,data_gr['Confirmed'].iloc[-1]/correction_gr*normalizing_factor,

                                     data_uk['Confirmed'].iloc[-1]/correction_uk*normalizing_factor,data_kr['Confirmed'].iloc[-1]/correction_kr*normalizing_factor,data_sp['Confirmed'].iloc[-1]/correction_sp*normalizing_factor]



df_normalized['Death / Official Cases'] = df_normalized['Deaths']/df_normalized['Official Cases']*100



df_normalized['Death / Normalized Cases'] = df_normalized['Deaths']/df_normalized['Normalized Cases']*100

df_normalized
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')

plt.plot(df_normalized['Countries'],df_normalized['Death / Official Cases'], label='Official Cases')

plt.plot(df_normalized['Countries'],df_normalized['Death / Normalized Cases'], label='Normalized Cases')

plt.xticks(rotation='vertical')

plt.title('Covid-19 - Fatality rate')

plt.xlabel('Countries')

plt.ylabel('Fatality rate %')

plt.grid()

plt.legend()

plt.show()
import scipy

days = np.array(range(36))

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(data_it.index,(data_it['CurrentCases']/correction_it), label='Normalized Cases Italy', color='black', linestyle='-')

plt.plot(data_br.index,(data_br['CurrentCases']/correction_br), label='Normalized Cases Brazil', color = 'black', linestyle='--')

(a,b,c),_ = scipy.optimize.curve_fit(lambda t,a,b,c: c*a**(b*t)-c,  data_br.index,  data_br['CurrentCases']/correction_br,  p0=(4, 0.1, 0.1))

#(a,b),_ = scipy.optimize.curve_fit(lambda t,a,b: a**(b*t)-1,  data_br.index,  data_br['CurrentCases'],  p0=(4, 0.1))

#plt.plot(days,a**(b*days)-1, label='Brazil Projection', color = 'black', linestyle='-.')

plt.plot(days,c*a**(b*days)-c, label='Brazil Projection', color = 'black', linestyle='-.')

plt.xticks(rotation='vertical')

plt.grid()

plt.title('Projection of cases')

plt.ylabel('Cases')

plt.xlabel('Days')

plt.legend()

plt.show()
import scipy

days = np.array(range(40))

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(data_it.index,(data_it['CurrentCases']), label='Normalized Cases Italy', color='black', linestyle='-')

plt.plot(data_br.index,(data_br['CurrentCases']), label='Normalized Cases Brazil', color = 'black', linestyle='--')

(a,b),_ = scipy.optimize.curve_fit(lambda t,a,b: a**(b*t)-1,  data_br.index,  data_br['CurrentCases'],  p0=(4, 0.1))

plt.plot(days,a**(b*days)-1, label='Brazil Projection', color = 'black', linestyle='-.')

plt.xticks(rotation='vertical')

plt.grid()

plt.title('Projection of cases')

plt.ylabel('Cases')

plt.xlabel('Days')

plt.legend()

plt.show()