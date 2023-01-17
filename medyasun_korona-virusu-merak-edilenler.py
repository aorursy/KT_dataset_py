# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Javascript

from IPython.core.display import display

from IPython.core.display import HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')



#dataset name coronavirus-2019ncov

df = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

#df=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

df["Date"] = pd.to_datetime(df["Date"])



df_temp=pd.read_csv('/kaggle/input/covid19-global-weather-data/temperature_dataframe.csv')

df_temp["date"] = pd.to_datetime(df_temp["date"])

df_pop=pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
df_pop.rename(columns={'Country (or dependency)': 'country',

                             'Population (2020)' : 'population',

                             'Density (P/Km²)' : 'density',

                             'Fert. Rate' : 'fertility',

                             'Med. Age' : "age",

                             'Urban Pop %' : 'urban_percentage'}, inplace=True)

df.rename(columns={'Country/Region': 'country'}, inplace=True)

df_temp.rename(columns={'date': 'Date'}, inplace=True)

df_temp['country'] = df_temp['country'].replace('USA', 'US')

df_pop['country'] = df_pop['country'].replace('United States', 'US')

df['country'] = df['country'].replace('Mainland China', 'China')

df_pop = df_pop[["country", "population", "density", "fertility", "age", "urban_percentage"]]

df = df.merge(df_pop, on=['country'], how='left')

df_temp.drop_duplicates(subset =["Date",'country'], 

                     keep = 'first', inplace = True)

df = df.merge(df_temp, on=['Date','country'], how='left')

tarih=df['Date'].max()

guncel=df[df['Date']==tarih]

olum=guncel['Deaths'].sum()

iyilesme=guncel['Recovered'].sum()

vaka=guncel['Confirmed'].sum()

turkiye=guncel[guncel['country']=='Turkey']

turkiye_vaka=turkiye['Confirmed'].sum()

turkiye_olum=turkiye['Deaths'].sum()

turkiyeOlum_orani=(turkiye_olum/turkiye_vaka)*100

turkiye_iyilesme=turkiye['Recovered'].sum()

print ('Bilgilerin Son Güncellenme Tarihi: {}'.format(tarih))

print ('Türkiye Vaka: {:,.0f}'.format(turkiye_vaka))

print ('Türkiye Ölüm: {:,.0f}'.format(turkiye_olum))

print ('Türkiye İyileşme: {:,.0f}'.format(turkiye_iyilesme))

print ('Türkiye Ölüm Oranı: {:,.1f}%'.format(turkiyeOlum_orani))

print ('Toplam Ölüm: {:,.0f}'.format(olum))

print ('Toplam İyileşme: {:,.0f}'.format(iyilesme))

print ('Toplam Vaka: {:,.0f}'.format(vaka))
df['Active']=df['Confirmed']-df['Deaths']-df['Recovered']

temp = df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')





fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Yayılma Hızı', color_discrete_sequence = [rec, dth, act])

fig.show()




top_deaths=guncel[['country','Deaths','Recovered','Confirmed']].groupby('country').sum().sort_values(by='Deaths',ascending=False)

top_deaths['Iyilesme Orani(%)']=(top_deaths['Recovered']/top_deaths['Confirmed'])*100

top_deaths['Iyilesme Orani(%)']=top_deaths['Iyilesme Orani(%)'].round(1)

top_deaths['Olum Orani(%)']=(top_deaths['Deaths']/top_deaths['Confirmed'])*100

top_deaths['Olum Orani(%)']=top_deaths['Olum Orani(%)'].round(1)

top_deaths.head(10)

top_deaths.style.background_gradient(cmap='Reds')
epidemics = pd.DataFrame({

    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],

    'start_year' : [2019, 2003, 2014, 2012, 2009],

    'end_year' : [2020, 2004, 2016, 2017, 2010],

    'confirmed' : [guncel['Confirmed'].sum(), 8096, 28646, 2494, 6724149],

    'deaths' : [guncel['Deaths'].sum(), 774, 11323, 858, 19654]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)



temp = epidemics.melt(id_vars='epidemic', value_vars=['confirmed', 'deaths', 'mortality'],

                      var_name='Case', value_name='Value')



fig = px.bar(temp, x="epidemic", y="Value", color='epidemic', text='Value', facet_col="Case",

             color_discrete_sequence = px.colors.qualitative.Bold)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_yaxes(showticklabels=False)

fig.layout.yaxis2.update(matches=None)

fig.layout.yaxis3.update(matches=None)

fig.show()
gun_fut=15

confirmed = df.groupby("Date")[['Confirmed']].sum().reset_index()

confirmed.columns=['ds','y']

confirmed['ds'] = confirmed['ds'].dt.date

from fbprophet import Prophet

m = Prophet()

m.fit(confirmed)

future = m.make_future_dataframe(periods=gun_fut)

forecast_con = m.predict(future)

forecast_con[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
recovered = df.groupby("Date")[['Recovered']].sum().reset_index()

recovered.columns=['ds','y']

recovered['ds'] = recovered['ds'].dt.date

from fbprophet import Prophet

m = Prophet()

m.fit(recovered)

future = m.make_future_dataframe(periods=gun_fut)

forecast_rec = m.predict(future)

forecast_rec[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
death = df.groupby("Date")[['Deaths']].sum().reset_index()

death.columns=['ds','y']

death['ds'] = death['ds'].dt.date

from fbprophet import Prophet

m = Prophet()

m.fit(death)

future = m.make_future_dataframe(periods=gun_fut)

forecast_det = m.predict(future)

forecast_det[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
guncel_vaka = guncel['Confirmed'].sum()

guncel_olum=guncel['Deaths'].sum()

guncel_iyilesme=guncel['Recovered'].sum()

surv=guncel_vaka-guncel_olum-guncel_iyilesme

surv_fut=forecast_con['yhat'].iloc[-1].round(1)-forecast_rec['yhat'].iloc[-1].round(1)-forecast_det['yhat'].iloc[-1].round(1)

print('{} tarihi itibari ile görülen vaka sayısı {:,.0f} , virüs taşıyan kişi sayısı {:,.0f} (ölüm ve iyileşenleri çıkarttığımızda) '.format(df['Date'].max(),guncel_vaka,surv))

print('Yayılma hızı böyle giderse gelecek {} gün içerisinde tahminen görülen vaka sayısı {:,.0f} adete yükselebilir,'.format(gun_fut,forecast_con['yhat'].iloc[-1].round(1)))

print('Vakaların {:,.0f} adedi iyileşebilir,'.format(forecast_rec['yhat'].iloc[-1].round(1)))

print('Güncel verilere baktığımızda tahmini ölüm adedi {:,.0f} olabilir, (Guncel Rakam: {:,.0f})'.format(forecast_det['yhat'].iloc[-1].round(1),guncel_olum))

print('Neticede {} gün sonra toplamda Korona virüsünü taşıyan insan sayısı {:,.0f} olabilir.'.format(gun_fut,surv_fut.round(1)))



HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
HTML('<table class="table table-hover table-bordered table-condensed table-list"> <tbody> <tr bgcolor="#FCF8F8" class="scrollable bordered"> <td height="33"><div align="left">AGE</div></td> <td>DEATH RATE<br> confirmed cases <br></td> <td>DEATH RATE<br> all cases</td> </tr> <tr class="scrollable bordered"> <td width="244"><div align="left"><strong>80+ years old </strong></div></td> <td width="125"><div align="right"><strong>21.9%</strong></div></td> <td width="125"><div align="right"><strong>14.8%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>70-79 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>8.0%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>60-69 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>3.6%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>50-59 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>1.3%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>40-49 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>0.4%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>30-39 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>0.2%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>20-29 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>0.2%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>10-19 years old </strong></div></td> <td><div align="right"></div></td> <td><div align="right"><strong>0.2%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><strong>0-9 years old </strong></td> <td><div align="right"></div></td> <td><div align="right"><strong>no fatalities </strong></div></td> </tr> </tbody></table>')
HTML('<table class="table table-hover table-bordered table-condensed table-list"> <tbody> <tr bgcolor="#FCF8F8" class="scrollable bordered"> <td height="33"><div align="left">SEX</div></td> <td>DEATH RATE<br> confirmed cases <br></td> <td>DEATH RATE<br> all cases</td> </tr> <tr class="scrollable bordered"> <td width="209"><div align="left"><strong>Male</strong></div></td> <td width="151"><div align="right"><strong>4.7%</strong></div></td> <td width="142"><div align="right"><strong>2.8%</strong></div></td> </tr> <tr class="scrollable bordered"> <td><div align="left"><strong>Female</strong></div></td> <td><div align="right"><strong>2.8%</strong></div></td> <td><div align="right"><strong>1.7%</strong></div></td> </tr> </tbody></table>')
HTML('<img src="https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-800x429.png" alt="Coronavirus symptoms – who joint mission 2" class="wp-image-30460 lightbox-enabled" srcset="https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-800x429.png 800w, https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-400x214.png 400w, https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-150x80.png 150w, https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-768x412.png 768w, https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-1536x823.png 1536w, https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2-2048x1098.png 2048w" sizes="(max-width: 800px) 100vw, 800px" data-high-res-src="https://ourworldindata.org/uploads/2020/03/Coronavirus-Symptoms-–-WHO-joint-mission-2.png">')
HTML('<img src="https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China.png" alt="Coronavirus cfr by health condition in china" class="wp-image-30235 lightbox-enabled" srcset="https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China.png 1309w, https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China-400x214.png 400w, https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China-800x428.png 800w, https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China-150x80.png 150w, https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China-768x411.png 768w" sizes="(max-width: 1309px) 100vw, 1309px" data-high-res-src="https://ourworldindata.org/uploads/2020/03/Coronavirus-CFR-by-health-condition-in-China.png">')


tahmin=df[['Confirmed','Date','country','tempC','population']]

tahmin=tahmin.drop_duplicates(subset=['country','Date'],keep='first')

tahmin=tahmin.sort_values(by=['country','Date'])

tahmin['ulke_onceki']=tahmin['country'].shift()

tahmin['Confirmed_onceki']=tahmin['Confirmed'].shift()

tahmin['onceki']= tahmin.apply( lambda x: x['Confirmed_onceki'] if x['ulke_onceki']== x['country'] else 0 ,axis=1)

tahmin=tahmin.dropna()

tahmin['vaka_gunluk_fark']=tahmin['Confirmed']-tahmin['onceki']

tahmin=tahmin.drop(['Confirmed_onceki','ulke_onceki','onceki'],axis=1)

tahmin['sicaklik_grup']=pd.cut(tahmin['tempC'],[-40,-10,-5,0,5,10,15,20,30,35,40,50])

tahmin.sort_values(by=['vaka_gunluk_fark','tempC'],ascending=False)

#g=tahmin.groupby('sicaklik_grup')['vaka_gunluk_fark'].sum()

g=tahmin.groupby(['sicaklik_grup','country']).agg({'vaka_gunluk_fark': 'sum',

                                    'population': 'max'})

g=g.dropna()

gg=g.groupby('sicaklik_grup').agg({'vaka_gunluk_fark': 'sum',

                                    'population': 'sum'})



gg['ort_by_pop']=gg['vaka_gunluk_fark']/gg['population']*100

sg=pd.DataFrame(gg)

sg=sg.sort_values(by=['ort_by_pop'],ascending=False)

sg=sg.rename(columns={'vaka_gunluk_fark':'Vaka_Sayisi',

                      'population':'Nufus',

                      'ort_by_pop':'Nüfusa_Gore_Vaka_Yuzdesi'})

sg.style.background_gradient(cmap='Reds')