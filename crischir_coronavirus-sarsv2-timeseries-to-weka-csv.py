import numpy as np
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
time_series_covid_19_confirmed.head()
overseas_places=['Faroe Islands','St Martin','Channel Islands','Saint Barthelemy','Gibraltar','French Polynesia','French Guiana','Mayotte','Guadeloupe','Cayman Islands','Reunion','Montserrat','Greenland','New Caledonia','Bermuda','Isle of Man','Martinique','Anguilla','British Virgin Islands','Turks and Caicos Islands','Falkland Islands (Islas Malvinas)','Saint Pierre and Miquelon']
df=time_series_covid_19_confirmed
edfd=time_series_covid_19_deaths
edfr=time_series_covid_19_recovered
df = df.loc[~df['Province/State'].isin(overseas_places)]
edfd=edfd.loc[~edfd['Province/State'].isin(overseas_places)]
edfr=edfr.loc[~edfr['Province/State'].isin(overseas_places)]
europe_places=['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Hungary','Iceland','Italy','Korea, South','Latvia','Liechtenstein','Lithuania','Luxembourg','Malta','Norway','Poland','Portugal','Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
df = df.loc[df['Country/Region'].isin(europe_places)]
edfd= edfd.loc[edfd['Country/Region'].isin(europe_places)]
edfr= edfr.loc[edfr['Country/Region'].isin(europe_places)]

df.tail()
df=df.drop(['Province/State','Lat','Long'], axis=1)
edfd=edfd.drop(['Province/State','Lat','Long'], axis=1)
edfr=edfr.drop(['Province/State','Lat','Long'], axis=1)
#df=df.drop(['Lat','Long'], axis=1)
#df.head()
df.rename(columns={'Country/Region':'Time'}, inplace=True)
edfd.rename(columns={'Country/Region':'Time'}, inplace=True)
edfr.rename(columns={'Country/Region':'Time'}, inplace=True)
df_t=df.set_index('Time').transpose()
edfd_t=edfd.set_index('Time').transpose()
edfr_t=edfd.set_index('Time').transpose()
#df_t=df_t.loc[~(df_t==0).all(axis=1)]
#df_t.head()
#df_t.reindex(inplace=True)
#df_t.head()
#df_t.describe()
#edfd.describe()
#edfr.describe()
#edfd_t[25:].head()
df_t=df_t.loc[:, df_t.max().sort_values(ascending=False).index]
edfd_t=edfd_t.loc[:, df_t.max().sort_values(ascending=False).index]
edfr_t=edfr_t.loc[:, df_t.max().sort_values(ascending=False).index]
#df_t=df_t.drop(columns=['Korea, South'])
#edfr_t=df_t.drop(columns=['Korea, South'])
#edfd_t=df_t.drop(columns=['Korea, South'])
#list(df_t.columns) 
df_t.to_csv(r'SARS-v2_Europe_cases.csv', index = True,index_label ="Time")
edfd_t.to_csv(r'SARS-v2_Europe_death.csv', index = True,index_label ="Time")
edfr_t.to_csv(r'SARS-v2_Europe_recovery.csv', index = True,index_label ="Time")
cdf =time_series_covid_19_confirmed
cdfd=time_series_covid_19_deaths
cdfr=time_series_covid_19_recovered
mainland_china=['China',""]
mainland_china
cdf = cdf.loc[cdf['Country/Region'].isin(mainland_china)]
cdfd = cdfd.loc[cdfd['Country/Region'].isin(mainland_china)]
cdfr = cdfr.loc[cdfr['Country/Region'].isin(mainland_china)]

cdfr

cdf=cdf.drop(['Country/Region','Lat','Long'], axis=1)
cdfd=cdfd.drop(['Country/Region','Lat','Long'], axis=1)
cdfr=cdfr.drop(['Country/Region','Lat','Long'], axis=1)
cdf.rename(columns={'Province/State':'Time'}, inplace=True)
cdfd.rename(columns={'Province/State':'Time'}, inplace=True)
cdfr.rename(columns={'Province/State':'Time'}, inplace=True)
cdf.tail()
cdf_t=cdf.set_index('Time').transpose()
cdfd_t=cdfd.set_index('Time').transpose()
cdfr_t=cdfr.set_index('Time').transpose()
cdfr_t
cdf_t=cdf_t.loc[:, cdf_t.max().sort_values(ascending=False).index]
cdfd_t=cdfd_t.loc[:, cdf_t.max().sort_values(ascending=False).index]
cdfr_t=cdfr_t.loc[:, cdf_t.max().sort_values(ascending=False).index]
cdfr_t.tail()
cdf_t.to_csv(r'SARS-v2_China_confirmed.csv', index = True,index_label ="Time")
cdfd_t.to_csv(r'SARS-v2_China_deaths.csv', index = True,index_label ="Time")
cdfr_t.to_csv(r'SARS-v2_China_recovery.csv', index = True,index_label ="Time")
usdf =time_series_covid_19_confirmed
usdfd=time_series_covid_19_deaths
usdfr=time_series_covid_19_recovered
usdfr
us_states=['US']
us_states
usdf = usdf.loc[usdf['Country/Region'].isin(us_states)]
#usdfd = usdfd.loc[usdfd['Country/Region'].isin(us_states)]
#usdfr = usdfr.loc[usdfd['Country/Region'].isin(us_states)]


usdfd = usdfd.loc[usdfd['Country/Region'].isin(us_states)]
usdfr = usdfr.loc[usdfr['Country/Region'].isin(us_states)]
usdfr.tail()
usdf=usdf.drop(['Country/Region','Lat','Long'], axis=1)
usdfd=usdfd.drop(['Country/Region','Lat','Long'], axis=1)
usdfr=usdfr.drop(['Country/Region','Lat','Long'], axis=1)
#df=df.drop(['Lat','Long'], axis=1)
usdf.head()
usdf.rename(columns={'Province/State':'Time'}, inplace=True)
usdfd.rename(columns={'Province/State':'Time'}, inplace=True)
usdfr.rename(columns={'Province/State':'Time'}, inplace=True)
usdf_t=usdf.set_index('Time').transpose()
usdfd_t=usdfd.set_index('Time').transpose()
usdfr_t=usdfr.set_index('Time').transpose()
usdf_t.columns = ['US']
usdfd_t.columns = ['US']
#usdfr_t.columns = ['US']
usdf_t.columns = ['US']

usdf_t.to_csv(r'SARS-v2_US_confirmed.csv', index = True,index_label ="Time")
usdfd_t.to_csv(r'SARS-v2_US_deaths.csv', index = True,index_label ="Time")
usdfr_t.to_csv(r'SARS-v2_US_recovered.csv', index = True,index_label ="Time")