import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.offline as py

from datetime import date, timedelta

from sklearn.cluster import KMeans

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

#df_confirmed.head(2).append(df_confirmed.tail(2)).T
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

#df_deaths.head(2).append(df_deaths.tail(2)).T
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])

# new dataset 

df_covid19 = df_covid19.drop(["People_Tested","People_Hospitalized","UID","ISO3","Mortality_Rate"],axis =1)

#df_covid19.head(2).append(df_covid19.tail(2)).T

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])

# new dataset 

df_covid19 = df_covid19.drop(["People_Tested","People_Hospitalized","UID","ISO3","Mortality_Rate"],axis =1)

#df_covid19.head(2).append(df_covid19.tail(2)).T
df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})

df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})

df_covid19 = df_covid19.rename(columns={"Country_Region": "country"})

df_covid19["Active"] = df_covid19["Confirmed"]-df_covid19["Recovered"]-df_covid19["Deaths"]



# Changing the conuntry names as required by pycountry_convert Lib

df_confirmed = df_confirmed.replace(np.nan, '', regex=True)

df_deaths = df_deaths.replace(np.nan, '', regex=True)



df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"

df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"

df_covid19.loc[df_covid19['country'] == "US", "country"] = "USA"



df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'

df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'

df_covid19.loc[df_covid19['country'] == "Korea, South", "country"] = "South Korea"



df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_covid19.loc[df_covid19['country'] == "Taiwan*", "country"] = "Taiwan"



df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_covid19.loc[df_covid19['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"



df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_covid19.loc[df_covid19['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"



df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"

df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"

df_covid19.loc[df_covid19['country'] == "Reunion", "country"] = "Réunion"



df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_covid19.loc[df_covid19['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"



df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_covid19.loc[df_covid19['country'] == "Bahamas, The", "country"] = "Bahamas"



df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'

df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'

df_covid19.loc[df_covid19['country'] == "Gambia, The", "country"] = "Gambia"

df_countries_cases = df_covid19.copy().drop(['Lat','Long_','Last_Update'],axis =1)

df_countries_cases.index = df_countries_cases["country"]

df_countries_cases = df_countries_cases.drop(['country'],axis=1)

df_countries_cases.fillna(0,inplace=True)

#print(df_confirmed.head(3))



temp = df_confirmed.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_confirmed.columns[-1], ascending= False)

target_countries = temp.iloc[:10,:].index.tolist()

target_countries += ['South Korea']

temp = temp.loc[temp.index.isin(target_countries)].T





temp.plot(figsize=(12,24),grid=True, linewidth=4).legend(title='Country', bbox_to_anchor=(1, 1))



plt.title('Global Trend Comparison(number of confirmed cases accumulated)', fontsize=15)

plt.xlabel("date", labelpad=15)

plt.ylabel("number of confirmed Cases accumulated", labelpad=15)

plt.show()

from scipy.interpolate import make_interp_spline, BSpline



temp = df_confirmed.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_confirmed.columns[-1], ascending= False)



threshold = 1

f = plt.figure(figsize=(10,12))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    if i >= 10:

        if country != "South Korea" :

            continue

    days = 80

    t = temp.loc[temp.index== country].values[0]

    t = t[t>threshold][:days]

     

    date = np.arange(0,len(t[:days]))

    xnew = np.linspace(date.min(), date.max(), 30)

    spl = make_interp_spline(date, t, k=1)  # type: BSpline

    power_smooth = spl(xnew)

    plt.plot(xnew,power_smooth,'-o',label = country,linewidth =3, markevery=[-1])



plt.tick_params(labelsize = 14)        

plt.xticks(np.arange(0,days,7),[str(i) for i in range(days)][::7])     



# Reference lines 

x = np.arange(0,18)

y = 2**(x+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,int(days-12))

y = 2**(x/2+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every second day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,int(days-5))

y = 2**(x/4+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every 4 days",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,int(days-4))

y = 2**(x/7+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,int(days-4))

y = 2**(x/30+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



# plot Params

plt.xlabel("Days",fontsize=17)

plt.ylabel("Number of confirmed cases in log-scale",fontsize=17)

plt.title("Global Trend Comparison(confirmed cases accumulated in log-scale) ",fontsize=22)

plt.legend(loc = "upper left")

plt.yscale("log")

plt.grid(which="both")

plt.savefig('Global Trend Comparison(confirmed cases in log-scale).png')

plt.show()
#print(df_deaths.head(3))



temp = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)

target_countries = temp.iloc[:10,:].index.tolist()

target_countries += ['South Korea']

temp = temp.loc[temp.index.isin(target_countries)].T





temp.plot(figsize=(12,24),grid=True, linewidth=4).legend(title='Country', bbox_to_anchor=(1, 1))



plt.title('Global Trend Comparison(number of deaths accumulated)', fontsize=15)

plt.xlabel("date", labelpad=15)

plt.ylabel("number of deaths accumulated", labelpad=15)

plt.show()
temp = df_deaths.groupby('country').sum().drop(["Lat","Long"],axis =1).sort_values(df_deaths.columns[-1], ascending= False)



threshold = 1

f = plt.figure(figsize=(10,12))

ax = f.add_subplot(111)

for i,country in enumerate(temp.index):

    if i >= 10:

        if country != "South Korea":

            continue

    days = 80

    t = temp.loc[temp.index== country].values[0]

    t = t[t>threshold][:days]

     

    date = np.arange(0,len(t[:days]))

    xnew = np.linspace(date.min(), date.max(), 30)

    spl = make_interp_spline(date, t, k=1)  # type: BSpline

    power_smooth = spl(xnew)

    plt.plot(xnew,power_smooth,'-o',label = country,linewidth =3, markevery=[-1])





plt.tick_params(labelsize = 14)        

plt.xticks(np.arange(0,days,7),[str(i) for i in range(days)][::7])     



# Reference lines 

x = np.arange(0,18)

y = 2**(x+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,days-12)

y = 2**(x/2+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every second day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,int(days-5))

y = 2**(x/4+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every 4 days",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.8)



x = np.arange(0,days-3)

y = 2**(x/7+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



x = np.arange(0,days-3)

y = 2**(x/30+np.log2(threshold))

plt.plot(x,y,"--",linewidth =2,color = "gray")

plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



# plot Params

plt.xlabel("Days",fontsize=17)

plt.ylabel("Number of deaths in log scale",fontsize=17)

plt.title("Global Trend Comparison(deaths accumulated in log-scale)",fontsize=22)

plt.legend(loc = "upper left")

plt.yscale("log")

plt.grid(which="both")

plt.savefig('Global Trend Comparison(deaths accumulated in log-scale).png')

plt.show()
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

#cases.head(2).append(cases.tail(2)).T
py.init_notebook_mode(connected=True)



grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

grp = grp.reset_index()

grp['Date'] = pd.to_datetime(grp['ObservationDate'])

grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')

grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']

grp['Country'] =  grp['Country/Region']



fig = px.choropleth(grp, locations="Country", locationmode='country names', 

                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",

                     animation_frame="Date",width=1000, height=700,

                     color_continuous_scale='Reds',

                     range_color=[1000,300000],



                     title='World Map of Coronavirus')



fig.update(layout_coloraxis_showscale=True)

py.offline.iplot(fig)
grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

grp = grp.reset_index()

grp['Confirmed_shift1'] = grp.groupby(['Country/Region'])['Confirmed'].shift(1)

grp['Confirmed_new'] = grp['Confirmed'] - grp['Confirmed_shift1']                                            

grp['Date'] = pd.to_datetime(grp['ObservationDate'])

grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')

grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']

grp['Country'] =  grp['Country/Region']



#grp.tail(10)
fig = px.choropleth(grp, locations="Country", locationmode='country names', 

                     color="Confirmed_new", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",

                     animation_frame="Date",width=1000, height=700,

                     color_continuous_scale='Reds',

                     range_color=[0,5000],



                     title='World Map of Coronavirus')



fig.update(layout_coloraxis_showscale=True)

py.offline.iplot(fig)
df_route = pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

comp = pd.read_excel('/kaggle/input/covid19327/COVID-19-3.27-top30-500.xlsx')

df_patient = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

weather = pd.read_csv("../input/coronavirusdataset/Weather.csv")
#df_route.shape #5321x8

df_route['date']= pd.to_datetime(df_route['date']) 

df_route['Month'] = pd.DatetimeIndex(df_route['date']).month

df_route['Day'] = pd.DatetimeIndex(df_route['date']).day

#df_route.head(2).append(df_route.tail(2)).T 
import folium

from folium.plugins import FastMarkerCluster, MarkerCluster

sk_map = folium.Map(location=[36,128], zoom_start=7) #tiles=Stamen Terrain, Stamen Toner, Mapbox Bright, and Mapbox Control Room



fmc = FastMarkerCluster(df_route[['latitude', 'longitude']].values.tolist())

sk_map.add_child(fmc)

sk_map.save('./sk_map.html')



#from IPython.core.display import display, HTML

#display(HTML(filename='./sk_map.html'))
from IPython.display import IFrame

IFrame('./sk_map.html', width=800, height=800)
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==1) & (df_route.Day>15)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==2) & (df_route.Day<=15)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==2) & (df_route.Day<=20) & (df_route.Day>15)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==2) & (df_route.Day<=25) & (df_route.Day>20)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==2) & (df_route.Day>25)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==3) & (df_route.Day<=15)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
import folium

sk_map = folium.Map(location=[36,128], zoom_start=6, width=400, height=400)

df_temp = df_route.loc[(df_route.Month==3) & (df_route.Day>15)]

print(df_temp.shape)

for i, row in df_temp.iterrows():

    lat, lon, city = row['latitude'], row['longitude'], row['city']

    folium.CircleMarker([lat, lon], radius=1, color='red', popup =('City: ' + str(city) + '<br>'),

                        fill_color='red', fill_opacity=0.7 ).add_to(sk_map)

sk_map
time_df = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')

time_df['test_new'] = time_df['test'] - time_df['test'].shift(1)

time_df['negative_new'] = time_df['negative'] - time_df['negative'].shift(1)

time_df['confirmed_new'] = time_df['confirmed'] - time_df['confirmed'].shift(1)

time_df['released_new'] = time_df['released'] - time_df['released'].shift(1)

time_df['deceased_new'] = time_df['deceased'] - time_df['deceased'].shift(1)



time_df = time_df.fillna(0.0)



time_df['test_new'] = time_df['test_new'].astype(int)

time_df['negative_new'] = time_df['negative_new'].astype(int)

time_df['confirmed_new'] = time_df['confirmed_new'].astype(int)

time_df['released_new'] = time_df['released_new'].astype(int)

time_df['deceased_new'] = time_df['deceased_new'].astype(int)
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Tests & Results Accumulated in South Korea', fontsize=15)

for col in time_df.columns[2:5]:

    plt.plot(np.arange(1, time_df.shape[0]+1), time_df[col], label=col) # first to last day on the data

ax.legend()

plt.show()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Tests & Results Incremental in South Korea', fontsize=15)

for col in time_df.columns[7:10]:

    plt.plot(np.arange(1, time_df.shape[0]+1), time_df[col], label=col) # first to last day on the data

ax.legend()

plt.show()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Confirmed Persons & Results Accumulated in South Korea', fontsize=15)

for col in time_df.columns[4:7]:

    plt.plot(np.arange(1, time_df.shape[0]+1), time_df[col], label=col) # first to last day on the data

ax.legend()

plt.show()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Confirmed Persons & Results Incremental in South Korea', fontsize=15)

for col in time_df.columns[9:]:

    plt.plot(np.arange(1, time_df.shape[0]+1), time_df[col], label=col) # first to last day on the data

ax.legend()

plt.show()
patient_info = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')
#print(patient_info['contact_number'].isna().sum())

patient_contact = patient_info[patient_info['contact_number'].notna()]

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.hist(patient_contact['contact_number'], bins=100)

plt.gca().set(title='Number of contacts of a patient before confirmed', ylabel='Frequency')
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive']

provinces = ['Seoul','Busan', 'Gyeonggi-do', 'Gyeongsangbuk-do','Chungcheongnam-do']

for i in range(0,1):

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(patient_contact.loc[patient_contact['province']==provinces[i],'contact_number'], bins=100, color=colors[i])

    plt.gca().set(title=provinces[i], ylabel='Frequency')

    plt.show()
for i in range(1,2):

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(patient_contact.loc[patient_contact['province']==provinces[i],'contact_number'], bins=100, color=colors[i])

    plt.gca().set(title=provinces[i], ylabel='Frequency')

    plt.show()
for i in range(2,3):

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(patient_contact.loc[patient_contact['province']==provinces[i],'contact_number'], bins=100, color=colors[i])

    plt.gca().set(title=provinces[i], ylabel='Frequency')

    plt.show()
for i in range(3,4):

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(patient_contact.loc[patient_contact['province']==provinces[i],'contact_number'], bins=100, color=colors[i])

    plt.gca().set(title=provinces[i], ylabel='Frequency')

    plt.show()
for i in range(4,5):

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(patient_contact.loc[patient_contact['province']==provinces[i],'contact_number'], bins=100, color=colors[i])

    plt.gca().set(title=provinces[i], ylabel='Frequency')

    plt.show()
fig, axes = plt.subplots(1, 5, figsize=(10,2.5), dpi=100, sharey=True) #sharex=True

for i, (ax, province) in enumerate(zip(axes.flatten(), provinces)):

    x = patient_contact.loc[patient_contact['province']==str(province), 'contact_number'].values

    ax.hist(x,  

            bins=100, 

            density=True, 

            stacked=False, #the sum of the histograms is normalized to 1

            color=colors[i])

    ax.set_title(province)

    ax.set_xlabel('number of contacts')

plt.suptitle('Probability Histogram of Contact Numbers by Provinces', y=1.05, size=16)

#ax.set_xlim(0, 100); 

axes.flatten()[0].set_ylim(0, 0.7)

axes.flatten()[0].set_ylabel('probability')

plt.tight_layout()