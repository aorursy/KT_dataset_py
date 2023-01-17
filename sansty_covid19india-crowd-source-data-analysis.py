import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

from plotly.offline import init_notebook_mode, plot

from scipy.interpolate import make_interp_spline, BSpline

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os



import warnings

warnings.filterwarnings('ignore')



import plotly.graph_objects as go

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import datetime

from datetime import date as datefun

from datetime import timedelta

import requests



data_raw=requests.get('https://api.covid19india.org/raw_data.json').json()  # read API

head=list(data_raw['raw_data'][0].keys())

head # Keys


temp = pd.DataFrame([]) 

for i in range(0,len(data_raw['raw_data'])):

    data1=pd.DataFrame([data_raw['raw_data'][i].values()], columns=head)

    temp=temp.append(data1,ignore_index = True)



#------------  Remove No data rows ------------------------------------

temp1= list(temp.loc[0:len(data_raw['raw_data']),'currentstatus'])

valid_data=[i for i, item in enumerate(temp1) if item != '']

data_raw=temp[0:len(valid_data)]
data_raw 

for index, row in data_raw.iterrows():

    if row['detectedstate'] in 'Delhi':

            data_raw.at[index,'detecteddistrict'] = 'Delhi'

def get_state_df_from_api(data_temp,start_date,end_date):

    tot_days=(datefun.today()-datetime.datetime.strptime(start_date, '%Y-%m-%d').date()).days



    temp_data=data_raw.copy()

    temp1= list(data_raw.loc[:,'dateannounced'])

    DATE = [datetime.datetime.strptime(x,'%d/%m/%Y') for x in temp1]

    temp_data.loc[:,'dateannounced'] = DATE



    temp2= pd.to_datetime(temp_data.dateannounced,format='%Y-%m-%d')

    temp_data.loc[:,'dateannounced'] = temp2.dt.strftime('%Y-%m-%d')





    # #---------Set till-date as the Last date available in data_raw-------

    yesterday = datefun.today() - timedelta(days=1)

    date_yesterday=yesterday.strftime('%Y-%m-%d')

    date_yesterday=end_date



    data_temp = temp_data[(temp_data['dateannounced'] <= date_yesterday)]

    data=data_temp

    data.loc[:,'Confirmed']=np.ones((data_temp['dateannounced'].size),dtype='int')

    data.loc[:,'Recovered'] = np.zeros((data_temp['dateannounced'].size),dtype='int')

    temp=data[data['currentstatus']=='Recovered']

    data.loc[list(temp.index),'Recovered']=1

    data.loc[:,'Fatalities'] = np.zeros((data_temp['dateannounced'].size),dtype='int')

    temp=data[data['currentstatus']=='Fatalities']

    data.loc[list(temp.index),'Fatalities']=1



    data=data.rename({'detectedstate': 'State'}, axis='columns')

    data=data.rename({'dateannounced': 'Date'}, axis='columns')



    temp_data=data.copy()

    temp2= pd.to_datetime(temp_data.Date,format='%Y-%m-%d')

    data.loc[:,'Date'] = temp2.dt.strftime('%Y-%m-%d')





    ''' Prepare State Data '''

    States=list(data.State.unique())

    final_data=pd.DataFrame([])

    for j in range (0,len(States)):

        st= data.query('State == '+ '"'+States[j]+'"')

        #a=st['Confirmed'].cumsum()

        date=list(st.Date.unique())

        a=list(pd.date_range(start=start_date, end=end_date).date)#date[len(date)-1]).date)

        dat1=dict()

        dat2=dict()

        for i in a:

            dt=st["Date"] == str(i)

            aa=st.loc[dt].Confirmed.sum()

            bb=st.loc[dt].Fatalities.sum()

            dat1[i]=aa

            dat2[i]=bb

        datc=np.array(list(dat1.values())).cumsum()

        datf=np.array(list(dat2.values())).cumsum()

        nam=pd.Series([States[j]])

        name=nam.repeat(len(a))

        days=np.arange(len(a))

        tempdata={'State':name,'Date': pd.to_datetime(a,format='%Y-%m-%d'),'Day':days,'ConfirmedCases': datc.T,'Fatalities': datf.T,}

        dd=pd.DataFrame.from_dict(tempdata)

        final_data=final_data.append(dd,ignore_index = True)



    temp1= pd.to_datetime(final_data.Date)

    final_data['Date'] = temp1.dt.strftime('%Y-%m-%d')

    return final_data



def get_district_df_from_api(data_temp,start_date,end_date):

    tot_days=(datefun.today()-datetime.datetime.strptime(start_date, '%Y-%m-%d').date()).days



    temp_data=data_raw.copy()

    temp1= list(data_raw.loc[:,'dateannounced'])

    DATE = [datetime.datetime.strptime(x,'%d/%m/%Y') for x in temp1]

    temp_data.loc[:,'dateannounced'] = DATE



    temp2= pd.to_datetime(temp_data.dateannounced,format='%Y-%m-%d')

    temp_data.loc[:,'dateannounced'] = temp2.dt.strftime('%Y-%m-%d')





    # #---------Set till-date as the Last date available in data_raw-------

    yesterday = datefun.today() - timedelta(days=1)

    date_yesterday=yesterday.strftime('%Y-%m-%d')

    date_yesterday=end_date

    

    data_temp = temp_data[(temp_data['dateannounced'] <= date_yesterday)]

    datad=data_temp

    datad.loc[:,'Confirmed']=np.ones((data_temp['dateannounced'].size),dtype='int')

    datad.loc[:,'Recovered'] = np.zeros((data_temp['dateannounced'].size),dtype='int')

    temp=datad[datad['currentstatus']=='Recovered']

    datad.loc[list(temp.index),'Recovered']=1

    datad.loc[:,'Fatalities'] = np.zeros((data_temp['dateannounced'].size),dtype='int')

    temp=datad[datad['currentstatus']=='Fatalities']

    datad.loc[list(temp.index),'Fatalities']=1



    datad=datad.rename({'detecteddistrict': 'District'}, axis='columns')

    datad=datad.rename({'dateannounced': 'Date'}, axis='columns')

   

    #datad['Date']= pd.to_datetime(datad['Date'],format='%Y-%m-%d')

    

    temp_data=datad.copy()

    temp2= pd.to_datetime(temp_data.Date,format='%Y-%m-%d')

    datad.loc[:,'Date'] = temp2.dt.strftime('%Y-%m-%d')



    ''' Prepare District Data '''

    Districts=list(datad.District.unique())

    final_datad=pd.DataFrame([])

    for j in range (0,len(Districts)):

        st= datad.query('District == '+ '"'+Districts[j]+'"')

        #a=st['Confirmed'].cumsum()

        date=list(st.Date.unique())

        a=list(pd.date_range(start=start_date, end=end_date).date)

        dat1=dict()

        dat2=dict()

        for i in a:

            dt=st["Date"] == str(i)

            aa=st.loc[dt].Confirmed.sum()

            bb=st.loc[dt].Fatalities.sum()

            dat1[i]=aa

            dat2[i]=bb

        datc=np.array(list(dat1.values())).cumsum()

        datf=np.array(list(dat2.values())).cumsum()

        nam=pd.Series([Districts[j]])

        name=nam.repeat(len(a))

        days=np.arange(len(a))

        tempdata={'District':name,'Date': pd.to_datetime(a,format='%Y-%m-%d'),'Day':days,'ConfirmedCases': datc.T,'Fatalities': datf.T,}

        dd=pd.DataFrame.from_dict(tempdata)

        final_datad=final_datad.append(dd,ignore_index = True)



    temp1= pd.to_datetime(final_datad.Date)

    final_datad['Date'] = temp1.dt.strftime('%Y-%m-%d')

    return final_datad


start_date='2020-03-01'; end_date='2020-04-27'

df_state = get_state_df_from_api(data_raw,start_date,end_date)



#df_district = get_district_df_from_api(data_raw,start_date,end_date)

#df_district
latest_date=df_state['Date'].max()

df_state_latest = df_state[df_state['Date'] == latest_date].sort_values(by='ConfirmedCases',ascending=False)

#df_state_= df_latest.groupby(['District']).sum().reset_index() # get sum of cases for each province

df_state_latest.style.background_gradient(cmap='Reds')
df_k= pd.read_csv('../input/covid19-in-india/covid_19_india.csv')



df_k['Date']  = pd.to_datetime(df_k['Date'],format="%d/%m/%y") # new clean date columnn

df_k_latest = df_k[(df_k['Date'] == end_date)].sort_values(by='Confirmed',ascending=False)

#df_temp=df[df['Sno']==latest_index]

#latest_date=df_temp['Date'].max()

#df_latest = df[df['Date'] == latest_date]

df_k
print("Total number of Cases According to Kaggle Data base as on ",end_date, "is = ",df_k_latest['Confirmed'].sum())

print("Total number of Cases According to Covid19India.org as on ",end_date, "is = ",df_state_latest['ConfirmedCases'].sum())
def plot_trend_rowdf(df,threshold,Days,First_n,highlight,Label_X,Label_Y,Title):

# modified from the awesome by Tarun Kumar work at https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons and modified for use



    temp_I = df#.sort_values(df.columns[-1], ascending= True)

    temp_I_sum = temp_I.sum(axis=1)

   # print(temp_I_sum)

    last_row=temp_I.tail(1)

    last_row1 = last_row.sort_values(by=last_row.last_valid_index(),ascending=False, axis=1)



 #   threshold = 50

 #   Days=51

    f = plt.figure(figsize=(10,12))

    ax = f.add_subplot(111)

    x = Days

    t1_I = temp_I_sum.to_numpy()

    t2_I = t1_I[t1_I>threshold][:x]

    date = np.arange(0,len(t2_I[:x]))

    xnew = np.linspace(date.min(), date.max(), Days)

    spl = make_interp_spline(date, t2_I, k=1)  # type: BSpline

    power_smooth = spl(xnew)

    marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=6, markerfacecolor='#ffffff')

    plt.plot(xnew,power_smooth,"-.",label = 'All The Cases',**marker_style)

          

    for i,col in enumerate(last_row1.columns[:15]):

        if col not in  ['Date','date']:

            x = Days

            t1_I = temp_I[col].to_numpy()

            t2_I = t1_I[t1_I>threshold][:x]

            if t2_I.size>0 :

                date = np.arange(0,len(t2_I[:x]))

                xnew = np.linspace(date.min(), date.max(), Days)

                spl = make_interp_spline(date, t2_I, k=1)  # type: BSpline

                power_smooth = spl(xnew)

                if col in highlight:

                    marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=6, markerfacecolor='#000000')

                    plt.plot(xnew,power_smooth,"-.",label = col,**marker_style)

                else:  

                    plt.plot(xnew,power_smooth,'-o',label = col,linewidth =3, markevery=[-1])

             #   else:

              #  



    plt.tick_params(labelsize = 14)        

    plt.xticks(np.arange(0,Days,7),[ "Day "+str(i) for i in range(Days)][::7])     



    # Reference lines 

    x = np.arange(0,Days/3)

    y = 2**(x+np.log(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days/2)

    y = 2**(x/2+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every socend day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/7+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/30+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "gray")

    plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)



    x = np.arange(0,Days-4)

    y = 2**(x/4+np.log2(threshold))

    plt.plot(x,y,"--",linewidth =2,color = "Red")

    plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)



    # plot Params

    plt.xlabel(Label_X,fontsize=17)

    plt.ylabel(Label_Y,fontsize=17)

    plt.title(Title,fontsize=22)

    plt.legend(loc = "upper left")

    plt.yscale("log")

    plt.grid(which="both")

    plt.show()
df_state=df_state.sort_values(by='Date',ascending=True)

#gb_state_time = df_state.groupby(['Date']).sum().reset_index()

gb_state_time = df_state.pivot_table(index=['Date'], 

            columns=['State'], values='ConfirmedCases').fillna(0)

gb_state_time['Date']=gb_state_time.index

gb_state_time.index.name = None

gb_state_time = gb_state_time.sort_values(by='Date',ascending=True)

gb_state_time['Date'] = pd.to_datetime(gb_state_time['Date'],format="%Y-%m-%d")

gb_state_time.index = pd.to_datetime(gb_state_time['Date'],format="%Y-%m-%d")

gb_state_time=gb_state_time.drop(['Date'],axis = 1) 

gb_state_time.tail(3)


start_date='2020-03-01'; end_date='2020-04-23'

df_district = get_district_df_from_api(data_raw,start_date,end_date)



df_district.tail(10)
df_district=df_district.sort_values(by='Date',ascending=True)

#gb_district_time = df_district.groupby(['Date']).sum().reset_index()

gb_district_time = df_district.pivot_table(index=['Date'], 

            columns=['District'], values='ConfirmedCases').fillna(0)

gb_district_time['Date']=gb_district_time.index

gb_district_time.index.name = None

gb_district_time = gb_district_time.sort_values(by='Date',ascending=True)

gb_district_time['Date'] = pd.to_datetime(gb_district_time['Date'],format="%Y-%m-%d")

gb_district_time.index = pd.to_datetime(gb_district_time['Date'],format="%Y-%m-%d")

gb_district_time=gb_district_time.drop(['Date'],axis = 1) 

gb_district_time.rename(columns = {list(gb_district_time)[0]:'Unkown'}, inplace=True)

gb_district_time.tail(10)
highlight=["Mumbai","Pune","Indore","Delhi","Kasaragod"]

First_n=15

Days=50

Threshold=50

Label_X="Days( Referenced to Threshold )"

Label_Y="Number of Confirmed Cases (Log Scale)"

Title="Trend Comparison of Different District (confirmed)\n Top 15 Cities in terms of COVID-19 Cases"

plot_trend_rowdf(gb_district_time,Threshold,Days,First_n,highlight,Label_X,Label_Y,Title)
latest_date=df_district['Date'].max()

df_latest = df_district[df_district['Date'] == latest_date]

df_district_m = df_latest.groupby(['District']).sum().reset_index()

#df_district_m1=df_district_m[df_district_m.District != []]

#df_district_m  = df_district_m['District'].str.replace(" ","")

df_district_m.at[0,'District']="Unknown"

df_district_m.head(10)
from geopy.extra.rate_limiter import RateLimiter

from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable

import time

from  geopy.geocoders import Nominatim



def get_lat_lon(geolocator, city):

    location = None

    country="India"

    try:

        location = geolocator.geocode(city+','+ country)

    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable):

        time.sleep(1)

        try:

            location = geolocator.geocode(city+','+ country)

        except (

                GeocoderTimedOut, GeocoderServiceError,

                GeocoderUnavailable):

            return None, None

    if location:

        return location

    else:

        return None


geolocator = Nominatim()



city ="una"

location=get_lat_lon(geolocator,city)

country ="India"

loc = geolocator.geocode(city+','+ country)

print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)

print(location)
df_lat_lon= pd.read_csv('../input/district-lat-lon-covid/India_District_Lat_Long')

df_lat_lon

df_lat = df_lat_lon.pivot_table(columns=['District'],values='lat').fillna(0)

df_lon = df_lat_lon.pivot_table(columns=['District'],values='long').fillna(0)



df_district_m['lat']=""

df_district_m['long']=""

missing_list = ["Unknown"]

for index, row in df_district_m.iterrows():

    if row['District'] in df_lat.columns:

        df_district_m.at[index,'lat'] = df_lat[row['District']]['lat']

        df_district_m.at[index,'long'] = df_lon[row['District']]['long']

        #print(df_lat[row['District']])

    else :

        df_district_m.at[index,'lat'] = "0"

        df_district_m.at[index,'long'] = "0"

        if row['District'] not in missing_list :

            city=row['District']

            #loc = geolocator.geocode(city+','+ 'India')

            loc=get_lat_lon(geolocator,city)

            print(loc)

            if not loc:

                print(city)

                missing_list.append(row['District'])

            else :

                df_district_m.at[index,'lat'] = loc.latitude

                df_district_m.at[index,'long'] = loc.longitude

                
df1 =df_district_m[df_district_m['lat'].notna()]

df1.head(10)

# Cases which are not linked to any district is shown as Unkown District

#Wirintg data to csv file to avoid api calls for getting lat long

#df1.to_csv("India_District_Lat_lon", mode='w', columns=['District','lat','long'], index=False)

scalar=0.8

map = folium.Map(location=[20, 80], zoom_start=4.5,tiles='cartodbpositron')



df1['color']=df1['ConfirmedCases'].apply(lambda count:"red" if count>=400 else

                                         "green" if count>=200 and count<400 else

                                         "darkblue" if count>=100 and count<200 else

                                         "brown" if count>=50 and count<100 else

                                         "grey" if count>=10 and count<50 else

                                         "black")



df1['size']=df1['ConfirmedCases'].apply(lambda count:12 if count>=400 else

                                         8 if count>=200 and count<400 else

                                         6 if count>=100 and count<200 else

                                         3 if count>=50 and count<100 else

                                         1 if count>=10 and count<50 else

                                         0.1)



for lat, lon, value, name, color1,size in zip(df1['lat'], df1['long'], df1['ConfirmedCases'], df1['District'],df1['color'],df1['size']):

    folium.CircleMarker([lat, lon],

                        radius=size*3*scalar,

                        popup = ('<strong>District</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),

                        color=color1,

                        

                        fill_color=color1,

                        fill_opacity=0.7 ).add_to(map)

map
stationArr = df1[['lat', 'long','Day']].as_matrix()

from folium import plugins

map.add_children(plugins.HeatMap(stationArr, radius=25))

map
map_small = folium.Map(location=[20, 80], zoom_start=4,tiles='cartodbpositron')

scalar=0.5

for lat, lon, value, name, color1,size in zip(df1['lat'], df1['long'], df1['ConfirmedCases'], df1['District'],df1['color'],df1['size']):

    folium.CircleMarker([lat, lon],

                        radius=size*3*scalar,

                        popup = ('<strong>District</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),

                        color=color1,

                        

                        fill_color=color1,

                        fill_opacity=0.7 ).add_to(map_small)

map_small
map_small.add_children(plugins.HeatMap(stationArr, radius=20))

map_small
data_raw.head(10)
backupnotes=data_raw['backupnotes']

notes = data_raw['notes']

backupnotes=backupnotes.dropna()

notes = notes.dropna()

transmission=data_raw['typeoftransmission']

transmission=transmission.dropna()
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

from matplotlib.colors import LinearSegmentedColormap

from PIL import Image



colors = ["#BF0A30", "#002868"]

cmap = LinearSegmentedColormap.from_list("mycmap", colors)

mask = np.array(Image.open("../input/circle/circle.png"))



text = " ".join(str(each) for each in notes)

stopwords = set(STOPWORDS)

stopwords.update(["Details", "awaited"])

wordcloud = WordCloud(stopwords=stopwords,max_words=100,colormap=cmap, background_color="white",mask=mask).generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

text = " ".join(str(each) for each in backupnotes)

stopwords = set(STOPWORDS)

stopwords.update(["Details", "awaited"])

colors = ["#BF0A30", "#002868"]

cmap = LinearSegmentedColormap.from_list("mycmap", colors)

wordcloud = WordCloud(stopwords=stopwords,max_words=200,colormap=cmap, background_color="white",mask=mask).generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

text = " ".join(str(each) for each in transmission)

stopwords = set(STOPWORDS)

stopwords.update([ "Details", "awaited"])

colors = ["#BF0A30", "#002868"]

cmap = LinearSegmentedColormap.from_list("mycmap", colors)

wordcloud = WordCloud(stopwords=stopwords,max_words=200,colormap=cmap, background_color="white",mask=mask).generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()