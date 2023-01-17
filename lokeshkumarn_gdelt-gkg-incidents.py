import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gdelt as gd



from matplotlib import pyplot as plt



import spacy
gs = gd.gdelt(version=2.0)
#IRMA - 30 August 2017 – 13 September 2017

df=gs.Search(date = ['2017 08 30','2017 09 13'],table='gkg',normcols=True,output='df')
df.shape
df.isnull().sum()
df.drop(columns = ['dates','socialimageembeds','socialvideoembeds','quotations','translationinfo'],inplace=True)
df.isnull().sum()
df_loc = df[~df['v2locations'].isnull()]
df_loc['v2locations'].reset_index(drop=True)[0].split(';')
df_loc[df_loc['v2locations'].str.contains('US')].shape
for r in df_loc[df_loc['v2locations'].str.contains('US')]['v2locations'][:50]:

    print(r)

    print()
df_loc_us = df_loc[df_loc['v2locations'].str.contains('US')]
df_loc_us['url'] = df_loc_us['documentidentifier'].str.lower()
df_loc_us_irma = df_loc_us[df_loc_us['url'].str.contains('irma')]
df_loc_us_irma['locations']
df_loc_us_irma['locations'].reset_index(drop=True)[0]
locations=[]

for r in df_loc_us_irma['v2locations']:

    for addr in r.split(';'):

        addr_split= addr.split('#')        

        try:

            locations.append({'latitude':float(addr_split[5]),'longtitude':float(addr_split[6])})

        except:

            print(addr_split[5],addr_split[6])
len(locations)
df_locations = pd.DataFrame(locations)
df_locations.isnull().sum()
df_locations.head()
!pip install folium
df_locations[(df_locations['latitude'] < 34.0) & (df_locations['latitude'] > -95.0)]
import folium

m = folium.Map(location=[37.0902, -95.7129],zoom_start=4)

for i,r in df_locations[(df_locations['latitude'] < 34.0) & (df_locations['latitude'] > -95.0)][:2000].iterrows():

    #print(r['latitude'], r['longtitude'])

    pop_text = str([r['latitude'], r['longtitude']])

    folium.CircleMarker(location=[r['latitude'], r['longtitude']],

                        radius=5,

                        color='crimson', 

                        #popup=pop_text,

                        fill=True,

                        fill_color='crimson').add_to(m)

m
m.save('map.html')
news_txt = '''Multiple people killed in fiery crash near Denver



Semi hit cars stuck in evening rush hour jam





By:  Joe Sterling, Amanda Watts and Joe Sutton, CNN 

  



Posted: Apr 25, 2019 10:58 PM MDT



Updated: Apr 26, 2019 02:12 PM MDT

'''
news_txt
nlp = spacy.load("en_core_web_sm")
doc =nlp(news_txt)
[(tok,tok.dep_) for tok in doc if tok.pos_=='PROPN']
df = gs.Search(['2019 04 24','2019 04 27'],table='gkg',output='df', normcols=True)
df.shape
df[(~df['v2counts'].isnull()) & (df['v2locations'].str.contains('Denver')) 

   & (df['documentidentifier'].str.contains('Denver'))]['documentidentifier']