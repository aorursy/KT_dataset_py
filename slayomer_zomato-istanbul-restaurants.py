import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/zomato-restaurants-autoupdated-dataset/zomato.csv")

df.head()
df.info()
top20 = df[df.City == "İstanbul"].sort_values("Average Cost for two",ascending=False)[["Restaurant Name","Longitude","Latitude","Average Cost for two","Currency","Price range","Aggregate rating","Rating color","Rating text","Votes"]].head(20)

top20
df_istanbul = df[df.City == "İstanbul"]



import folium



lat_mean = df_istanbul.Latitude.mean()

lon_mean = df_istanbul.Longitude.mean()



m = folium.Map(location=[lat_mean, lon_mean],zoom_start=11)



for i in df_istanbul.index:

    html = df_istanbul.loc[i].to_frame().to_html()

    folium.Marker([df_istanbul.loc[i].Latitude,df_istanbul.loc[i].Longitude],

                  tooltip=df_istanbul.loc[i,"Restaurant Name"],

                  popup = folium.Popup(folium.IFrame(html=html, width=300, height=400)),

#                   icon=folium.Icon(icon=df_istanbul.loc[i,"Aggregate rating"])

                 ).add_to(m)

m