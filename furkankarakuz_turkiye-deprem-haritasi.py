#Harita özelliğini kullanmak için folium modülü indirilir



!pip install folium
#Modüller dahil edilir



import folium

from folium import plugins

import numpy as np

import pandas as pd
#İnternet adresinden verileri almak için requests ve BeautifulSoup modülleri dahil edilir



import requests

from bs4 import BeautifulSoup



url="http://www.koeri.boun.edu.tr/scripts/lst9.asp"



sonuc=requests.get(url)

sonuc2=BeautifulSoup(sonuc.content,"lxml")

liste=[]

sonuc2=sonuc2.text

sonuc2=sonuc2.strip()

sonuc2=sonuc2.split("\r")



liste_enlem=[]

liste_boylam=[]

liste_yer=[]



for i in range(14,len(sonuc2)-20):

    i=sonuc2[i].split()

    if(len(i)>=9):

        liste_enlem.append(i[2])

        liste_boylam.append(i[3])

        liste_yer.append(str(i[8])+" "+str(i[9]))
#Alınan veriler tablo haline dönüştürülür



df=pd.DataFrame({"Enlem":liste_enlem,"Boylam":liste_boylam,"Yer":liste_yer})
#Verilerin ilk 5 satırı gösterilir



df.head()
#Enlem ve Boylam Bilgileri float yani sürekli sayısal değerine dönüştürülür



df["Enlem"]=df["Enlem"].astype("float64")

df["Boylam"]=df["Boylam"].astype("float64")
#Sadece Enlem ve Boylam olacak şekilde filtreleme işlemi yapılır



location=df[["Enlem","Boylam"]]
#Enlem ve boylam bilgilerine göre bölgelerin konum haritası oluşturuldu



m = folium.Map(location=[40, 32],zoom_start=7)

for i in range(0,len(df)):

    folium.Marker([df.iloc[i]['Enlem'], df.iloc[i]['Boylam']], popup=df.iloc[i]['Yer']).add_to(m)

m
#Enlem ve boylam bilgilerine göre bölgelerin konum haritası oluşturuldu



m = folium.Map(location=[40, 32],zoom_start=7)

for i in range(0,len(df)):

    folium.CircleMarker(location=[df.iloc[i]['Enlem'], df.iloc[i]['Boylam']],radius=15, fill_color='red',fill=True).add_to(m)

m
#Enlem ve boylam bilgilerine göre bölgelerin konum haritası oluşturuldu



m = folium.Map(location=[41, 30],zoom_start=8)

for i in range(0,len(df)):

    folium.Marker([df.iloc[i]['Enlem'], df.iloc[i]['Boylam']], popup=df.iloc[i]['Yer']).add_to(m)

    folium.CircleMarker(location=[df.iloc[i]['Enlem'], df.iloc[i]['Boylam']],radius=30, fill_color='red',fill=True).add_to(m)

m
#Enlem ve boylam bilgilerine göre bölgelerin konum haritası oluşturuldu



m = folium.Map([40 ,32], zoom_start=6,width="%100",height="%100")

location=df[["Enlem","Boylam"]]



plugins.MarkerCluster(location).add_to(m)



m
#Enlem ve boylam bilgilerine göre bölgelerin ısı haritası oluşturuldu



m=folium.Map(location=[40,33],tiles="OpenStreetMap",zoom_start=6)

heat_df=df[["Enlem","Boylam"]]

heat_data=list(zip(df.Enlem, df.Boylam))

folium.plugins.HeatMap(heat_data).add_to(m)

m