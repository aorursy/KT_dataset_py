# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data
data1=pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")
dataall=pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")

x=dataall["Country/Region"]=="Italy"
italy=pd.DataFrame(dataall[x].reset_index(drop=True))

italy=italy.loc[28:53].reset_index(drop=True)

italy["Dates"]=["1.","2.","3.","4.","5.","6.","7.","8.","9.","10.","11.","12.","13.","14.","15.","16.","17.","18.","19.","20.","21.","22.","23.","24.","25.","26."]
hastaartis=[0]
hastaartis1=[0]
for i in range(len(data)-1):
    artis=data["Confirmed"][i+1]-data["Confirmed"][i]
    hastaartis.append(artis)
for i in range(len(italy)-1):
    artis=italy["Confirmed"][i+1]-italy["Confirmed"][i]
    hastaartis1.append(artis)
data["Hasta_Artışı"]=hastaartis
italy["Hasta_Artışı"]=hastaartis1
italy.head()
data["Dates"]=["1.","2.","3.","4.","5.","6.","7.","8.","9.","10.","11.","12.","13.","14.","15.","16.","17.","18.","19.","20.","21.","22.","23.","24.","25.","26."]

test_sayisi = [0,0,0,0,0,0,0,0,1981,3656,2953,0,3672,3952,5035,7286,7533,7641,9982,11515,15422,14396,18757,16160,19664,20065]
data['Test_Sayısı'] = test_sayisi
data.head()
olumoranı=[]
iyilesmeoranı=[]
for i in range(len(data)):
    
    olumorangun=data["Deaths"].iloc[i]/data["Confirmed"].iloc[i]
    olumoranı.append(olumorangun)
    iyilesorangun=data["Recovered"].iloc[i]/data["Confirmed"].iloc[i]
    iyilesmeoranı.append(iyilesorangun)
data["Ölüm oranı"]=olumoranı
data["İyileşme oranı"]=iyilesmeoranı

data.head()
olumoranı=[]
iyilesmeoranı=[]
for i in range(len(italy)):
    
    olumorangun=italy["Deaths"].iloc[i]/italy["Confirmed"].iloc[i]
    olumoranı.append(olumorangun)
    iyilesorangun=italy["Recovered"].iloc[i]/italy["Confirmed"].iloc[i]
    iyilesmeoranı.append(iyilesorangun)
gunler=["1.","2.","3.","4.","5.","6.","7.","8.","9.","10.","11.","12.","13.","14.","15.","16.","17.","18.","19.","20.","21.","22.","23.","24.","25.","26."]

italy["Ölüm_oranı"]=olumoranı
italy["İyileşme_oranı"]=iyilesmeoranı
                    
italy.head()
plt.figure(figsize=(13,8))
plt.plot(data.Dates,data.Confirmed,color="darkslategrey",linewidth=4,marker="o",markersize=8,markerfacecolor="darkslategrey")
plt.plot(data.Dates,italy.Confirmed,color="darkred",linewidth=4,marker="o",markersize=8,markerfacecolor="maroon")
plt.legend(["Türkiye","İtalya"])
plt.xlabel("Gün")
plt.ylabel("Vaka Sayısı")
plt.title("Türkiye ile İtalyanın Vaka Sayıları")
plt.grid(linewidth=2)
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.show()
plt.figure(figsize=(13,8))
plt.plot(data.Dates,data.Deaths,color="darkslategrey",linewidth=3,marker="o",markersize=8,markerfacecolor="darkslategrey")
plt.plot(data.Dates,italy.Deaths,color="darkred",linewidth=3,marker="o",markersize=8,markerfacecolor="maroon")
plt.legend(["Türkiye","İtalya"])
plt.xlabel("Gün")
plt.ylabel("Ölü Sayısı")
plt.title("Türkiye ve İtalyanın Ölüm Sayıları")
plt.grid(linewidth=1)
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.show()
plt.figure(figsize=(13,8))
plt.plot(data.Dates,data.Recovered,color="darkslategrey",linewidth=3,marker="o",markersize=8,markerfacecolor="darkslategrey")
plt.plot(data.Dates,italy.Recovered,color="darkred",linewidth=3,marker="o",markersize=8,markerfacecolor="maroon")
plt.xlabel("Günler")
plt.ylabel("İyileşen hasta sayısı")
plt.title("İtalya ve Türkiye'nin iyileşen hasta sayıları")
plt.grid(linewidth=1)
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.legend(["Türkiye","İtalya"])
plt.show()
plt.figure(figsize=(13,8))
plt.plot(data.Dates,data["Hasta_Artışı"],color="darkslategrey",linewidth=4,marker="o",markersize=8,markerfacecolor="darkslategrey")
plt.plot(data.Dates,italy["Hasta_Artışı"],color="darkred",linewidth=4,marker="o",markersize=8,markerfacecolor="maroon")
plt.xlabel("Gün")
plt.ylabel("Vaka Artışları")
plt.title("İtalya ve Türkiye'nin vaka artış sayıları")
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.legend(["Türkiye","İtalya"])
plt.grid(linewidth=2)
plt.show()
plt.figure(figsize=(13,8))
plt.plot(data.Dates,data.Confirmed,color="darkslategrey",linewidth=4,marker="o",markersize=8,markerfacecolor="darkslategrey",markeredgewidth=3)
plt.plot(data.Dates,data.Deaths,color="darkred",linewidth=4,marker="o",markersize=8,markerfacecolor="maroon",markeredgewidth=3)
plt.xlabel("Tarih")
plt.ylabel("Vaka Sayısı")
plt.title("Türkiye'deki Corona Vaka Sayıları")
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.grid()
plt.legend(["Vaka","Ölüm"])

plt.show()

plt.figure(figsize=(13,8))
plt.plot(data.Dates,data.Recovered,color="darkslategrey",linewidth=4,linestyle="--",marker="o",markersize=10,markerfacecolor="gold")
plt.xlabel("Tarih")
plt.ylabel("İyileşme Sayıları")
plt.title("İyileşen hasta sayısı")
plt.legend(loc='lower right')
plt.grid()
ax = plt.gca()
ax.set_facecolor("papayawhip")
plt.text(x=23,y=500,s="İyileşen hasta sayısı ani artışı")
plt.show()
plt.figure(figsize=(13,8))
plt.scatter(data.Hasta_Artışı,data.Test_Sayısı,c="darkred")
plt.xlabel("Hasta Artışı")
plt.ylabel("Test Sayısı")
plt.title("Hasta Artışı ve Test Sayısı arasındaki ilişki")
ax = plt.gca()
ax.set_facecolor("papayawhip")

plt.show()