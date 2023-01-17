# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode,iplot





cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/covid-turkey/Covid19_Turkey_Complete_Data.csv")

df["tarih"]=pd.to_datetime(df["tarih"])

df['aktif_vaka']=df['vaka']-df['vefat']-df['iyilesen']

df['aktif_vaka_entube_orani']=(df['toplam_entube']/df['aktif_vaka'])

df['aktif_vaka_yogun_bakim_orani']=(df['toplam_yogun_bakim']/df['aktif_vaka'])

df["vefat_orani"]=df["vefat"]/df["vaka"]

df["iyilesme_orani"]=df["iyilesen"]/df["vaka"]

df['gunluk_vaka']=df['vaka'].diff()

df['gunluk_vefat']=df['vefat'].diff()

df['gunluk_iyilesen']=df['iyilesen'].diff()

df['gunluk_entube_degisim']=df['toplam_entube'].diff()

df['gunluk_yogunbakim_degisim']=df['toplam_yogun_bakim'].diff()

df['test_vaka_orani']=(df['gunluk_vaka']/df['test_sayisi'])

guncel_tar=df['tarih'].max()

df_guncel=df[df['tarih']==guncel_tar]

vefat_guncel=df_guncel['vefat'].sum()

vaka_guncel=df_guncel['vaka'].sum()

iyilesen_guncel=df_guncel['iyilesen'].sum()

vefat_orani_guncel=df_guncel['vefat_orani'].sum()

iyilesme_orani_guncel=df_guncel['iyilesme_orani'].sum()

test_vaka_orani_guncel=df_guncel['test_vaka_orani'].sum()

gunluk_iyilesen_sayisi=df_guncel['gunluk_iyilesen'].sum()

gunluk_vaka_sayisi=df_guncel['gunluk_vaka'].sum()

gunluk_vefat_sayisi=df_guncel['gunluk_vefat'].sum()
print ('Bilgilerin Son Güncellenme Tarihi: {}'.format(guncel_tar))

print ('Toplam Vaka Sayısı: {:,.0f}'.format(vaka_guncel))

print ('Toplam Vefat Sayısı: {:,.0f}'.format(vefat_guncel))

print ('Toplam İyileşen Sayısı: {:,.0f}'.format(iyilesen_guncel))

print ('İyileşme Oranı: {:,.1%}'.format(iyilesme_orani_guncel))

print ('Ölüm Oranı: {:,.1%}'.format(vefat_orani_guncel))

print ('Yapılan Testlere Göre Vaka Oranı: {:,.1%}'.format(test_vaka_orani_guncel))

print ('Son Gün Vaka Sayısı: {:,.0f}'.format(gunluk_vaka_sayisi))

print ('Son Gün İyileşen Sayısı: {:,.0f}'.format(gunluk_iyilesen_sayisi))

print ('Son Gün Vefat Sayısı: {:,.0f}'.format(gunluk_vefat_sayisi))
plt.figure(figsize=(15,10))

sns.lineplot(x="tarih",y="aktif_vaka",data=df, label="Aktif Vaka")

sns.lineplot(x="tarih",y="vefat",data=df,label="Vefat")

sns.lineplot(x="tarih",y="iyilesen",data=df,label="İyileşen")

plt.legend(loc="upper left",prop={'size': 20})

plt.show()

plt.figure(figsize=(20,10))

sns.lineplot(x="tarih",y="aktif_vaka_entube_orani",data=df[df["tarih"]>="2020-03-27"],label="Entübe Oranı")

sns.lineplot(x="tarih",y="aktif_vaka_yogun_bakim_orani",data=df[df["tarih"]>="2020-03-27"],label="Yoğun Bakım Oranı")

sns.lineplot(x="tarih",y="vefat_orani",data=df[df["tarih"]>="2020-03-27"],label="Vefat Oranı")

plt.legend(loc="upper right",prop={'size': 20})

plt.show()
trace1=go.Scatter(

                    x=df.tarih,

                    y=df.vaka,

                    mode="lines",

                    name="Vaka Sayısı",

                    marker=dict(color='rgba(196,174,7,0.8)'),

#                    text="Vaka"

                )

    

trace2=go.Scatter(

                    x=df.tarih,

                    y=df.iyilesen,

                    mode="lines",

                    name="İyileşen Hasta Sayısı",

                    marker=dict(color='rgba(5,142,49,0.8)'),

#                    text="İyileşen"

                )



trace3=go.Scatter(

                    x=df.tarih,

                    y=df.vefat,

                    mode="lines",

                    name="Vefat Sayısı",

                    marker=dict(color='rgba(170,6,6,0.8)'),

 #                   text="Vefat"

                )



data=[trace1,trace2,trace3]

layout=dict(title="Genel Tablo",

           xaxis=dict(title="Tarih",ticklen=1,zeroline=False)

           )

            

    

fig=dict(data=data,layout=layout)

iplot(fig)
df2=df[df["tarih"]>="2020-03-27"]

trace4=go.Scatter(

                    x=df2.tarih,

                    y=df2.aktif_vaka_entube_orani,

                    mode="lines",

                    name="Entübe Oranı",

                    marker=dict(color='rgba(96,3,87,0.8)'),

#                    text="Entübe"

                )



trace5=go.Scatter(

                    x=df2.tarih,

                    y=df2.aktif_vaka_yogun_bakim_orani,

                    mode="lines",

                    name="Yoğun Bakım Oranı",

                    marker=dict(color='rgba(196,174,7,0.8)'),

#                    text="Yoğun Bakım"

                )





trace6=go.Scatter(

                    x=df2.tarih,

                    y=df2.vefat_orani,

                    mode="lines",

                    name="Vefat Oranı",

                    marker=dict(color='rgba(170,6,6,0.8)'),

#                    text="Vefat"

                )





trace7=go.Scatter(

                    x=df2.tarih,

                    y=df2.iyilesme_orani,

                    mode="lines",

                    name="İyileşme Oranı",

                    marker=dict(color='rgba(5,142,49,0.8)'),

#                    text="Vefat"

                )



data=[trace4,trace5,trace6]

layout=dict(title="Önemli Oranlar",

           xaxis=dict(title="Tarih",ticklen=1,zeroline=False),

           yaxis=dict(tickformat=".2%")

           )

            

    

fig=dict(data=data,layout=layout)

iplot(fig)
trace8=go.Scatter(

                    x=df.tarih,

                    y=df.gunluk_vaka,

                    mode="lines",

                    name="Günlük Vaka Sayısı",

                    marker=dict(color='rgba(196,174,7,0.8)'),

                )



trace9=go.Scatter(

                    x=df.tarih,

                    y=df.gunluk_vefat,

                    mode="lines",

                    name="Günlük Vefat Sayısı",

                    marker=dict(color='rgba(170,6,6,0.8)'),

                )

trace10=go.Scatter(

                    x=df.tarih,

                    y=df.gunluk_iyilesen,

                    mode="lines",

                    name="Günlük İyileşme Sayısı",

                    marker=dict(color='rgba(5,142,49,0.8)'),

                )



data=[trace8,trace9,trace10]

layout=dict(title="Günlük Adetler",

           xaxis=dict(title="Tarih",ticklen=1,zeroline=False)

           )

            

    

fig=dict(data=data,layout=layout)

iplot(fig)
trace11=go.Scatter(

                    x=df.tarih,

                    y=df.test_vaka_orani,

                    mode="lines",

                    name="Test Vaka Oranı",

                    marker=dict(color='rgba(96,3,87,0.8)'),

                )



data=[trace11]

layout=dict(title="Günlük Yapılan Test Adetlerine Göre Tespit Edilen Vaka Oranı",

           xaxis=dict(title="Tarih",ticklen=1,zeroline=False),

           yaxis=dict(tickformat=".2%")

           )

            

    

fig=dict(data=data,layout=layout)

iplot(fig)