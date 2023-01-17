
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
sehir_data = pd.read_csv("../input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
data.rename(columns={"Last_Update":"Tarih","Confirmed":"Vaka","Deaths":"Vefat","Recovered":"Tedavi_Edilen"},inplace=True)
data.drop("Province/State",axis=1,inplace=True)

#Manuel Giriş!
test_sayisi = [0,0,0,0,0,0,0,0,1981,3656,2953,1738,3672,3952,5035,7286,7533,
               7641,9982,11535,15422,14396,18757,16160,19664
               ,20065,21400,20023,24900,28578,30864,33170,35720,
               34456,33070,34090,40427,40270,40520,35344,39703,39429,37535,40962,38351,38308,
               30177,20143,29230,43498,42004,41431,36318,24001,35771,33283,30303,30395,33687,35605,36187

]

data["Test_Sayısı"] = test_sayisi

# uzunlukları 10 olmalı !
son10_gunlukTest = test_sayisi[-10:]
#Manuel Giriş !
son10_gunlukVaka = [2188,1983,1670,1614,1832,2253,1977,1848,1546,1547]
vaka_orani = [0]
olum_orani = [0]
vaka_artisi = [0]
olum_artisi = [0]
test_orani = [0]
test_artisi = [0]

aktif_hasta = data["Vaka"]-(data["Vefat"]+data["Tedavi_Edilen"])
pasif_hasta = data["Vefat"]+data["Tedavi_Edilen"]


for i in range(len(data)-1):
    testOrani = round((data["Test_Sayısı"][i+1] - data["Test_Sayısı"][i])/
                      data["Test_Sayısı"][i],2)
    
    testArtisi = data["Test_Sayısı"][i+1] - data["Test_Sayısı"][i]
    
    vakaArtisi = data["Vaka"][i+1] - data["Vaka"][i]
    
    vakaOrani = round((data["Vaka"][i+1]-data["Vaka"][i])/
                     data["Vaka"][i],2)
    
    olumOrani = round((data["Vefat"][i+1] - data["Vefat"][i])/
                     data["Vefat"][i],2)
    
    olumArtisi = data["Vefat"][i+1] - data["Vefat"][i]
        
        
    test_orani.append(testOrani)
    test_artisi.append(testArtisi)
    vaka_artisi.append(vakaArtisi)
    vaka_orani.append(vakaOrani)
    olum_orani.append(olumOrani)
    olum_artisi.append(olumArtisi)
    
    
    
    
data["Test Artış Sayısı"] = test_artisi
data["Test Artış Oranı"] = test_orani
data["Vaka Artış Sayısı"] = vaka_artisi
data["Vaka Artış Oranı"] = vaka_orani
data["Vefat Artış Sayısı"] = olum_artisi
data["Vefat Artış Oranı"] = olum_orani
data["Aktif Hasta Sayısı"] = aktif_hasta
data["Pasif Hasta Sayısı"] = pasif_hasta

vaka_test_yuzdesi = []
for i in range(len(son10_gunlukTest)):
    vaka_test_yuzdesi.append(round(((son10_gunlukVaka[i]/son10_gunlukTest[i])*100),2))
    


data.fillna(0, inplace=True)
data = data.replace([np.inf,-np.inf], np.nan)
data.fillna(0, inplace=True)

    

data
vaka = go.Scatter(
    x = data.Tarih,
    y = data.Vaka,
    mode = "lines+markers",
    name = "Vaka",
    marker = dict(color="rgba(171, 28, 28, 0.68)"),
    text = data.Vaka
                 )
layout = dict(title="Logaritmik Toplam Vaka Sayısı",
              xaxis = dict(title = "Tarih"),
              yaxis = dict(title = "Kişi Sayısı"),
              xaxis_tickangle = -45,
              yaxis_type = "log"
             )
fig = dict(data = vaka, layout = layout)
iplot(fig)
olum = go.Scatter(
    x = data.Tarih,
    y = data.Vefat,
    mode = "lines+markers",
    name = "Vefat",
    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
    text = data.Vefat
)

tedavi = go.Scatter(
    x = data.Tarih,
    y = data.Tedavi_Edilen,
    mode = "lines+markers",
    name = "Tedavi Edilen",
    marker = dict(color = "rgba(0, 180, 0, 0.8)"),
    text = data.Tedavi_Edilen
)

data2 = [olum,tedavi]

layout = dict(
    title = "Toplam Ölüm Ve Toplam Tedavi Sayıları",
    xaxis = dict(title = "Tarih"),
    yaxis = dict(title = "Kişi Sayısı"),
    xaxis_tickangle = -45,
             )

fig = dict(data = data2, layout = layout)
iplot(fig)
toplam_test = go.Scatter(
    x = data.Tarih,
    y = data["Test_Sayısı"],
    mode = "lines+markers",
    name = "Test Sayısı",
    marker = dict(color = 'rgba(179, 80, 0, 0.8)'),
    text = data["Test_Sayısı"]
)


data2 = [toplam_test]

layout = dict(
    title = "Günlük Test Sayısı",
    xaxis = dict(title = "Tarih"),
    yaxis = dict(title = "Test Sayısı"),
    xaxis_tickangle = -45,
             )

fig = dict(data = data2, layout = layout)
iplot(fig)

test_artis_sayisi = go.Scatter(
    x = data.Tarih[-30:],
    y = data["Test Artış Sayısı"][-30:],
    mode = "lines+markers",
    name = "Test Artış Sayısı",
    marker = dict(color = 'rgba(224, 213, 0, 1)'),
    text = data["Test Artış Sayısı"][-30:]
)


data2 = [test_artis_sayisi]

layout = dict(
    title = "Önceki Güne Göre Yapılan Test Farkı",
    xaxis = dict(title = "Tarih"),
    yaxis = dict(title = "Önceki Güne Göre Test Farkı"),
    xaxis_tickangle = -45,
             )

fig = dict(data = data2, layout = layout)
iplot(fig)
testArtis_oran_graf = go.Scatter(
    x = data.Tarih[-30:],
    y = data["Test Artış Oranı"][-30:],
    mode = "lines+markers",
    name = "Test Artış Oranı",
    marker = dict(color = "rgba(0, 179, 140, 0.8)"),
    text = data["Test Artış Oranı"][-30:]
)

vaka_oran_graf = go.Scatter(
    x = data.Tarih[-30:],
    y = data["Vaka Artış Oranı"][-30:],
    mode = "lines+markers",
    name = "Vaka Artış Oranı",
    marker = dict(color = "rgba(107, 0, 179, 0.8)"),
    text = data["Vaka Artış Oranı"][-30:]
)

data2 = [testArtis_oran_graf,vaka_oran_graf]

layout = dict(
    title = "Son 30 Günün Test Artış Oranı ve Vaka Artış Oranı",
    xaxis = dict(title = "Tarih"),
    yaxis = dict(title = "Oran"),
    xaxis_tickangle = -45
)

fig = dict(data=data2,layout=layout)
iplot(fig)


vaka_yuzdesi = go.Scatter(
    x = data.Tarih[-10:],
    y = vaka_test_yuzdesi,
    mode = "lines+markers",
    name = "Vaka/Test Yüzdesi",
    marker = dict(color = "rgba(255, 56, 56, 0.88)"),
    text = vaka_test_yuzdesi
)

data2 = [vaka_yuzdesi]
layout = dict(
    title = "Son 10 Günün Vaka/Test Yüzdesi",
    xaxis = dict(title = "Tarih"),
    yaxis = dict(title = "Yüzde(%)"),
    xaxis_tickangle = -45
             )

fig = dict(data = data2, layout = layout)
iplot(fig)
fig = go.Figure(
    data = [
        go.Bar(
            name = "Vaka",
            x = data["Tarih"][-10:],
            y = data["Vaka"][-10:],
            marker_color = "rgba(214, 138, 244, 0.8)",
            marker_line_color="rgba(146, 18, 196, 1)",
              ),
        go.Bar(
            name = "Vefat",
            x = data["Tarih"][-10:],
            y = data["Vefat"][-10:],
            marker_color = "rgba(241, 101, 101, 0.8)",
            marker_line_color="rgba(235, 36, 36, 1)",
        ),
        go.Bar(
            name = "Tedavi_Edilen",
            x = data["Tarih"][-10:],
            y = data["Tedavi_Edilen"][-10:],
            marker_color = "rgba(131, 241, 101, 0.8)",
            marker_line_color="rgba(63, 219, 20, 1)",
        )
    ]
)

fig.update_layout(
    barmode = "group",
    title_text = "Son 10 Günün TOPLAM Vaka, TOPLAM Vefat, TOPLAM İyileşen Hasta Sayıları",
    xaxis_tickangle = -45
)

fig.show()
fig = go.Figure(
    data = [
        go.Bar(
            name = "Aktif Hasta",
            x = data["Tarih"],
            y = data["Aktif Hasta Sayısı"],
            marker_color = "rgba(84, 207, 162, 0.8)",
            marker_line_color="rgba(146, 18, 196, 1)",
              ),
        go.Bar(
            name = "Pasif Hasta",
            x = data["Tarih"],
            y = data["Pasif Hasta Sayısı"],
            marker_color = "rgba(241, 101, 101, 0.8)",
            marker_line_color="rgba(235, 36, 36, 1)",
        )
    ]
)

fig.update_layout(
    barmode = "group",
    title_text = "Aktif Hasta [Vaka - (İyileşen + Vefat)] Ve Pasif Hasta [İyileşen + Vefat] Sayısı",
    xaxis_tickangle = -45
)

fig.show()
sehir_data
sehir_data.rename(columns = {"Province":"Şehir", "Number of Case":"Vaka Sayısı"},
                 inplace=True)
sehir_data.sort_values(by=["Vaka Sayısı"], ascending=False, inplace=True)
fig = px.pie(
    sehir_data.head(),
    values = "Vaka Sayısı",
    names = "Şehir",
    title = "En Yüksek Vaka Sayısına Sahip 5 Şehir"
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.show()
sehir_temp_data = sehir_data[5:]
fig = px.pie(
    sehir_temp_data,
    values = "Vaka Sayısı",
    names = "Şehir",
    title = "Diğer Şehirlerdeki Vaka Sayısı",
    hover_data =["Vaka Sayısı"]
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.show()
fig = px.bar(
    sehir_data[0:5],
    x = "Şehir",
    y = "Vaka Sayısı",
    title = "En Yüksek Vaka Sayısına Sahip 5 Şehir"
)
fig.update_layout(barmode="group")
fig.update_traces(
    marker_color='rgba(240, 92, 92, 0.6)',
    marker_line_color="rgba(191, 18, 18, 1)",
)
fig.show()
fig = px.bar(
    sehir_data[5:15],
    x = "Şehir",
    y = "Vaka Sayısı",
    title = "En Yüksek Vaka Sayısına Sahip [5-15] Aralığındaki Şehirler")
fig.update_layout(barmode="group")
fig.update_traces(
    marker_color='rgba(215,137,86,0.6)',
    marker_line_color="rgba(153, 83, 36, 1)",
)
fig.show()
fig = px.bar(
    sehir_data[15:],
    x = "Şehir",
    y = "Vaka Sayısı",
    title = "En Yüksek Vaka Sayısına Sahip [15-81] Aralığındaki Şehirler",
    
)
fig.update_layout(barmode="group")
fig.update_traces(marker_color='rgb(158,202,225)',
                  marker_line_color="rgb(8,48,107)",
                 )
fig.show()