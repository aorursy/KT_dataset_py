
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
conf_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recv_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
conf_df.head()
conf_df.columns
print(conf_df.columns)
dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', 
         '1/29/20', '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', 
         '2/5/20', '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', 
         '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',
         '2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20',
'2/27/20','2/28/20','2/29/20','3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20',
'3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20','3/13/20','3/14/20','3/15/20',
'3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20','3/23/20','3/24/20','3/25/20','3/26/20',
        '3/27/20', '3/28/20', '3/29/20', '3/30/20', '3/31/20', '4/1/20',
       '4/2/20', '4/3/20', '4/4/20', '4/5/20', '4/6/20', '4/7/20', '4/8/20',
       '4/9/20', '4/10/20', '4/11/20', '4/12/20', '4/13/20', '4/14/20',
       '4/15/20', '4/16/20', '4/17/20', '4/18/20', '4/19/20', '4/20/20', '4/21/20', '4/22/20',
        '4/23/20', '4/24/20', '4/25/20', '4/26/20', '4/27/20', '4/28/20',
        '4/29/20', '4/30/20', '5/1/20', '5/2/20', '5/3/20', '5/4/20', '5/5/20', '5/6/20',
       '5/7/20', '5/8/20', '5/9/20', '5/10/20', '5/11/20', '5/12/20', '5/13/20', '5/14/20', '5/15/20', '5/16/20',
       '5/17/20', '5/18/20', '5/19/20', '5/20/20', '5/21/20', '5/22/20', '5/23/20', '5/24/20',
       '5/25/20', '5/26/20', '5/27/20', '5/28/20']

conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)
full_table.head()
# uygun veri formatına dönüştürme
full_table['Date'] = pd.to_datetime(full_table['Date'])
full_table['Recovered'] = full_table['Recovered'].astype('float')

# Mainland Çin yerine sadece Çin
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# eksik değerli sütunları 0 ile doldurma ('Confirmed', 'Deaths', 'Recovered')
full_table[['Confirmed', 'Deaths', 'Recovered']] = full_table[['Confirmed', 'Deaths', 'Recovered']].fillna(0)
full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')

# Diamond Princess gemisindeki vakalar
ship = full_table[full_table['Province/State']=='Diamond Princess cruise ship']

# tabloya doldurma
full_table = full_table[full_table['Province/State']!='Diamond Princess cruise ship']
full_table.head()
# dataframe leri türetmek
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
row_latest = full_latest[full_latest['Country/Region']!='China']

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
temp = full_latest.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()
temp.style.background_gradient(cmap='Pastel1_r')
#Vaka sayısı en fazla olan 10 ülke
temp_f = full_latest_grouped[['Country/Region', 'Confirmed']]
temp_f = temp_f.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.head(10).style.background_gradient(cmap='Pastel1_r')
#ölüm sayısı en fazla olan ülkeden aza doğru sıralama
temp_flg = full_latest_grouped[['Country/Region', 'Deaths']]
temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)
temp_flg = temp_flg.reset_index(drop=True)
temp_flg = temp_flg[temp_flg['Deaths']>0]
temp_flg.style.background_gradient(cmap='Pastel1_r')
# Covid-19 vakalarının sınıflara göre tanımlanması
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Aktif vaka tanımlama: Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# Veri tablosunda Çini yeniden adlandırma
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# eksik değerleri doldurma 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[cases] = full_table[cases].fillna(0)

# gemideki vakalar
ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]

# çin ve satırlar
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']

# en son
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
row_latest = full_latest[full_latest['Country/Region']!='China']

# yoğunlaştırılmış son hali
full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
#Adım 3: Ülkelere toplam vakaları akıllıca veren bir tablo oluşturmak

temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')
# dataseti okuma
data= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.head()
# Her ülke için toplam onaylanmış vaka sayısıyla bir veri seti oluşturma
Number_of_countries = len(data['Country/Region'].value_counts())


cases = pd.DataFrame(data.groupby('Country/Region')['Confirmed'].sum())
cases['Country/Region'] = cases.index
cases.index=np.arange(1,Number_of_countries+1)

global_cases = cases[['Country/Region','Confirmed']]
#global_cases.sort_values(by=['Confirmed'],ascending=False)
global_cases
#Türkiye
Turkey = data[data['Country/Region']=='Turkey']
Turkey
#Farklı değişkenler arasında ilişki kurma
corr = data.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
'''Plot kullanarak pasta grafiği oluşturma'''

def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace
'''Görselleştirme .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True) # Jupyter not defterinde "plotly" çevridışı kullanmak için
py.iplot([pie_plot(data['Province/State'].value_counts(), ['cyan', 'gold'], 'State')]) #Çindeki şehirlerin pasta grafiği
py.iplot([pie_plot(data['Country/Region'].value_counts(), ['cyan', 'gold'], 'Country')]) #Ülkelerin pasta grafiği
py.iplot([pie_plot(data['Deaths'].value_counts(), ['cyan', 'gold'], 'Deaths')]) #Ölümlerin pasta grafiği
py.iplot([pie_plot(data['Recovered'].value_counts(), ['cyan', 'gold'], 'Recovered')]) #İyileşenlerin pasta grafiği
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        import warnings
warnings.filterwarnings('ignore')
province_data= pd.read_csv("/kaggle/input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data.tail()
data.info()
data.rename(columns = {"Last_Update" : "Tarih", "Confirmed" : "Vaka", "Deaths" : "Vefat", "Recovered" : "Tedavi_Edilen"}, inplace = True)
data.drop("Province/State", axis = 1, inplace = True)
test_sayisi = [0,0,0,0,0,0,0,0,1981,3656,2953,1738,3672,3952,5035,7286,7533,7641,9982,11535,15422,14396,18757,16160,19664
               ,20065,21400,20023,24900,28578,30864,33170,35720,34456,33070,34090,40427,40270,40520,35344,39703,39429,37535
               ,40962,38351,38308,30177,20143,29230,43498,42004,41431,36318,24001,35771,33283,30303,30395,33687,35605,36187
               ,32722,37351,33332,34821,38565,42236,35369,25141,25382,20838,33633,37507,40178,24589]
data['Test Sayısı'] = test_sayisi #Test sayısı açıklanmayan günler 0 olarak belirtilmiştir. 
toplamYogunBakim = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,344,445,568,725,847,979,1101,1251,1311,1381,1415,1474,1492,1552,1667,1626,1665,1786,1809,1820,1854,1845,1894
                    ,1922,1909,1865,1814,1816,1790,1782,1776,1736,1621,1574,1514,1480,1445,1424,1384,1338,1278,1260,1219,1168,1154,1126,1045,998,963,944,906,914
                    ,903,882,877,820,800,775,769]
data['toplamYogunBakim'] = toplamYogunBakim #Yogun bakim hastasi sayısı açıklanmayan günler 0 olarak belirtilmiştir.
toplamEntubeSayisi = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,241,309,394,523,622,692,783,867,909,935,966,987,995,1017,1062,1021,978,1063,1087
                      ,1052,1040,1014,1054,1031,1033,1006,985,982,929,900,883,882,845,831,803,818,778,766,727,707,669,665,653,628,598,578
                      ,576,535,508,490,474,468,463,455,445,424,401,388,358]
data['toplamEntubeSayisi'] = toplamEntubeSayisi #Entübe sayısı açıklanmayan günler 0 olarak belirtilmiştir.
vaka_oran = [0] #ilk değerden önce bir değer olmadığı için ilk değeri 0 verdik
olu_oran = [0]
vaka_artis= [0]
olu_artis = [0]
test_oran = [0]
test_artis= [0]
yogun_bakim_oran = [0]
yogun_bakim_artis = [0]
entube_hasta_oran = [0]
entube_hasta_artis = [0]
yogun_bakim_vaka_orani = [0]
entube_vaka_orani = [0]
aktif_vaka = [0]

for i in range(len(data)-1):
    testOran =  round((data["Test Sayısı"][i+1] - data["Test Sayısı"][i]) / data["Test Sayısı"][i], 2)
    testArtis = data["Test Sayısı"][i+1] - data["Test Sayısı"][i] 
    vakaOran = round((data["Vaka"][i+1] - data["Vaka"][i]) / data["Vaka"][i], 2)
    vakaArtis = data["Vaka"][i+1] - data["Vaka"][i]
    oluOran =  round((data["Vefat"][i+1] - data["Vefat"][i]) / data["Vefat"][i], 2)
    olumArtis = data["Vefat"][i+1] - data["Vefat"][i]
    yogunBakimOran =  round((data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i]) / data["toplamYogunBakim"][i], 2)
    yogunBakimArtis = data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i] 
    entubehastaOran =  round((data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]) / data["toplamEntubeSayisi"][i], 2)
    entubeHastaArtis = data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]
    yogunBakimVakaOrani = round((data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i]) / data["Vaka"][i], 5)
    entubeVakaOrani = round((data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]) / data["Vaka"][i], 5)
    aktifVaka = data["Vaka"][i] - data["Tedavi_Edilen"][i]
    

    test_oran.append(testOran)
    test_artis.append(testArtis)
    vaka_oran.append(vakaOran)
    olu_oran.append(oluOran)
    vaka_artis.append(vakaArtis)
    olu_artis.append(olumArtis)
    yogun_bakim_oran.append(yogunBakimOran)
    yogun_bakim_artis.append(yogunBakimArtis)
    entube_hasta_oran.append(entubehastaOran)
    entube_hasta_artis.append(entubeHastaArtis)
    yogun_bakim_vaka_orani.append(yogunBakimVakaOrani)
    entube_vaka_orani.append(entubeVakaOrani)
    aktif_vaka.append(aktifVaka)


data["Test Artış Sayısı"] = test_artis
data["Test Artış Oranı"] = test_oran
data["Vaka Artış Sayısı"] = vaka_artis
data["Vaka Artış Oranı"] = vaka_oran
data["Vefat Artış Sayısı"] = olu_artis
data["Vefat Artış Oranı"] = olu_oran
data["Yogun Bakim Artış Sayısı"] = yogun_bakim_artis
data["Yogun Bakim Artış Oranı"] = yogun_bakim_oran
data["Entube Hasta Artış Sayısı"] = entube_hasta_artis
data["Entube Hasta Artış Oranı"] = entube_hasta_oran
data["Yoğun Bakım/Vaka Oranı"] = yogun_bakim_vaka_orani
data["Entube/Vaka Oranı"] = entube_vaka_orani
data['Aktif Vaka'] = aktif_vaka

data.fillna(0, inplace = True) # 0/0'dan kaynaklanan NaN degerlerini 0 ile degistirdik.
data = data.replace([np.inf, -np.inf], np.nan) #sayi/0 sonsuzlugunu 0 olarak degistirdik.
test_pozitif=[]
for i in range(len(data)):
    test_pozitif_orani =  round((data["Vaka Artış Sayısı"][i] / data["Test Sayısı"][i]), 2)
    test_pozitif.append(test_pozitif_orani)
        
data["Pozitif/Test Oranı"] = test_pozitif
data = data.replace([np.inf, -np.inf], np.nan)
data.fillna(0, inplace = True)
data
fig = go.Figure(data=[
    go.Bar(name='Vaka', x=data['Tarih'], y=data['Vaka'], marker_color='rgba(135, 206, 250, 0.8)'),
    go.Bar(name='Vefat', x=data['Tarih'], y = data['Vefat'], marker_color='rgba(255, 0, 0, 0.8)'),
    go.Bar(name='Tedavi_Edilen', x=data['Tarih'] , y=data['Tedavi_Edilen'], marker_color='rgba(0, 255, 0, 0.8)')
])
fig.update_layout(barmode='group', title_text='Türkiye Günlük Vaka, Vefat ve Taburcu Hasta Sayıları', xaxis_tickangle=-45)
fig.show()
fig = go.Figure(data=[
    go.Bar(name='Vefat', x=data['Tarih'], y = data['Vefat'], marker_color='rgb(255, 0, 0)'),
    go.Bar(name='Tedavi_Edilen', x=data['Tarih'] , y=data['Tedavi_Edilen'], marker_color='rgb(0, 255, 0)')
])
fig.update_layout(barmode='group', title_text='Türkiye Günlük Vefat ve Taburcu Hasta Sayıları', xaxis_tickangle=-45)
fig.show()
mart17Sonrasi = data.loc[data['Tarih'] > '3/17/2020']
mart15Sonrasi = data.loc[data['Tarih'] > '3/15/2020']
mart26Sonrasi = data.loc[data['Tarih'] > '3/27/2020']
vaka = go.Scatter(x = data.Tarih,
                    y = data.Vaka,
                    mode = "lines+markers",
                    name = "Vaka / Cases",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= data.Vaka
                   )
olum = go.Scatter(x = data.Tarih,
                    y = data.Vefat,
                    mode = "lines+markers",
                    name = "Vefat / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data.Vefat
                   )
taburcu = go.Scatter(x = data.Tarih,
                    y = data.Tedavi_Edilen,
                    mode = "lines+markers",
                    name = "Tedavi/Recovered",
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),
                    text= data.Tedavi_Edilen
                   )
data2 = [vaka, olum, taburcu]
layout = dict(title = "Türkiye'deki Covid-19 Vaka, Vefat ve Tedavi Sayıları -  Covid-19 Number of Case and Deaths in Turkey", 
              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
olum = go.Scatter(x = data.Tarih,
                    y = data.Vefat,
                    mode = "lines+markers",
                    name = "Vefat / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data.Vefat
                   )
taburcu = go.Scatter(x = data.Tarih,
                    y = data.Tedavi_Edilen,
                    mode = "lines+markers",
                    name = "Tedavi/Recovered",
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),
                    text= data.Tedavi_Edilen
                   )
data2 = [olum,taburcu]
layout = dict(title = "Türkiye'deki Covid-19 Vefat ve Tedavi Sayıları - Covid-19 Number of Deaths and Recovered in Turkey", 
              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
vaka = go.Scatter(x =mart15Sonrasi.Tarih,
                    y = mart15Sonrasi.Vaka,
                    mode = "lines+markers",
                    name = "Vaka / Cases",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= mart15Sonrasi.Vaka
                   )
layout = dict(title = "Türkiye'deki Covid-19 Vaka Sayıları (Logaritmik) -  Covid-19 Number of Case in Turkey (Logarithmic)", 
              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")
fig = dict(data = vaka, layout = layout)
iplot(fig)
Vefat = go.Scatter(x =mart17Sonrasi.Tarih,
                    y = mart17Sonrasi.Vefat,
                    mode = "lines+markers",
                    name = "Vefat / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= mart17Sonrasi.Vefat
                   )
layout = dict(title = "Türkiye'deki Covid-19 Vefat Sayıları (Logaritmik) -  Covid-19 Number of Death in Turkey (Logarithmic)", 
              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")
fig = dict(data = Vefat, layout = layout)
iplot(fig)
province_data.head()
province_data.info()
province_data.rename(columns = {"Province" : "Sehir", "Number of Case" : "Vaka Sayisi"}, inplace = True)
province_data.sort_values(by=['Vaka Sayisi'], ascending=False, inplace = True)
#province_df = province_data.head(10)
fig = px.pie(province_data.head(10), values='Vaka Sayisi', names='Sehir', title='Şehirlerdeki Vaka Sayıları')
fig.show()
province_df2 = province_data[1:]
fig = px.pie(province_df2, values='Vaka Sayisi', names='Sehir', title='İstanbul Dışındaki Şehirlerin Vaka Sayıları', 
             hover_data=['Vaka Sayisi'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = px.bar(province_data, x="Sehir", y="Vaka Sayisi", title='Şehirlerdeki Vaka Sayıları')
fig.update_layout(barmode='group')
fig.show()
fig = px.bar(province_df2, x="Sehir", y="Vaka Sayisi", title='İstanbul Dışındaki Şehirlerin Vaka Sayıları')
fig.update_layout(barmode='group')
fig.show()
fig = px.bar(province_df2.head(15), x="Sehir", y="Vaka Sayisi", title='İstanbul Dışındaki İlk 15 Şehirin Vaka Sayıları')
fig.update_layout(barmode='group')
fig.show()