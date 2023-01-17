import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot, plot
import seaborn as sns
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv',encoding='ISO-8859-1')
eksik_degerler = data.isnull().sum()
eksik_degerler_yüzde = 100*eksik_degerler/len(data)

eksik_deger_tablosu = pd.DataFrame({"Eksik Değer Sayısı" : eksik_degerler , "Eksik Değerlerin Yüzdesi" : eksik_degerler_yüzde})

eksik_deger_tablosu
data.drop("high_school" , axis = 1 , inplace = True)
data.drop("height",axis = 1 , inplace = True)
data.drop("weight",axis = 1 , inplace = True)
data.rename({
    "height_cm" : "height",
    "weight_kg" : "weight"}, axis=1 ,inplace = True)
data.shape
data.height.fillna(0,inplace = True)
boy_ortalama = round(data.height.mean(),)
data.height.replace(0,boy_ortalama,inplace = True)
data.height.sort_values(ascending = True)
kilo = data.weight.values
boy = data.height.values
boy_kilo_data = pd.DataFrame({"Weight (kg)" : kilo , "Height (cm)" : boy})

x_train = boy_kilo_data[boy_kilo_data['Weight (kg)'].notnull()].drop(columns='Weight (kg)')
y_train = boy_kilo_data[boy_kilo_data['Weight (kg)'].notnull()]['Weight (kg)']
x_test = boy_kilo_data[boy_kilo_data['Weight (kg)'].isnull()].drop(columns='Weight (kg)')
y_test = boy_kilo_data[boy_kilo_data['Weight (kg)'].isnull()]['Weight (kg)']
linear_reg = LinearRegression()
linear_reg.fit(x_train , y_train)

y_test =pd.DataFrame({"Weight (kg)" : linear_reg.predict(x_test)})

accuracy = linear_reg.score(x_test, y_test)
print(accuracy*100,'%')

data.weight.fillna(0 , inplace = True)

for i in y_test:
    data.weight.replace(0,i,inplace = True)
array = np.array([180,190,220]).reshape(-1,1)
plt.scatter(x_train.head(100),y_train.head(100))
y_head = linear_reg.predict(array)
plt.plot(array , y_head , color = "red")
plt.show()
data.birth_year.fillna(method = "pad" , inplace = True)

data.birth_year.sort_values(ascending  = True)
data.birth_month.fillna(method = "backfill" , inplace = True)

data.birth_month.sort_values(ascending  = True)
data.birth_date.fillna(method = "pad" , inplace = True)

data.birth_date.sort_values(ascending  = True)
data.Team.fillna(method = "pad" , inplace = True)

data.Team.sort_values(ascending  = True)
data.nationality.fillna(method = "backfill" , inplace = True)

data.nationality.sort_values(ascending  = True)
eksik_degerler = data.isnull().sum()
eksik_degerler_yüzde = 100*eksik_degerler/len(data)

eksik_deger_tablosu = pd.DataFrame({"Eksik Değer Sayısı" : eksik_degerler , "Eksik Değerlerin Yüzdesi" : eksik_degerler_yüzde})

eksik_deger_tablosu
kişi_sayıları = data.League.value_counts().head(15)
x = kişi_sayıları.index
y=kişi_sayıları.values

kişi_sayı = pd.DataFrame({"Ligler" : x , " Kişi Sayısı" : y})

fig = px.bar(x=x, y=y, labels={'x':'Leauge', 'y':'Number People'} ,color_discrete_sequence=["grey"])
fig.show()
ırklar = data.nationality.value_counts().head(15)
x = ırklar.index
y=ırklar.values


fig = px.pie(values=y,names = x,color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
kişiler = data[(data["League"]=="NBA") & (data["nationality"] == "Turkey")] 
kişi_isimleri = kişiler.Player.value_counts()

x = kişi_isimleri.index
y=kişi_isimleri.values

kişi_listesi = pd.DataFrame({"NBA Liginde Oynayan Türk Oyuncular" : x, "Oynadıkları Sezon Sayısı" : y})
fig = px.bar(kişi_listesi,x="NBA Liginde Oynayan Türk Oyuncular",y="Oynadıkları Sezon Sayısı")
fig.show()
en_çok_oynayanlar = data.Player.value_counts().head(20)

x = en_çok_oynayanlar.index
y = en_çok_oynayanlar.values

en_çok_listesi = pd.DataFrame({"En Fazla Sezonda Oynayan Oyuncular" : x , "Oynadıkları Sezon Sayısı" : y})

fig = px.bar(en_çok_listesi , x= "En Fazla Sezonda Oynayan Oyuncular" , y="Oynadıkları Sezon Sayısı")
fig.show()

sezonlar = data.Season.unique()
ucluk_sayilari = []

for i in sezonlar:
    x = data[data["Season"] == i]
    ucluk_sayi_bireysel = sum(x["3PM"])
    ucluk_sayilari.append(ucluk_sayi_bireysel)
    

ucluk_sayi = pd.DataFrame({"Sezon Adı" : sezonlar , "Üçlük Sayıları" : ucluk_sayilari})

fig = px.bar(ucluk_sayi , x="Sezon Adı" , y="Üçlük Sayıları",title = "Sezonlara Göre 3 Puanlık Atış Yapılma Sayıları")
fig.show()
ligler = data.League.unique()
ucluk_sayilari = []

for i in ligler:
    x = data[data["League"] == i]
    ucluk_sayi_bireysel = sum(x["3PM"])
    ucluk_sayilari.append(ucluk_sayi_bireysel)
      
ucluk_sayi = pd.DataFrame({"Lig Adı" : ligler , "Üçlük Sayıları" : ucluk_sayilari})
ucluk_index=ucluk_sayi['Üçlük Sayıları'].sort_values(ascending =False).index.values
ucluk_sorted_data = ucluk_sayi.reindex(ucluk_index).head(15)



fig = px.bar(ucluk_sorted_data , x="Lig Adı" , y="Üçlük Sayıları",title = "Liglere Göre 3 Puanlık Atış Yapılma Sayıları")
fig.show()

f,ax=plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,linewidths = 0.5,linecolor = 'white' ,fmt= '.3f' )
plt.show()
sezon = data.Season.unique()
isabet_oranı_genel = []

for i in sezon:
    x = data[data["Season"] == i]
    isabet_oranı = 100*sum(x.FGM)/sum(x.FGA)
    isabet_oranı_genel.append(isabet_oranı)
    
sezon_oranı = pd.DataFrame({"Sezon Adı" : sezon , "İsabet Oranı" : isabet_oranı_genel})
sezon_index=sezon_oranı['İsabet Oranı'].sort_values(ascending =False).index.values
sezon_sorted_data = sezon_oranı.reindex(sezon_index)


fig = px.bar(sezon_sorted_data , x = "Sezon Adı", y="İsabet Oranı" )
fig.show()
ligler = data.League.unique()
isabet_oranı_genel = []

for i in ligler:
    x = data[data["League"] == i]
    isabet_oranı = 100*sum(x["3PM"])/sum(x["3PA"])
    isabet_oranı_genel.append(isabet_oranı)
    
sezon_oranı = pd.DataFrame({"Lig Adı" : ligler , "İsabet Oranı" : isabet_oranı_genel})
sezon_index=sezon_oranı['İsabet Oranı'].sort_values(ascending =False).index.values
sezon_sorted_data = sezon_oranı.reindex(sezon_index).head(15)


fig = px.bar(sezon_sorted_data , x = "Lig Adı", y="İsabet Oranı" )
fig.show()
yil = data.birth_year.value_counts()

dogum_yili = yil.index.sort_values(ascending = True).astype(int)
kişi_sayisi = yil.values

yila_göre_dogum = pd.DataFrame({"Doğum Yılı" : dogum_yili , "Kişi Sayısı" : kişi_sayisi})

fig = px.bar(yila_göre_dogum , x = "Doğum Yılı" , y = "Kişi Sayısı")
fig.show()

