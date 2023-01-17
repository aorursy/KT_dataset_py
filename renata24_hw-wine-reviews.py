from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://img1.goodfon.com/wallpaper/big/8/b7/vinograd-korzina-vetki-listya.jpg")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pycountry

import matplotlib.pyplot as plt

import matplotlib.colors



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



#Andmete sisselugemine:

df = pd.read_csv("../input/winemag-data_first150k.csv", encoding="utf-8")

df
#Vaatame andmebaasi kirjeldavat statistikat:

df.info()
df2=pd.DataFrame(df.groupby("country")["points"].mean())

df2.sort_values("points", ascending=False)
#Punktiskoori veergude lisamine:

df2["max_points"]=pd.DataFrame(df.groupby("country")["points"].max())

df2["min_points"]=pd.DataFrame(df.groupby("country")["points"].min())

df2["max-min_points"]=df2["max_points"]-df2["min_points"] #lahutame ühe Series objekti teisest

df2["stdev_points"]=pd.DataFrame(df.groupby("country")["points"].std())

df2["hinnanguid"]=df["country"].value_counts()



df2.sort_values("points", ascending=False)
#Filtreerime:

df3=df2[df2.hinnanguid>=20]



#Riikide ISO-koodide lisamine: enamiku ISO-koode saab kätte pythoni funktsiooniga, erandiks on 

#kaks riiki, mille puhul määrame koodi käsitsi:

def lisa_ISO(dfx):

    koodid=[]

    for riik in list(dfx.index.values): #veerunimede (riikide) list

        if riik=='Moldova':

            koodid.append('MD')

        elif riik=='US':

            koodid.append('US') 

        else:

            koodid.append(pycountry.countries.get(name=riik).alpha_2) #tuvastame ISO-koodi

    dfx["ISO_kood"]=koodid # lisame uue veeru tabelisse



lisa_ISO(df3)

print("Riike kokku:", len(df3)) #palju riike nüüd veel tabelisse on jäänud



df3.sort_values("points", ascending=False)
#TULPDIAGRAMM:

def joonista(dfx, veerg, ytelg_title, pealkiri, y_max, y_min):

    dfx=dfx.sort_values(veerg, ascending=True)

    labels = dfx['ISO_kood'] #tulpade nimed

    data   = dfx[veerg] #andmed

    y_pos = range(len(labels)) #tulpade järjekord (mis järjekorras kuvatakse)



    plt.figure(figsize=(20,10))

    plt.bar(y_pos, data, color=matplotlib.colors.to_hex("#3399ff", keep_alpha=False), width=0.6)

    #värvi valimiseks saab kasutada värvide hex-koode: https://www.w3schools.com/colors/colors_picker.asp

    plt.xticks(y_pos, labels, fontsize=20, fontweight='bold') # tulpade nimed  #rotation=90, 

    plt.yticks(fontsize=20, fontweight='bold')

    plt.ylim(ymax = y_max, ymin = y_min) #y-telje väärtuste vahemik



    # joonise ja telgede nimed:

    plt.title(pealkiri, size=30, fontweight='bold')

    plt.xlabel('\nRIIGID', size=24, fontweight='bold')

    plt.ylabel(ytelg_title+"\n", size=24, fontweight='bold')



    # kuvame joonise:

    plt.show()



joonista(df3, "points", "HINNANGUD", "VEINIDELE ANTUD HINNANGUD", 90, 80)
df4=pd.DataFrame(df.groupby("country")["price"].mean())

#df4.sort_values("price", ascending=False)



#Veergude lisamine:

df4["max_price"]=pd.DataFrame(df.groupby("country")["price"].max())

df4["min_price"]=pd.DataFrame(df.groupby("country")["price"].min())

df4["max-min_price"]=df4["max_price"]-df4["min_price"] #lahutame ühe Series objekti teisest

df4["stdev_price"]=pd.DataFrame(df.groupby("country")["price"].std())

#df4["var_price"]=pd.DataFrame(df.groupby("country")["price"].var())

df4["hinnanguid"]=df["country"].value_counts()



##Filtreerime:

df5=df4[df4.hinnanguid>=20]



lisa_ISO(df5) #lisame ISO-koodide veeru



print("Riike kokku:", len(df5)) #palju riike nüüd veel tabelisse on jäänud

df5=df5.sort_values("price", ascending=False) 

df5
joonista(df5, "price", "HINNAD", "VEINIDE KESKMINE HIND RIIKIDE LÕIKES", 50, 10)
max_price=df["price"].max()



df6=df.loc[[indeks for indeks in range(len(df)) if df["price"][indeks]==max_price]] 

print('KALLEIMA VEINI KIRJELDUS:', df6["description"].tolist())

df6
#ODAVAIMAD VEINID:

#min_price=df["price"].min()

#df.loc[[indeks for indeks in range(len(df)) if df["price"][indeks]==min_price]] 
#Kontrollime, kas hindade ja punktide tabeli reaindeksid kattuvad või on erinevusi:

#sorted(list(df3.index.values))==sorted(list(df6.index.values))  #True -> kattuvad



#Sorteerime dataframe'id, et nende indeksid oleksid samas järjekorras:

df3=df3.sort_index(ascending=True)

df5=df5.sort_index(ascending=True)



#Teeme uue dataframe'i kahe olemasoleva kolmest veerust:

data=[df3["points"], df5["price"], df3["ISO_kood"]]

a = pd.concat(data, axis=1, keys=[s.name for s in data])



#Lisame hinna ja kvaliteedi suhte veeru:

a["hinna-kvaliteedi suhe"]=round(a["points"]/a["price"], 2) 



a=a.sort_values("hinna-kvaliteedi suhe", ascending=False) 

a
joonista(a, "hinna-kvaliteedi suhe", "HINNA JA KVALITEEDI SUHE", "VEINIDE KESKMINE HINNA JA KVALITEEDI SUHE RIIKIDE LÕIKES", 9, 0)
#SCATTERPLOT:



hinnad=a['price'].values.tolist()

punktid=a['points'].values.tolist()

n=list(a.index.values)



#plt.figure(figsize=(20,10))



fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(punktid, hinnad, color=matplotlib.colors.to_hex("#ff66ff", keep_alpha=False), s=400)



plt.yticks(fontsize=20, fontweight='bold')

plt.xticks(fontsize=20, fontweight='bold')

#plt.ylim(ymax = y_max, ymin = y_min) #y-telje väärtuste vahemik







# joonise ja telgede nimed:

plt.title("HINNA JA KVALITEEDI SUHE\n", size=30, fontweight='bold')

plt.xlabel('\nKVALITEET', size=24, fontweight='bold')

plt.ylabel("HIND\n", size=24, fontweight='bold')



for i, txt in enumerate(n):

    ax.annotate(txt, (punktid[i],hinnad[i]), fontsize=20, fontweight='bold')


