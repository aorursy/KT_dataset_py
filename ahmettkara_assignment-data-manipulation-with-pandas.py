# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

df2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

df2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

df2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

df2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
df15 = df2015.copy()

df16 = df2016.copy()

df17 = df2017.copy()

df18 = df2018.copy()

df19 = df2019.copy()
df15.head()
df16.head()
df17.head()
df18.head()
df19.head()
df15.shape, df16.shape, df17.shape, df18.shape, df19.shape

df15.columns, df16.columns, df17.columns, df18.columns, df19.columns
df15.drop(['Happiness Rank','Standard Error', 'Economy (GDP per Capita)', 

           'Family', 'Dystopia Residual'],axis=1,inplace=True)



df15.rename(columns={"Happiness Score":"Score",

                     "Health (Life Expectancy)":"Health",

                     "Trust (Government Corruption)":"Corruption"}, inplace=True)

df15["Year"]=2015

#df15
df16.drop(['Happiness Rank','Lower Confidence Interval', 'Upper Confidence Interval', 'Economy (GDP per Capita)', 

           'Family', 'Dystopia Residual'],axis=1,inplace=True)



df16.rename(columns={"Happiness Score":"Score",

                     "Health (Life Expectancy)":"Health",

                     "Trust (Government Corruption)":"Corruption"}, inplace=True)

df16["Year"]=2016

#df16
df17.drop(['Happiness.Rank','Whisker.high', "Whisker.low", 'Economy..GDP.per.Capita.', 'Family', 'Dystopia.Residual'],axis=1,inplace=True)



df17.rename(columns={"Happiness.Score":"Score",

                     "Health..Life.Expectancy.":"Health",

                     "Trust..Government.Corruption.":"Corruption"}, inplace=True)

df17["Year"]=2017

#df17
df18.drop(["Overall rank","GDP per capita","Social support"],axis=1,inplace=True)



df18.rename(columns={"Country or region":"Country",

                     "Healthy life expectancy":"Health",

                     "Freedom to make life choices":"Freedom",

                     "Perceptions of corruption":"Corruption"}, inplace=True)

df18["Year"]=2018

#df18
df19.drop(["Overall rank","GDP per capita","Social support"],axis=1,inplace=True)



df19.rename(columns={"Country or region":"Country",

                     "Healthy life expectancy":"Health",

                     "Freedom to make life choices":"Freedom",

                     "Perceptions of corruption":"Corruption"}, inplace=True)

df19["Year"]=2019

df19
df = pd.concat([df15,df16,df17,df18, df19],ignore_index=True, sort=False)

df
df.isna().sum() # concat yapıldıktan sonra region sütununda 467 nan değer var. önce bunu halletmemiz lazım.
df1516 = pd.concat([df15,df16])

regions = list(df1516["Region"])

countries = list(df1516["Country"])



# 2015 ve 2016 tablolarında ülkelerin bölgeleri yer aldığı için bu iki tabloyu birleşitrdik ve 

# oluşan tabloda yer alan bütün ulkeleri ve bolgeleri iki ayrı değişkene atadık.



co_re=[[str(regions[i]),str(countries[i])] for i in range(len(regions))]



# Yukarıdaki iki listeden faydalanarak (ülke, bölge) şeklinde iki elemanlı alt kümeler oluşturduk, ülkeleri bölgeleri ile eşleştirmiş olduk.



for i in range(len(df)):   

    if str(df.loc[i,"Region"]) == str(np.nan): 

        country = str(df.loc[i,"Country"])



# concat ettiğimiz (5 tablodan oluşan) df tablosunun satır sayısı kadar döngü kurduk.         

# eğer gezilen satırın bölge sütunu nan ise; (not: str'ye çevirmeden başarılı olamadım.)   

# o satırda yer alan ülke adını bir değişkene atadık.        

        

        for j in co_re:

            if j[1] == country:

                region = j[0]               

                df.loc[i,"Region"] = region

        

# daha sonra ulke ve bölge adlarından oluştrduğumuz liste için döngü kurduk.

# yukarıda elde ettiğimiz ülke değişkeni hangi bölge ile eşleşiyorsa, örneğin; (Switzerland, Western Europe) gibi,

# eşleşen değeri, df tablosunda bölge ismi olarak nan değere atadık.



df



#bu işlemi "merge" ile denedim ancak başarılı olamadım. bu konuda yardımınızı bekliyorum.
df["Region"].isna().sum() # Region sütununda kaç tane nan değeri olduğunu verir.
zoneless = pd.isnull(df["Region"]) 

df[zoneless]

# Region sütunundaki nan değerlerini tablo olarak gösteriyoruz. 
empty_corruption = pd.isnull(df["Corruption"])

df[empty_corruption]
df.isna().sum() # tüm sütunlardaki toplam nan değerlerini saydırıyoruz.
df["Region"].fillna("Unspecified", inplace=True) 

# region sütunundaki nan değerlere yeni bir isim girdik.

# bu ülkeler araştırılıp hangi bölgede ise manuel olarak tek tek yazılabilir ama yetiştiremedim. başka yolu var mı bilmiyorum.
df["Corruption"].fillna(df["Corruption"].mean(), inplace=True) 

# Corruption sütunundaki nan değere bu sütunun ortalamasını girdik.
df.isna().sum() 

# tekrar sorguladığımızda tabloda nan değer kalmadığını teyit ediyoruz.
df_pv = df.pivot_table(index=["Region","Country","Year"])

df_pv
df_pv.loc["Western Europe"] 

# pivot table ın istediğimiz indexlerine ulaşabilme.

# bu tablo batı avrupadaki ülkeleri ve yıllara göre değişkenlerini gösteriyor.
df_pv.loc["Western Europe","Netherlands"] 

# bu tablo ile daha fazla özelleştirme yapıp sadece Hollanda'nın yıllara göre değişkenlerini görebiliyoruz.
df.groupby("Region").mean() 

# toplam kaç bölge var bu şekkilde görebiliriz. aynı zamanda bölgelerin ortalamalarını görüyoruz.
len(df_pv.groupby("Region"))-1    # Bölge sayısını buluyoruz.
print(f"""

Australia and New Zealand: {len(df[df["Region"]=="Australia and New Zealand"].groupby("Country"))} 

Central and Eastern Europe: {len(df[df["Region"]=="Central and Eastern Europe"].groupby("Country"))}

Eastern Asia: {len(df[df["Region"]=="Eastern Asia"].groupby("Country"))}

Latin America and Caribbean: {len(df[df["Region"]=="Latin America and Caribbean"].groupby("Country"))}

Middle East and Northern Africa: {len(df[df["Region"]=="Middle East and Northern Africa"].groupby("Country"))}

North America: {len(df[df["Region"]=="North America"].groupby("Country"))}

Southeastern Asia: {len(df[df["Region"]=="Southeastern Asia"].groupby("Country"))}

Southern Asia: {len(df[df["Region"]=="Southern Asia"].groupby("Country"))}

Sub-Saharan Africa: {len(df[df["Region"]=="Sub-Saharan Africa"].groupby("Country"))}

Western Europe: {len(df[df["Region"]=="Western Europe"].groupby("Country"))}

""")
# df.pivot_table(index=["Country"]) 



# df.groupby("Country").mean()



# bu iki yöntem de tabloyu ülkelere göre grupluyor ve ortalamalarını veriyor.
df.groupby("Country").mean()
df.info()
df.describe().T
df["Health"].corr(df["Score"]), df["Freedom"].corr(df["Score"]),df["Corruption"].corr(df["Score"]), df["Generosity"].corr(df["Score"])
first_three = df.sort_values(by="Score").tail(3)

last_three = df.sort_values(by="Score").head(3)



middle = df.sort_values(by="Score")[int(len(df)/2)-1 : int(len(df)/2)+2]



# tablonun ortasından üç tane eleman seçmenin şık bir yolu var mı?

# middle için başka yöntem: df[(df["Score"] <= df["Score"].mean()+0.01) & (df["Score"] >= df["Score"].mean()-0.01)]



pd.concat([first_three, middle, last_three]).sort_values(by="Score", ascending=False)
df["Score"].mean() # tüm ülkeler 5 yılın mutluluk ortalaması
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Score").head(3) 



# 5 yılın ortalamasında en mutsuz üç ülke.

# ayrıca df.iloc[:,:-1] diyerek son sütunda yer alan "year" değişkenini göstermeyebiliriz.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Score").tail(3) # 5 yılın ortalamasında en mutlu üç ülke.
df.iloc[:,:-1].groupby("Country")["Score"].mean().sort_values().tail(3) 



# yukarıdaki ile aynı sorguyu bu şekilde yaparsam tablo olarak vermiyor. Farkı görmek için koydum.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Health").head(3) # 5 yılın ortalamasında en sağlıksız üç ülke.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Health").tail(3) # 5 yılın ortalamasında en sağlıklı üç ülke.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Corruption").head(3) # 5 yılın ortalamasında yolsuzlukta en kötü üç ülke
df.groupby("Region")["Freedom"].mean() # 5 yılın bölgelere göre özgürlük ortalamaları.
df.iloc[:,:-1].groupby("Region").mean().sort_values(by="Freedom").tail(1) # en özgür bölge.
df.iloc[:,:-1].groupby("Region").mean().sort_values(by="Freedom").head(1) # en özgür olmayan bölge.