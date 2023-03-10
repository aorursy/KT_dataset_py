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
df.isna().sum() # concat yap??ld??ktan sonra region s??tununda 467 nan de??er var. ??nce bunu halletmemiz laz??m.
df1516 = pd.concat([df15,df16])

regions = list(df1516["Region"])

countries = list(df1516["Country"])



# 2015 ve 2016 tablolar??nda ??lkelerin b??lgeleri yer ald?????? i??in bu iki tabloyu birle??itrdik ve 

# olu??an tabloda yer alan b??t??n ulkeleri ve bolgeleri iki ayr?? de??i??kene atad??k.



co_re=[[str(regions[i]),str(countries[i])] for i in range(len(regions))]



# Yukar??daki iki listeden faydalanarak (??lke, b??lge) ??eklinde iki elemanl?? alt k??meler olu??turduk, ??lkeleri b??lgeleri ile e??le??tirmi?? olduk.



for i in range(len(df)):   

    if str(df.loc[i,"Region"]) == str(np.nan): 

        country = str(df.loc[i,"Country"])



# concat etti??imiz (5 tablodan olu??an) df tablosunun sat??r say??s?? kadar d??ng?? kurduk.         

# e??er gezilen sat??r??n b??lge s??tunu nan ise; (not: str'ye ??evirmeden ba??ar??l?? olamad??m.)   

# o sat??rda yer alan ??lke ad??n?? bir de??i??kene atad??k.        

        

        for j in co_re:

            if j[1] == country:

                region = j[0]               

                df.loc[i,"Region"] = region

        

# daha sonra ulke ve b??lge adlar??ndan olu??trdu??umuz liste i??in d??ng?? kurduk.

# yukar??da elde etti??imiz ??lke de??i??keni hangi b??lge ile e??le??iyorsa, ??rne??in; (Switzerland, Western Europe) gibi,

# e??le??en de??eri, df tablosunda b??lge ismi olarak nan de??ere atad??k.



df



#bu i??lemi "merge" ile denedim ancak ba??ar??l?? olamad??m. bu konuda yard??m??n??z?? bekliyorum.
df["Region"].isna().sum() # Region s??tununda ka?? tane nan de??eri oldu??unu verir.
zoneless = pd.isnull(df["Region"]) 

df[zoneless]

# Region s??tunundaki nan de??erlerini tablo olarak g??steriyoruz. 
empty_corruption = pd.isnull(df["Corruption"])

df[empty_corruption]
df.isna().sum() # t??m s??tunlardaki toplam nan de??erlerini sayd??r??yoruz.
df["Region"].fillna("Unspecified", inplace=True) 

# region s??tunundaki nan de??erlere yeni bir isim girdik.

# bu ??lkeler ara??t??r??l??p hangi b??lgede ise manuel olarak tek tek yaz??labilir ama yeti??tiremedim. ba??ka yolu var m?? bilmiyorum.
df["Corruption"].fillna(df["Corruption"].mean(), inplace=True) 

# Corruption s??tunundaki nan de??ere bu s??tunun ortalamas??n?? girdik.
df.isna().sum() 

# tekrar sorgulad??????m??zda tabloda nan de??er kalmad??????n?? teyit ediyoruz.
df_pv = df.pivot_table(index=["Region","Country","Year"])

df_pv
df_pv.loc["Western Europe"] 

# pivot table ??n istedi??imiz indexlerine ula??abilme.

# bu tablo bat?? avrupadaki ??lkeleri ve y??llara g??re de??i??kenlerini g??steriyor.
df_pv.loc["Western Europe","Netherlands"] 

# bu tablo ile daha fazla ??zelle??tirme yap??p sadece Hollanda'n??n y??llara g??re de??i??kenlerini g??rebiliyoruz.
df.groupby("Region").mean() 

# toplam ka?? b??lge var bu ??ekkilde g??rebiliriz. ayn?? zamanda b??lgelerin ortalamalar??n?? g??r??yoruz.
len(df_pv.groupby("Region"))-1    # B??lge say??s??n?? buluyoruz.
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



# bu iki y??ntem de tabloyu ??lkelere g??re grupluyor ve ortalamalar??n?? veriyor.
df.groupby("Country").mean()
df.info()
df.describe().T
df["Health"].corr(df["Score"]), df["Freedom"].corr(df["Score"]),df["Corruption"].corr(df["Score"]), df["Generosity"].corr(df["Score"])
first_three = df.sort_values(by="Score").tail(3)

last_three = df.sort_values(by="Score").head(3)



middle = df.sort_values(by="Score")[int(len(df)/2)-1 : int(len(df)/2)+2]



# tablonun ortas??ndan ???? tane eleman se??menin ????k bir yolu var m???

# middle i??in ba??ka y??ntem: df[(df["Score"] <= df["Score"].mean()+0.01) & (df["Score"] >= df["Score"].mean()-0.01)]



pd.concat([first_three, middle, last_three]).sort_values(by="Score", ascending=False)
df["Score"].mean() # t??m ??lkeler 5 y??l??n mutluluk ortalamas??
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Score").head(3) 



# 5 y??l??n ortalamas??nda en mutsuz ???? ??lke.

# ayr??ca df.iloc[:,:-1] diyerek son s??tunda yer alan "year" de??i??kenini g??stermeyebiliriz.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Score").tail(3) # 5 y??l??n ortalamas??nda en mutlu ???? ??lke.
df.iloc[:,:-1].groupby("Country")["Score"].mean().sort_values().tail(3) 



# yukar??daki ile ayn?? sorguyu bu ??ekilde yaparsam tablo olarak vermiyor. Fark?? g??rmek i??in koydum.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Health").head(3) # 5 y??l??n ortalamas??nda en sa??l??ks??z ???? ??lke.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Health").tail(3) # 5 y??l??n ortalamas??nda en sa??l??kl?? ???? ??lke.
df.iloc[:,:-1].groupby("Country").mean().sort_values(by="Corruption").head(3) # 5 y??l??n ortalamas??nda yolsuzlukta en k??t?? ???? ??lke
df.groupby("Region")["Freedom"].mean() # 5 y??l??n b??lgelere g??re ??zg??rl??k ortalamalar??.
df.iloc[:,:-1].groupby("Region").mean().sort_values(by="Freedom").tail(1) # en ??zg??r b??lge.
df.iloc[:,:-1].groupby("Region").mean().sort_values(by="Freedom").head(1) # en ??zg??r olmayan b??lge.