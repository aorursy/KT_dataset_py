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
# * Bu 5 dataseti ulkeleri baz alarak " Country, Region, (happiness.score = score), Health, Freedom, Corruption, Generosity" 

# sutunlari olacak sekilde tekbir df'te birlestiriniz. Diger degiskenleri atabilirsiniz. 

# (Not: Birlestirme islemleri yaparken, gerekirse ek sutunlar olusturabilirsiniz.)



# * Bazi data setlerde ulkelerin region bilgileri yok, bu ulkeleri arastirarak, region sutununda eksik yerlere ekleyiniz.

# * NaN degerleri saydiriniz.

# * NaN degerleri o degiskenin ortalamasi ile doldurunuz.

# * Data uzerinde info ve describe bilgilerini aliniz.

# * Kac cesit bolge var hesaplayiniz.

# * Mutlulugu en fazla etkileyen degisken hangisidir. Yillara gore farklilik olup olmadigina bakiniz.

# * Hangi bolgede(Region) kac ulke var bulunuz.

# * 5 yilin MUTLULUK ortalamasini alarak en mutlu ve en mutsuz 3 ulkeyi bulunuz.

# * 5 yilin YOLSUZLUK ortalamasini alarak en iyi ve en kotu ulkeleri bulunuz.

# * 5 yila bakarak, ozgurluk ortalamasi en yuksek ve en dusuk BOLGEYI/REGION bulunuz.

# * En sagliksiz bolge hangisidir bulunuz.

# * Ulkeleri bolgelerine gore gruplayarak, Mutluluk, Ozgurluk ve Yolsuzluk degiskenlerinin ortalamalini aliniz.

country_region=pd.read_excel("/kaggle/input/countries/world.xlsx")

oku_2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")

oku_2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")

oku_2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")

oku_2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")

oku_2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")



oku_2019.head(3)



data_2019=oku_2019.rename(columns={"Country or region":"Country","Score":"2019_Happiness",

          "Healthy life expectancy":"2019_Healthy Life",

          "Freedom to make life choices":"2019_Freedom",

          "Perceptions of corruption":"2019_Corruption",

         })



data_2019=data_2019[["Country","2019_Happiness","2019_Healthy Life","2019_Freedom","2019_Corruption"]]

        





data_2019.head(3)


oku_2018.head(3)




data_2018=oku_2018.rename(columns={"Country or region":"Country","Score":"2018_Happiness",

          "Healthy life expectancy":"2018_Healthy Life",

          "Freedom to make life choices":"2018_Freedom",

          "Perceptions of corruption":"2018_Corruption",

         })



data_2018=data_2018[["Country","2018_Happiness","2018_Healthy Life","2018_Freedom","2018_Corruption"]]





data_2018


oku_2017.head(3)
data_2017=oku_2017.rename(columns={ "Happiness.Score":"2017_Happiness",

                                    "Health..Life.Expectancy.":"2017_Healthy Life",

                                    "Freedom":"2017_Freedom",

                                    "Trust..Government.Corruption.":"2017_Corruption"})







data_2017=data_2017[["Country","2017_Happiness","2017_Healthy Life","2017_Freedom","2017_Corruption"]]





data_2017.head(3)


oku_2016.head(3)
data_2016=oku_2016[["Country","Happiness Score",

          "Health (Life Expectancy)",

          "Freedom",

          "Trust (Government Corruption)",

         ]]



data_2016=data_2016.rename(columns={

                                    "Happiness Score":"2016_Happiness",

                                    "Health (Life Expectancy)":"2016_Healthy Life",

                                    "Freedom":"2016_Freedom",

                                    "Trust (Government Corruption)":"2016_Corruption"})



data_2016
oku_2015.info()

oku_2015.head(3)
data_2015=oku_2015[["Country","Happiness Score",

          "Health (Life Expectancy)",

          "Freedom",

          "Trust (Government Corruption)",

         ]]



data_2015=data_2015.rename(columns={

                                    "Happiness Score":"2015_Happiness",

                                    "Health (Life Expectancy)":"2015_Healthy Life",

                                      "Freedom":"2015_Freedom",

                                    "Trust (Government Corruption)":"2015_Corruption"})



data_2015.head(3)
data_2015.info()

data_2016.info()

data_2017.info()

data_2018.info()

data_2019.info()

data=data_2015.merge(data_2016,how="left")

data=data.merge(data_2017,how="left")

data=data.merge(data_2018,how="left")

data=data.merge(data_2019,how="left")



data
data=data.merge(country_region,how="left")

data.head(3)
## * Bazi data setlerde ulkelerin region bilgileri yok, bu ulkeleri arastirarak, region sutununda eksik yerlere ekleyiniz.



data=data[["Country","Region","2015_Happiness","2016_Happiness","2017_Happiness","2018_Happiness","2019_Happiness","2015_Healthy Life","2016_Healthy Life","2017_Healthy Life","2018_Healthy Life","2019_Healthy Life","2015_Freedom","2016_Freedom","2017_Freedom","2018_Freedom","2019_Freedom","2015_Corruption","2016_Corruption","2017_Corruption","2018_Corruption","2019_Corruption"]]

data
# * NaN degerleri saydiriniz.





data.isna().sum()

# * NaN degerleri o degiskenin ortalamasi ile doldurunuz.





for i in data.columns[2:]:

    

    data[i].fillna(data[i].mean(), inplace=True)

     

data.isna().sum()
# * Data uzerinde info ve describe bilgilerini aliniz.



data.info()

data.describe().T
# * Kac cesit bolge var hesaplayiniz.



data.Region.groupby(data.Region).count()
# * Mutlulugu en fazla etkileyen degisken hangisidir. Yillara gore farklilik olup olmadigina bakiniz.





data.corr()
# * Hangi bolgede(Region) kac ulke var bulunuz.



data.groupby("Region")["Country"].count()
# * 5 yilin MUTLULUK ortalamasini alarak en mutlu ve en mutsuz 3 ulkeyi bulunuz.

data["Mean_Happiness"]=(data["2015_Happiness"]+data["2016_Happiness"]+data["2017_Happiness"]+data["2018_Happiness"]+data["2019_Happiness"])/5

print(data["Mean_Happiness"].mean())



happy=data.sort_values(by="Mean_Happiness", ascending=False)   

happy.head(3)
happy.tail(3)
# * 5 yilin YOLSUZLUK ortalamasini alarak en iyi ve en kotu ulkeleri bulunuz.





data["Mean_Corruption"]=(data["2015_Corruption"]+data["2016_Corruption"]+data["2017_Corruption"]+data["2018_Corruption"]+data["2019_Corruption"])/5

print(data["Mean_Corruption"].mean())

corrupition=data.sort_values(by="Mean_Corruption", ascending=False)   

corrupition.head(3)
corrupition.tail(3)
# * 5 yila bakarak, ozgurluk ortalamasi en yuksek ve en dusuk BOLGEYI/REGION bulunuz.



data["Mean_Freedom"]=(data["2015_Freedom"]+data["2016_Freedom"]+data["2017_Freedom"]+data["2018_Freedom"]+data["2019_Freedom"])/5

print(data["Mean_Freedom"].mean())

freedom=data.sort_values(by="Mean_Freedom", ascending=False)   

freedom.head(3)
freedom.tail(3)
data.groupby("Region")["Mean_Freedom"].mean()

# * En sagliksiz bolge hangisidir bulunuz.



data["Mean_Healthy"]=(data["2015_Healthy Life"]+data["2016_Healthy Life"]+data["2017_Healthy Life"]+data["2018_Healthy Life"]+data["2019_Healthy Life"])/5

print(data["Mean_Healthy"].mean())

Healthy=data.sort_values(by="Mean_Healthy", ascending=False)   

Healthy.head(3)
data.groupby("Region")["Mean_Healthy"].mean()
# * Ulkeleri bolgelerine gore gruplayarak, Mutluluk, Ozgurluk ve Yolsuzluk degiskenlerinin ortalamalini aliniz.



data.groupby("Region")["Mean_Happiness"].mean()
data.groupby("Region")["Mean_Freedom"].mean()
data.groupby("Region")["Mean_Corruption"].mean()

data.apply(np.max)