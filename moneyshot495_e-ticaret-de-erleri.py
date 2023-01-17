#import numpy as np

#import pandas as pd

#import matplotlib.pyplot as plt

#import csv
# ana_veri = pd.read_csv("https://evds2.tcmb.gov.tr/service/evds/datagroups/key=BURAYA KEY YAZILMALIDIR&mode=0&type=csv")

# ana_veri.drop(["CATEGORY_ID","NOTE_ENG","METADATA_LINK","REV_POL_LINK_ENG",

#          "UPPER_NOTE_ENG","APP_CHA_LINK_ENG","REV_POL_LINK","APP_CHA_LINK",

#          "METADATA_LINK_ENG","DATAGROUP_NAME_ENG","UPPER_NOTE","DATASOURCE_ENG","NOTE"], 

#           axis=1,inplace=True)
#alt_veri= pd.read_csv("https://evds2.tcmb.gov.tr/service/evds/serieList/key=BURAYA KEY'İNİZYAZILMALIDIR&type=csv&code=bie_kkhartut")

#Veri setimizide istemediğimiz kolonları drop metodu ile atıyoruz

#alt_veri.drop(["DATASOURCE_ENG","METADATA_LINK","REV_POL_LINK_ENG","APP_CHA_LINK_ENG","TAG_ENG","METADATA_LINK_ENG","DEFAULT_AGG_METHOD_STR","TAG","REV_POL_LINK","APP_CHA_LINK","DEFAULT_AGG_METHOD"],axis=1,inplace=True)





#data= pd.read_csv("https://evds2.tcmb.gov.tr/service/evds/serieList/key=KEY GİRİLECEK &type=csv&code=TP.KKHARTUT.KT50")

#data.drop(["DATASOURCE_ENG","METADATA_LINK","REV_POL_LINK_ENG","APP_CHA_LINK_ENG","TAG_ENG","METADATA_LINK_ENG","DEFAULT_AGG_METHOD_STR","TAG","REV_POL_LINK","APP_CHA_LINK","DEFAULT_AGG_METHOD"],axis=1,inplace=True)

#series =data.loc[0,"SERIE_CODE"]

#series_name=data.loc[0,"SERIE_NAME"]

#

##merkez bankası formatında yazılmalıdır.

#startDate= "01-01-%202015"

#endDate="08-05-%202020"

#typee="csv"

#key="KEY ANAHTARINZ"

#aggregationTypes="avg"

#formulas="0"

#frequency = "1"

#

#url= "https://evds2.tcmb.gov.tr/service/evds/series={}&startDate={}&endDate={}&type={}&key={}&aggregationTypes={}&formulas={}&frequency={}".format(series,startDate,endDate,typee,key,aggregationTypes,formulas,frequency)

#

#

#a=pd.read_csv(url)
#p.to_csv("E_ticaret2015-2020.csv")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



import warnings

warnings.filterwarnings





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/tr-mb-haftalk-eticaret-verileri20152020/E_ticaret2015-2020.csv")
df.head(10)
df.info()
df.shape
df.drop("Unnamed: 0",axis=1,inplace=True)

df.drop("UNIXTIME",axis=1,inplace=True)

df.drop("Tarih",axis=1,inplace=True)

df.tail()
df["tl"] = df["TP_KKHARTUT_KT50"]



df["tarih"] = df["YEARWEEK"]

df.drop("TP_KKHARTUT_KT50",axis=1,inplace=True)

df.drop("YEARWEEK",axis=1,inplace=True)
onbes = df[:52]

onalti =  df[52:105]

onyedi= df[105:157]

onsekiz = df[157:209]

ondokuz= df[209:261]

yirmi = df[261:279]
#grafiğimizin boyutu

plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=onbes.tarih,y=onbes.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,onbes.shape[0]):

    p1.text(i,onbes.tl.iloc[i]+0.03,onbes.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı",fontsize=20)

    
plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=onalti.tarih,y=onalti.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,onalti.shape[0]):

    p1.text(i,onalti.tl.iloc[i]+0.03,onalti.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı-2016",fontsize=20)

    
plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=onyedi.tarih,y=onyedi.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,onyedi.shape[0]):

    p1.text(i,onyedi.tl.iloc[i]+0.03,onyedi.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı-2017",fontsize=20)

    
plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=onsekiz.tarih,y=onsekiz.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,onsekiz.shape[0]):

    p1.text(i,onsekiz.tl.iloc[i]+0.03,onsekiz.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı-2018",fontsize=20)

    
plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=ondokuz.tarih,y=ondokuz.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,ondokuz.shape[0]):

    p1.text(i,ondokuz.tl.iloc[i]+0.03,ondokuz.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı-2019",fontsize=20)

    
plt.figure(figsize=(16,8))



#çizimi

sns.set_style("whitegrid")

p1 = sns.pointplot(x=yirmi.tarih,y=yirmi.tl,color="blue",alpha=0.5)



#grafik üzerine değerlerin yazılması



for i in range (0,yirmi.shape[0]):

    p1.text(i,yirmi.tl.iloc[i]+0.03,yirmi.tl.iloc[i],size="medium",color="red",weight="semibold")

    

#x eksenine yazıları 90 derece açıyla yazma 

plt.xticks(rotation=90)



#x ve y eksenimizi isimlendirdik 



plt.xlabel("Tarih",fontsize=15)

plt.ylabel("Tl",fontsize=15)

plt.title("E-Ticaret Oranı-2020",fontsize=20)

    