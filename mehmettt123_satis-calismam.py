# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def satislar_load():
    dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
    return dff
df = satislar_load()

df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
#df['CategoryName'] = df['CategoryName'].apply(lambda x: x.strip(","))

#df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 
#TUM SUBELERDEN EN FAZLA ALINAN URUNLER (URUNLER VE ALINMA MİKTARLARI)
df_yedek = df.copy()
df_yedek = df_yedek.CategoryName.str.strip(",").value_counts()[0:40]
df_yedek.head()
#SUBEDEN EN FAZLA ALINAN URUNLER (URUNLER VE ALINMA MİKTARLARI)
df_en = df.copy()
df_en = df.loc[df.BranchId == 1146]
df_en.CategoryName.str.strip(",").value_counts()[0:5]
#SUBELERE GORE EN FAZLA SATILAN URUNLER
a = [1129, 1141, 1109, 1128, 1143, 1148, 1124, 1127, 1130, 1131, 1137,
       1102, 1104, 1110, 1111, 1115, 1118, 1120, 1136, 4010, 1103, 1122,
       1126, 1140, 1145, 1147, 1160, 1106, 1108, 1113, 1119, 1125, 1135,
       1139, 1142, 1146, 4003, 4004, 4006, 4007, 1101, 1112, 1114, 1116,
       1132, 1133, 1138, 1144, 1107, 1117, 1121, 1123, 1134, 4009, 4005]
for i in a:
    print(i , "*"*100)
    print(df.loc[df.BranchId == i].CategoryName.str.strip(",").value_counts()[0:5])
  
# TANIMSIZ URUNUNE BAKIS . TANIMSIZ URUNLERİN OGLEDEN SONRA DAHA FAZLA SATILDIGI GORULMEKTEDİR.
df_tanimsiz = df.copy()
df_tanimsiz = df.loc[df.BranchId == 1146]
df_tanimsiz['CategoryName'] = df_tanimsiz['CategoryName'].apply(lambda x: x.strip(","))
df_tanimsiz["InvoiceDate"] = df_tanimsiz["InvoiceDate"].astype('datetime64[ns]')
df_tanimsiz["year"] =df_tanimsiz["InvoiceDate"].dt.year
df_tanimsiz["month"] =df_tanimsiz["InvoiceDate"].dt.month
df_tanimsiz["day"] =df_tanimsiz["InvoiceDate"].dt.day
df_tanimsiz["hour"] =df_tanimsiz["InvoiceDate"].dt.hour
df_tanimsiz["minute"] =df_tanimsiz["InvoiceDate"].dt.minute
df_tanimsiz = df_tanimsiz.loc[df_tanimsiz.CategoryName == "TANIMSIZ"]

sns.countplot(x="hour", hue="day", data=df_tanimsiz);
# en fazla hangi subelerden alısberis yapılmıs. Toplamda 55 sube var.
df.groupby("BranchId")["InvoiceNo"].count().sort_values(ascending= False)
# 1146 subesinde daha cok siparis oldugu icin ilk ona bakıyorum. 48146 adet geçmiş.
df_en = df.loc[df.BranchId == 1146]

df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
df_en["year"] =df_en["InvoiceDate"].dt.year
df_en["month"] =df_en["InvoiceDate"].dt.month
df_en["day"] =df_en["InvoiceDate"].dt.day
df_en["hour"] =df_en["InvoiceDate"].dt.hour
df_en["minute"] =df_en["InvoiceDate"].dt.minute


df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
df_en.head()
df_yedek = df_en.copy()

df_yedek = df_yedek.drop(["InvoiceDate","year"] , axis = 1)
df_yedek = df_yedek.drop(["month"] , axis = 1)
df_yedek.loc[df_yedek.day_name == "Monday"].head()
#Secilen subedeki urunlerin gecme miktarı.
df_yedek.CategoryName.value_counts().sort_values(ascending=False)[0:5]
df_urunler = df_yedek.CategoryName.unique()
# isci bayramında yapılan harcamaların saat bazında degerlendirilmesi
df_isci = df_yedek.loc[df_yedek.day == 1]
for a in df_urunler:
    for i in ["hour"]:
        sns.countplot(df_isci.loc[df_isci.CategoryName == a, "CategoryName"], hue=df_yedek[i])
        plt.show()
#tum zamanlara gore urunlerin satısı
for a in df_urunler:
    for i in ["day" , "hour" ,"day_name"]:
        sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
        plt.show()
# Birliktelik analizi
df_yedek.head()
df_genel = df_yedek.copy()
df_yedek = df_yedek.CategoryName.str.strip(",")
dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
dff.head()
dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()
dff = dff.drop("InvoiceNo" , axis = 1)
dff.head()
data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
data[0:5]
from mlxtend.preprocessing import TransactionEncoder
tencoder = TransactionEncoder()
te_data = tencoder.fit(data).transform(data)
df = pd.DataFrame(te_data, columns=tencoder.columns_)
df.head()
from mlxtend.frequent_patterns import apriori,  association_rules
df1 = apriori(df, min_support=0.01, use_colnames=True)
df1.head()
df_association = association_rules(df1, metric = 'confidence', min_threshold=0.5)
df_association.sort_values(by='confidence', ascending=False).reset_index()[0:5]
# BİR SUBENİN TUM SEPETLERİNE YAPILAN BİRLİKTELİK ANALİZİDİR.

def normal(sube_no , support , threshold):
    """
    Secilecek olan bir subenin (sube_no) birliktelik analizlerini ve urunlerin zamanlara gore grafıgını verir.
    sube_no = Magazanın sube numarası
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
    #df['CategoryName'] = df['CategoryName'].apply(lambda x: x.strip(","))

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.loc[df.BranchId == sube_no] # onemli

    ## ucuncu. adım
    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
  

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)

    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    for a in df_urunler:
        for i in ["day" , "hour" ,"day_name"]:
            sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
            plt.show()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#normal(1146, 0.01 , 0.5)

# GUN OLARAK DEGERLENDİRME
def day(sube_no , support , threshold , day , plot = False ):
    """
    Secilecek olan bir subenin (sube_no) birliktelik analizlerini ve urunlerin gunlere gore grafıgını verir.
    sube_no = Magazanın sube numarası
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.
    day = Secilecek gun bazında birliktelik analizi uygular.
    plot = "True" yapıldıgında urunlerın gun bazlı grafiklerini verir.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 

    df_en = df.loc[df.BranchId == sube_no] # onemli

    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute

    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek = df_yedek.loc[df_yedek.day_name == day]
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    
    if plot:
        for a in df_urunler:
            for i in ["hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()

    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(day)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#day(1147 , 0.01 , 0.51 , "Monday" )

# GUNUN SAATLERİ OLARAK DEGERLENDİRME
def hour(sube_no , support , threshold , hour , plot = False):
    """
    Secilecek olan bir subenin (sube_no) birliktelik analizlerini ve urunlerin gunlere gore grafıgını verir.
    sube_no = Magazanın sube numarası
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.
    hour = Secilecek saat bazında birliktelik analizi uygular.
    plot = "True" yapıldıgında urunlerın saat bazlı grafiklerini verir.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']


    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.loc[df.BranchId == sube_no] # onemli


    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek = df_yedek.loc[df_yedek.hour == hour]
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    
    if plot:
        for a in df_urunler:
            for i in ["day" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()

    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(hour)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()

#hour(1146 , 0.01 , 0.51 , 18 )

# VAKİTLERE GORE ANALİZ.
def day_time(sube_no , support , threshold , vakit , plot = False):
    """
    Secilecek olan bir subenin (sube_no) birliktelik analizlerini ve urunlerin gunlere gore grafıgını verir.
    sube_no = Magazanın sube numarası
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.
    vakit  = Secilecek vakit(sabah ,ogle , aksam) bazında birliktelik analizi uygular.
    plot = "True" yapıldıgında urunlerın vakit(sabah ,ogle , aksam) bazlı grafiklerini verir.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']


    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.loc[df.BranchId == sube_no] # onemli


    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute



    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en


    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    vakitler = ['sabah', 'ogle', 'aksam']
    cut_bins = [8, 12, 16, 21]
    df_yedek['vakitler'] = pd.cut(df_yedek['hour'], bins=cut_bins, labels=vakitler)
    df_yedek = df_yedek.loc[df_yedek.vakitler == vakit]
    df_urunler = df_yedek.CategoryName.unique()


    if plot:
        for a in df_urunler:
            for i in ["day" , "hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()    


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(vakit , "*"*100)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#day_time(1146 , 0.01 , 0.51 , "aksam" )


#HAFTASONU VE HAFTAICI ANALİZİ
def ic_son(sube_no , support , threshold , ic_son , plot= False):
    """
    Secilecek olan bir subenin (sube_no) birliktelik analizlerini ve urunlerin gunlere gore grafıgını verir.
    sube_no = Magazanın sube numarası
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.
    ic_son  = Secilecek ic_son(haftaici , haftasonu) bazında birliktelik analizi uygular.
    plot = "True" yapıldıgında urunlerın vakit(sabah ,ogle , aksam) bazlı grafiklerini verir.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']


    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.loc[df.BranchId == sube_no] # onemli


    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute



    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en




    df_yedek = df_en.copy()


    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek["ic_and_son"] = np.NaN
    df_yedek.loc[(df_yedek.day_name == "Saturday") | (df_yedek.day_name == "Sunday") , "ic_and_son"] = "haftasonu"
    df_yedek.loc[~(df_yedek.day_name == "Saturday") | (df_yedek.day_name == "Sunday") , "ic_and_son"] = "haftaici"
    
    df_yedek = df_yedek.loc[df_yedek.ic_and_son == ic_son]
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    
    if plot:
        for a in df_urunler:
            for i in ["day" , "hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(ic_son , "*"*100)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#ic_son(1146 , 0.02 , 0.51 , "haftasonu")


#TUM SUBELERE YAPILAN TOPLU BİRLİKTELİK ANALİZİ

# TUM SUBELERİN SEPETLERİNE YAPILAN BİRLİKTELİK ANALİZİDİR.

def tum_normal( support , threshold):
    """
    Tum subelerin birliktelik analizlerini ve urunlerin zamanlara gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri.   
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']


    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy() # onemli


    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en


    df_yedek = df_en.copy()


    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)


    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    for a in df_urunler:
        for i in ["day" , "hour" ,"day_name"]:
            sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
            plt.show()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#tum_normal( 0.01 , 0.51)

# TUM GUN OLARAK DEGERLENDİRME
def tum_day(support , threshold , day , plot = False ):
    """
    Tum subelerin birliktelik analizlerini ve urunlerin gunlere gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    day = Gunlere("Monday"...) gore birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']


    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 

    df_en = df.copy()

    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en


    df_yedek = df_en.copy()


    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek = df_yedek.loc[df_yedek.day_name == day]
    df_urunler = df_yedek.CategoryName.unique()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(day)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()

#tum_day(0.01 , 0.51 , "Monday" , plot = False )

# GUNUN SAATLERİ OLARAK DEGERLENDİRME
def tum_hour( support , threshold , hour , plot = False):
    """
    Tum subelerin birliktelik analizlerini ve urunlerin hour gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    hour = Saatlere(8,9,10...) gore birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy()


    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en


    df_yedek = df_en.copy()


    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek = df_yedek.loc[df_yedek.hour == hour]
    df_urunler = df_yedek.CategoryName.unique()


    
    if plot:
        for a in df_urunler:
            for i in ["day" , "hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    
    return df_association.sort_values(by='confidence', ascending=False).reset_index()

#tum_hour(0.01 , 0.51 , 18 )

# TUM VAKİTLERE GORE ANALİZ.
def tum_day_time(support , threshold , vakit , plot = False):
    """
    Tum subelerin birliktelik analizlerini ve urunlerin vakitlere gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    vakit = Vakitlere("sabah","ogle","aksam") gore birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy()

    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute
    

    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    vakitler = ['sabah', 'ogle', 'aksam']
    cut_bins = [8, 12, 16, 21]
    df_yedek['vakitler'] = pd.cut(df_yedek['hour'], bins=cut_bins, labels=vakitler)
    df_yedek = df_yedek.loc[df_yedek.vakitler == vakit]
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    
    if plot:
        for a in df_urunler:
            for i in ["day" , "hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()    

                
    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(vakit , "*"*100)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#tum_day_time(0.01 , 0.51 , "sabah" )

#HAFTASONU VE HAFTAICI ANALİZİ
def tum_ic_son(support , threshold , ic_son , plot= False):
    """
    Tum subelerin birliktelik analizlerini ve urunlerin haftasonu-haftaicine gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    ic_son = Haftasonu ve haftaicine gore birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy() # onemli

    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute

    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    df_yedek["ic_and_son"] = np.NaN
    df_yedek.loc[(df_yedek.day_name == "Saturday") | (df_yedek.day_name == "Sunday") , "ic_and_son"] = "haftasonu"
    df_yedek.loc[~(df_yedek.day_name == "Saturday") | (df_yedek.day_name == "Sunday") , "ic_and_son"] = "haftaici"
    
    df_yedek = df_yedek.loc[df_yedek.ic_and_son == ic_son]
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara gore urunlerin satısı
    
    if plot:
        for a in df_urunler:
            for i in ["day" , "hour" ,"day_name"]:
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    print(ic_son , "*"*100)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#tum_ic_son(0.01 , 0.51 , "haftaici")

# Tum subelere yapılan secmeli islemlerdir.

def tum_secmeli( support , threshold , days_name , hours):
    """
    Tum subelerin birliktelik analizlerini days_name("Friday") ve hours(8,9,10 ...) gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    days_name = Verilen gune gore birliktelik analizi saglar.
    hour = Saatlere(8,9,10...) gore birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy() # onemli

    ## ucuncu. adım
    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.loc[(df_en.day_name == days_name) & (df_en.hour == hours)]

  
    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)

    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#tum_secmeli( 0.01 , 0.60 , "Friday" , 16)

# Bir subeye yapılan secmeli islemdir.

def sube_secmeli(sube_no, support , threshold , days_name , hours ):
    """
    Girilen sube numarasına birliktelik analizlerini days_name("Friday") ve hours(8,9,10 ...) gore grafıgını verir.
    support = Birliktelik analizinde kullanılan "min_support" degeri
    threshold = Birliktelik analizinde kullanılan "association_rules" fonksiyonun degeri. 
    days_name = Verilen gune gore girilen subenin birliktelik analizi saglar.
    hour = Saatlere(8,9,10...) gore girilen subenin birliktelik analizi yapmamızı saglar.
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 


    df_en = df.copy() # onemli

    ## ucuncu. adım
    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute


    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.loc[(df_en.BranchId == sube_no)&(df_en.day_name == days_name) & (df_en.hour == hours) ]


    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)


    ### birliktelik analizi
    df_genel = df_yedek.copy()

    df_yedek = df_yedek.CategoryName.str.strip(",")

    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
    dff

    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()

    dff = dff.drop("InvoiceNo" , axis = 1)
    dff.head()

    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    data


    from mlxtend.preprocessing import TransactionEncoder
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    df


    from mlxtend.frequent_patterns import apriori,  association_rules
    df1 = apriori(df, min_support=support, use_colnames=True)
    df1

    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    return df_association.sort_values(by='confidence', ascending=False).reset_index()

#sube_secmeli(1146, 0.01 , 0.51 , "Sunday" , 12  )

import matplotlib.pyplot as plt
#BİR SUBE BAZINDAKİ BİRLİKTELİK ANALİZ FONKSİYONLARI
normal(sube_no , support , threshold)
normal(1146, 0.01 , 0.5)
#####
day(sube_no , support , threshold , day , plot = False )
day(1147 , 0.01 , 0.51 , "Monday" )
#####    
hour(sube_no , support , threshold , hour , plot = False)
hour(1146 , 0.01 , 0.51 , 18 )
#####
day_time(sube_no , support , threshold , vakit , plot = False)
day_time(1146 , 0.01 , 0.51 , "aksam"  )
#####
ic_son(sube_no , support , threshold , ic_son , plot= False)
ic_son(1146 , 0.02 , 0.51 , "haftasonu")
#TUM SUBE BAZINDAKİ BİRLİKTELİK ANALİZ FONKSİYONLARI
tum_normal( support , threshold)
tum_normal( 0.01 , 0.51)
#####
tum_day(support , threshold , day , plot = False )
tum_day(0.01 , 0.51 , "Monday" , plot = False )
#####
tum_hour( support , threshold , hour , plot = False)
tum_hour(0.01 , 0.51 , 18 )
#####
tum_day_time(support , threshold , vakit , plot = False)
tum_day_time(0.01 , 0.51 , "sabah")
#####
tum_ic_son(support , threshold , ic_son , plot= False)
tum_ic_son(0.01 , 0.51 , "haftaici")
# TUM SECMELİ İSLEMLER BAZINDAKİ BİRLİKTELİK ANALİZ FONKSİYONLARI
tum_secmeli( support , threshold , days_name , hours)
tum_secmeli( 0.01 , 0.70 , "Friday" , 16)

# BİR SUBE SECMELİ İSLEMLER BAZINDAKİ BİRLİKTELİK ANALİZ FONKSİYONLARI
sube_secmeli(sube_no, support , threshold , days_name , hours )
sube_secmeli(1146, 0.01 , 0.51 , "Sunday" , 12 )
#SUBE BAZLI BİRLİKTELİK ANALİZİ. 1146 subesinin birliktelik kuralları.
normal(1146, 0.01 , 0.5)
# 1146 subesinin tum haftasonu alısverislerinin degerlendirilmesi
ic_son(1146 , 0.01 , 0.60 , "haftasonu")
# 1146 subesinin tum haftaici alısverislerinin degerlendirilmesi
ic_son(1146 , 0.01 , 0.60 , "haftaici")
#SABAHA GORE BİRLİKTELİK ANALİZİ
day_time(1146 , 0.01 , 0.58 , "sabah")
#OGLENE GORE BİRLİKTELİK ANALİZİ
day_time(1146 , 0.01 , 0.58 , "ogle")
#AKSAMA GORE BİRLİKTELİK ANALİZİ
day_time(1146 , 0.01 , 0.58 , "aksam")

# PAZARTESİ 
df1 = day(1146 , 0.01 , 0.50 , "Monday")
df2 = day(1125 , 0.01 , 0.50 , "Monday")
df3 = day(1137 , 0.01 , 0.50 , "Monday")
df4 = day(1108 , 0.01 , 0.50 , "Monday")
df5 = day(1143 , 0.01 , 0.50 , "Monday")
df6 = day(1118 , 0.01 , 0.50 , "Monday")
df7 = day(1131 , 0.01 , 0.50 , "Monday")
df8 = day(4005 , 0.01 , 0.50 , "Monday")
df9 = day(1133 , 0.01 , 0.50 , "Monday")
df10 = day(4007 , 0.01 , 0.50 , "Monday")
df11 = day(4006 , 0.01 , 0.50 , "Monday")
df12 = day(4003 , 0.01 , 0.50 , "Monday")
df13 = day(1106 , 0.01 , 0.50 , "Monday")
df14 = day(1115 , 0.01 , 0.50 , "Monday")
df15 = day(1107 , 0.01 , 0.50 , "Monday")
df16 = day(1117 , 0.01 , 0.50 , "Monday")
df17 = day(1135 , 0.01 , 0.50 , "Monday")
df18 = day(1119 , 0.01 , 0.50 , "Monday")
df19 = day(1148 , 0.01 , 0.50 , "Monday")
df20 = day(1126 , 0.01 , 0.50 , "Monday")
df21 = day(1140 , 0.01 , 0.50 , "Monday")
df22 = day(4004 , 0.01 , 0.50 , "Monday")
df23 = day(1124 , 0.01 , 0.50 , "Monday")
df24 = day(1130 , 0.01 , 0.50 , "Monday")
df25 = day(1127 , 0.01 , 0.50 , "Monday")
df26 = day(1122 , 0.01 , 0.50 , "Monday")
df27 = day(1142 , 0.01 , 0.50 , "Monday")
df28 = day(1114 , 0.01 , 0.50 , "Monday")
df29 = day(1145 , 0.01 , 0.50 , "Monday")
df30 = day(1109 , 0.01 , 0.50 , "Monday")
df31 = day(1139 , 0.01 , 0.50 , "Monday")
df32 = day(1160 , 0.01 , 0.50 , "Monday")
df33 = day(1128 , 0.01 , 0.50 , "Monday")
df34 = day(1102 , 0.01 , 0.50 , "Monday")
df35 = day(1103 , 0.01 , 0.50 , "Monday")
df36 = day(1141 , 0.01 , 0.50 , "Monday")
df37 = day(1132 , 0.01 , 0.50 , "Monday")
df38 = day(1129 , 0.01 , 0.50 , "Monday")
df39 = day(1134 , 0.01 , 0.50 , "Monday")
df40 = day(1123 , 0.01 , 0.50 , "Monday")
df41 = day(1113 , 0.01 , 0.50 , "Monday")
df42 = day(4010 , 0.01 , 0.50 , "Monday")
df43 = day(1144 , 0.01 , 0.50 , "Monday")
df44 = day(1116 , 0.01 , 0.50 , "Monday")
df45 = day(1136 , 0.01 , 0.50 , "Monday")
df46 = day(1112 , 0.01 , 0.50 , "Monday")
df47 = day(1138 , 0.01 , 0.50 , "Monday")
df48 = day(1104 , 0.01 , 0.50 , "Monday")
df49 = day(1101 , 0.01 , 0.50 , "Monday")
df50 = day(1110 , 0.01 , 0.50 , "Monday")
df51 = day(1147 , 0.01 , 0.50 , "Monday")
df52 = day(4009 , 0.01 , 0.50 , "Monday")
df53 = day(1111 , 0.01 , 0.50 , "Monday")
df54 = day(1120 , 0.01 , 0.50 , "Monday")
df55 = day(4010 , 0.01 , 0.50 , "Monday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]
# SALI 
df1 = day(1146 , 0.01 , 0.50 , "Tuesday")
df2 = day(1125 , 0.01 , 0.50 , "Tuesday")
df3 = day(1137 , 0.01 , 0.50 , "Tuesday")
df4 = day(1108 , 0.01 , 0.50 , "Tuesday")
df5 = day(1143 , 0.01 , 0.50 , "Tuesday")
df6 = day(1118 , 0.01 , 0.50 , "Tuesday")
df7 = day(1131 , 0.01 , 0.50 , "Tuesday")
df8 = day(4005 , 0.01 , 0.50 , "Tuesday")
df9 = day(1133 , 0.01 , 0.50 , "Tuesday")
df10 = day(4007 , 0.01 , 0.50 , "Tuesday")
df11 = day(4006 , 0.01 , 0.50 , "Tuesday")
df12 = day(4003 , 0.01 , 0.50 , "Tuesday")
df13 = day(1106 , 0.01 , 0.50 , "Tuesday")
df14 = day(1115 , 0.01 , 0.50 , "Tuesday")
df15 = day(1107 , 0.01 , 0.50 , "Tuesday")
df16 = day(1117 , 0.01 , 0.50 , "Tuesday")
df17 = day(1135 , 0.01 , 0.50 , "Tuesday")
df18 = day(1119 , 0.01 , 0.50 , "Tuesday")
df19 = day(1148 , 0.01 , 0.50 , "Tuesday")
df20 = day(1126 , 0.01 , 0.50 , "Tuesday")
df21 = day(1140 , 0.01 , 0.50 , "Tuesday")
df22 = day(4004 , 0.01 , 0.50 , "Tuesday")
df23 = day(1124 , 0.01 , 0.50 , "Tuesday")
df24 = day(1130 , 0.01 , 0.50 , "Tuesday")
df25 = day(1127 , 0.01 , 0.50 , "Tuesday")
df26 = day(1122 , 0.01 , 0.50 , "Tuesday")
df27 = day(1142 , 0.01 , 0.50 , "Tuesday")
df28 = day(1114 , 0.01 , 0.50 , "Tuesday")
df29 = day(1145 , 0.01 , 0.50 , "Tuesday")
df30 = day(1109 , 0.01 , 0.50 , "Tuesday")
df31 = day(1139 , 0.01 , 0.50 , "Tuesday")
df32 = day(1160 , 0.01 , 0.50 , "Tuesday")
df33 = day(1128 , 0.01 , 0.50 , "Tuesday")
df34 = day(1102 , 0.01 , 0.50 , "Tuesday")
df35 = day(1103 , 0.01 , 0.50 , "Tuesday")
df36 = day(1141 , 0.01 , 0.50 , "Tuesday")
df37 = day(1132 , 0.01 , 0.50 , "Tuesday")
df38 = day(1129 , 0.01 , 0.50 , "Tuesday")
df39 = day(1134 , 0.01 , 0.50 , "Tuesday")
df40 = day(1123 , 0.01 , 0.50 , "Tuesday")
df41 = day(1113 , 0.01 , 0.50 , "Tuesday")
df42 = day(4010 , 0.01 , 0.50 , "Tuesday")
df43 = day(1144 , 0.01 , 0.50 , "Tuesday")
df44 = day(1116 , 0.01 , 0.50 , "Tuesday")
df45 = day(1136 , 0.01 , 0.50 , "Tuesday")
df46 = day(1112 , 0.01 , 0.50 , "Tuesday")
df47 = day(1138 , 0.01 , 0.50 , "Tuesday")
df48 = day(1104 , 0.01 , 0.50 , "Tuesday")
df49 = day(1101 , 0.01 , 0.50 , "Tuesday")
df50 = day(1110 , 0.01 , 0.50 , "Tuesday")
df51 = day(1147 , 0.01 , 0.50 , "Tuesday")
df52 = day(4009 , 0.01 , 0.50 , "Tuesday")
df53 = day(1111 , 0.01 , 0.50 , "Tuesday")
df54 = day(1120 , 0.01 , 0.50 , "Tuesday")
df55 = day(4010 , 0.01 , 0.50 , "Tuesday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]
# CARSAMBA 

df1 = day(1146 , 0.01 , 0.50 , "Wednesday")
df2 = day(1125 , 0.01 , 0.50 , "Wednesday")
df3 = day(1137 , 0.01 , 0.50 , "Wednesday")
df4 = day(1108 , 0.01 , 0.50 , "Wednesday")
df5 = day(1143 , 0.01 , 0.50 , "Wednesday")
df6 = day(1118 , 0.01 , 0.50 , "Wednesday")
df7 = day(1131 , 0.01 , 0.50 , "Wednesday")
df8 = day(4005 , 0.01 , 0.50 , "Wednesday")
df9 = day(1133 , 0.01 , 0.50 , "Wednesday")
df10 = day(4007 , 0.01 , 0.50 , "Wednesday")
df11 = day(4006 , 0.01 , 0.50 , "Wednesday")
df12 = day(4003 , 0.01 , 0.50 , "Wednesday")
df13 = day(1106 , 0.01 , 0.50 , "Wednesday")
df14 = day(1115 , 0.01 , 0.50 , "Wednesday")
df15 = day(1107 , 0.01 , 0.50 , "Wednesday")
df16 = day(1117 , 0.01 , 0.50 , "Wednesday")
df17 = day(1135 , 0.01 , 0.50 , "Wednesday")
df18 = day(1119 , 0.01 , 0.50 , "Wednesday")
df19 = day(1148 , 0.01 , 0.50 , "Wednesday")
df20 = day(1126 , 0.01 , 0.50 , "Wednesday")
df21 = day(1140 , 0.01 , 0.50 , "Wednesday")
df22 = day(4004 , 0.01 , 0.50 , "Wednesday")
df23 = day(1124 , 0.01 , 0.50 , "Wednesday")
df24 = day(1130 , 0.01 , 0.50 , "Wednesday")
df25 = day(1127 , 0.01 , 0.50 , "Wednesday")
df26 = day(1122 , 0.01 , 0.50 , "Wednesday")
df27 = day(1142 , 0.01 , 0.50 , "Wednesday")
df28 = day(1114 , 0.01 , 0.50 , "Wednesday")
df29 = day(1145 , 0.01 , 0.50 , "Wednesday")
df30 = day(1109 , 0.01 , 0.50 , "Wednesday")
df31 = day(1139 , 0.01 , 0.50 , "Wednesday")
df32 = day(1160 , 0.01 , 0.50 , "Wednesday")
df33 = day(1128 , 0.01 , 0.50 , "Wednesday")
df34 = day(1102 , 0.01 , 0.50 , "Wednesday")
df35 = day(1103 , 0.01 , 0.50 , "Wednesday")
df36 = day(1141 , 0.01 , 0.50 , "Wednesday")
df37 = day(1132 , 0.01 , 0.50 , "Wednesday")
df38 = day(1129 , 0.01 , 0.50 , "Wednesday")
df39 = day(1134 , 0.01 , 0.50 , "Wednesday")
df40 = day(1123 , 0.01 , 0.50 , "Wednesday")
df41 = day(1113 , 0.01 , 0.50 , "Wednesday")
df42 = day(4010 , 0.01 , 0.50 , "Wednesday")
df43 = day(1144 , 0.01 , 0.50 , "Wednesday")
df44 = day(1116 , 0.01 , 0.50 , "Wednesday")
df45 = day(1136 , 0.01 , 0.50 , "Wednesday")
df46 = day(1112 , 0.01 , 0.50 , "Wednesday")
df47 = day(1138 , 0.01 , 0.50 , "Wednesday")
df48 = day(1104 , 0.01 , 0.50 , "Wednesday")
df49 = day(1101 , 0.01 , 0.50 , "Wednesday")
df50 = day(1110 , 0.01 , 0.50 , "Wednesday")
df51 = day(1147 , 0.01 , 0.50 , "Wednesday")
df52 = day(4009 , 0.01 , 0.50 , "Wednesday")
df53 = day(1111 , 0.01 , 0.50 , "Wednesday")
df54 = day(1120 , 0.01 , 0.50 , "Wednesday")
df55 = day(4010 , 0.01 , 0.50 , "Wednesday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]
# PERSEMBE 
df1 = day(1146 , 0.01 , 0.50 , "Thursday")
df2 = day(1125 , 0.01 , 0.50 , "Thursday")
df3 = day(1137 , 0.01 , 0.50 , "Thursday")
df4 = day(1108 , 0.01 , 0.50 , "Thursday")
df5 = day(1143 , 0.01 , 0.50 , "Thursday")
df6 = day(1118 , 0.01 , 0.50 , "Thursday")
df7 = day(1131 , 0.01 , 0.50 , "Thursday")
df8 = day(4005 , 0.01 , 0.50 , "Thursday")
df9 = day(1133 , 0.01 , 0.50 , "Thursday")
df10 = day(4007 , 0.01 , 0.50 , "Thursday")
df11 = day(4006 , 0.01 , 0.50 , "Thursday")
df12 = day(4003 , 0.01 , 0.50 , "Thursday")
df13 = day(1106 , 0.01 , 0.50 , "Thursday")
df14 = day(1115 , 0.01 , 0.50 , "Thursday")
df15 = day(1107 , 0.01 , 0.50 , "Thursday")
df16 = day(1117 , 0.01 , 0.50 , "Thursday")
df17 = day(1135 , 0.01 , 0.50 , "Thursday")
df18 = day(1119 , 0.01 , 0.50 , "Thursday")
df19 = day(1148 , 0.01 , 0.50 , "Thursday")
df20 = day(1126 , 0.01 , 0.50 , "Thursday")
df21 = day(1140 , 0.01 , 0.50 , "Thursday")
df22 = day(4004 , 0.01 , 0.50 , "Thursday")
df23 = day(1124 , 0.01 , 0.50 , "Thursday")
df24 = day(1130 , 0.01 , 0.50 , "Thursday")
df25 = day(1127 , 0.01 , 0.50 , "Thursday")
df26 = day(1122 , 0.01 , 0.50 , "Thursday")
df27 = day(1142 , 0.01 , 0.50 , "Thursday")
df28 = day(1114 , 0.01 , 0.50 , "Thursday")
df29 = day(1145 , 0.01 , 0.50 , "Thursday")
df30 = day(1109 , 0.01 , 0.50 , "Thursday")
df31 = day(1139 , 0.01 , 0.50 , "Thursday")
df32 = day(1160 , 0.01 , 0.50 , "Thursday")
df33 = day(1128 , 0.01 , 0.50 , "Thursday")
df34 = day(1102 , 0.01 , 0.50 , "Thursday")
df35 = day(1103 , 0.01 , 0.50 , "Thursday")
df36 = day(1141 , 0.01 , 0.50 , "Thursday")
df37 = day(1132 , 0.01 , 0.50 , "Thursday")
df38 = day(1129 , 0.01 , 0.50 , "Thursday")
df39 = day(1134 , 0.01 , 0.50 , "Thursday")
df40 = day(1123 , 0.01 , 0.50 , "Thursday")
df41 = day(1113 , 0.01 , 0.50 , "Thursday")
df42 = day(4010 , 0.01 , 0.50 , "Thursday")
df43 = day(1144 , 0.01 , 0.50 , "Thursday")
df44 = day(1116 , 0.01 , 0.50 , "Thursday")
df45 = day(1136 , 0.01 , 0.50 , "Thursday")
df46 = day(1112 , 0.01 , 0.50 , "Thursday")
df47 = day(1138 , 0.01 , 0.50 , "Thursday")
df48 = day(1104 , 0.01 , 0.50 , "Thursday")
df49 = day(1101 , 0.01 , 0.50 , "Thursday")
df50 = day(1110 , 0.01 , 0.50 , "Thursday")
df51 = day(1147 , 0.01 , 0.50 , "Thursday")
df52 = day(4009 , 0.01 , 0.50 , "Thursday")
df53 = day(1111 , 0.01 , 0.50 , "Thursday")
df54 = day(1120 , 0.01 , 0.50 , "Thursday")
df55 = day(4010 , 0.01 , 0.50 , "Thursday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]

# CUMA 


df1 = day(1146 , 0.01 , 0.50 , "Friday")
df2 = day(1125 , 0.01 , 0.50 , "Friday")
df3 = day(1137 , 0.01 , 0.50 , "Friday")
df4 = day(1108 , 0.01 , 0.50 , "Friday")
df5 = day(1143 , 0.01 , 0.50 , "Friday")
df6 = day(1118 , 0.01 , 0.50 , "Friday")
df7 = day(1131 , 0.01 , 0.50 , "Friday")
df8 = day(4005 , 0.01 , 0.50 , "Friday")
df9 = day(1133 , 0.01 , 0.50 , "Friday")
df10 = day(4007 , 0.01 , 0.50 , "Friday")
df11 = day(4006 , 0.01 , 0.50 , "Friday")
df12 = day(4003 , 0.01 , 0.50 , "Friday")
df13 = day(1106 , 0.01 , 0.50 , "Friday")
df14 = day(1115 , 0.01 , 0.50 , "Friday")
df15 = day(1107 , 0.01 , 0.50 , "Friday")
df16 = day(1117 , 0.01 , 0.50 , "Friday")
df17 = day(1135 , 0.01 , 0.50 , "Friday")
df18 = day(1119 , 0.01 , 0.50 , "Friday")
df19 = day(1148 , 0.01 , 0.50 , "Friday")
df20 = day(1126 , 0.01 , 0.50 , "Friday")
df21 = day(1140 , 0.01 , 0.50 , "Friday")
df22 = day(4004 , 0.01 , 0.50 , "Friday")
df23 = day(1124 , 0.01 , 0.50 , "Friday")
df24 = day(1130 , 0.01 , 0.50 , "Friday")
df25 = day(1127 , 0.01 , 0.50 , "Friday")
df26 = day(1122 , 0.01 , 0.50 , "Friday")
df27 = day(1142 , 0.01 , 0.50 , "Friday")
df28 = day(1114 , 0.01 , 0.50 , "Friday")
df29 = day(1145 , 0.01 , 0.50 , "Friday")
df30 = day(1109 , 0.01 , 0.50 , "Friday")
df31 = day(1139 , 0.01 , 0.50 , "Friday")
df32 = day(1160 , 0.01 , 0.50 , "Friday")
df33 = day(1128 , 0.01 , 0.50 , "Friday")
df34 = day(1102 , 0.01 , 0.50 , "Friday")
df35 = day(1103 , 0.01 , 0.50 , "Friday")
df36 = day(1141 , 0.01 , 0.50 , "Friday")
df37 = day(1132 , 0.01 , 0.50 , "Friday")
df38 = day(1129 , 0.01 , 0.50 , "Friday")
df39 = day(1134 , 0.01 , 0.50 , "Friday")
df40 = day(1123 , 0.01 , 0.50 , "Friday")
df41 = day(1113 , 0.01 , 0.50 , "Friday")
df42 = day(4010 , 0.01 , 0.50 , "Friday")
df43 = day(1144 , 0.01 , 0.50 , "Friday")
df44 = day(1116 , 0.01 , 0.50 , "Friday")
df45 = day(1136 , 0.01 , 0.50 , "Friday")
df46 = day(1112 , 0.01 , 0.50 , "Friday")
df47 = day(1138 , 0.01 , 0.50 , "Friday")
df48 = day(1104 , 0.01 , 0.50 , "Friday")
df49 = day(1101 , 0.01 , 0.50 , "Friday")
df50 = day(1110 , 0.01 , 0.50 , "Friday")
df51 = day(1147 , 0.01 , 0.50 , "Friday")
df52 = day(4009 , 0.01 , 0.50 , "Friday")
df53 = day(1111 , 0.01 , 0.50 , "Friday")
df54 = day(1120 , 0.01 , 0.50 , "Friday")
df55 = day(4010 , 0.01 , 0.50 , "Friday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]
# CUMARTESİ

df1 = day(1146 , 0.01 , 0.50 , "Saturday")
df2 = day(1125 , 0.01 , 0.50 , "Saturday")
df3 = day(1137 , 0.01 , 0.50 , "Saturday")
df4 = day(1108 , 0.01 , 0.50 , "Saturday")
df5 = day(1143 , 0.01 , 0.50 , "Saturday")
df6 = day(1118 , 0.01 , 0.50 , "Saturday")
df7 = day(1131 , 0.01 , 0.50 , "Saturday")
df8 = day(4005 , 0.01 , 0.50 , "Saturday")
df9 = day(1133 , 0.01 , 0.50 , "Saturday")
df10 = day(4007 , 0.01 , 0.50 , "Saturday")
df11 = day(4006 , 0.01 , 0.50 , "Saturday")
df12 = day(4003 , 0.01 , 0.50 , "Saturday")
df13 = day(1106 , 0.01 , 0.50 , "Saturday")
df14 = day(1115 , 0.01 , 0.50 , "Saturday")
df15 = day(1107 , 0.01 , 0.50 , "Saturday")
df16 = day(1117 , 0.01 , 0.50 , "Saturday")
df17 = day(1135 , 0.01 , 0.50 , "Saturday")
df18 = day(1119 , 0.01 , 0.50 , "Saturday")
df19 = day(1148 , 0.01 , 0.50 , "Saturday")
df20 = day(1126 , 0.01 , 0.50 , "Saturday")
df21 = day(1140 , 0.01 , 0.50 , "Saturday")
df22 = day(4004 , 0.01 , 0.50 , "Saturday")
df23 = day(1124 , 0.01 , 0.50 , "Saturday")
df24 = day(1130 , 0.01 , 0.50 , "Saturday")
df25 = day(1127 , 0.01 , 0.50 , "Saturday")
df26 = day(1122 , 0.01 , 0.50 , "Saturday")
df27 = day(1142 , 0.01 , 0.50 , "Saturday")
df28 = day(1114 , 0.01 , 0.50 , "Saturday")
df29 = day(1145 , 0.01 , 0.50 , "Saturday")
df30 = day(1109 , 0.01 , 0.50 , "Saturday")
df31 = day(1139 , 0.01 , 0.50 , "Saturday")
df32 = day(1160 , 0.01 , 0.50 , "Saturday")
df33 = day(1128 , 0.01 , 0.50 , "Saturday")
df34 = day(1102 , 0.01 , 0.50 , "Saturday")
df35 = day(1103 , 0.01 , 0.50 , "Saturday")
df36 = day(1141 , 0.01 , 0.50 , "Saturday")
df37 = day(1132 , 0.01 , 0.50 , "Saturday")
df38 = day(1129 , 0.01 , 0.50 , "Saturday")
df39 = day(1134 , 0.01 , 0.50 , "Saturday")
df40 = day(1123 , 0.01 , 0.50 , "Saturday")
df41 = day(1113 , 0.01 , 0.50 , "Saturday")
df42 = day(4010 , 0.01 , 0.50 , "Saturday")
df43 = day(1144 , 0.01 , 0.50 , "Saturday")
df44 = day(1116 , 0.01 , 0.50 , "Saturday")
df45 = day(1136 , 0.01 , 0.50 , "Saturday")
df46 = day(1112 , 0.01 , 0.50 , "Saturday")
df47 = day(1138 , 0.01 , 0.50 , "Saturday")
df48 = day(1104 , 0.01 , 0.50 , "Saturday")
df49 = day(1101 , 0.01 , 0.50 , "Saturday")
df50 = day(1110 , 0.01 , 0.50 , "Saturday")
df51 = day(1147 , 0.01 , 0.50 , "Saturday")
df52 = day(4009 , 0.01 , 0.50 , "Saturday")
df53 = day(1111 , 0.01 , 0.50 , "Saturday")
df54 = day(1120 , 0.01 , 0.50 , "Saturday")
df55 = day(4010 , 0.01 , 0.50 , "Saturday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]
# PAZAR

df1 = day(1146 , 0.01 , 0.50 , "Sunday")
df2 = day(1125 , 0.01 , 0.50 , "Sunday")
df3 = day(1137 , 0.01 , 0.50 , "Sunday")
df4 = day(1108 , 0.01 , 0.50 , "Sunday")
df5 = day(1143 , 0.01 , 0.50 , "Sunday")
df6 = day(1118 , 0.01 , 0.50 , "Sunday")
df7 = day(1131 , 0.01 , 0.50 , "Sunday")
df8 = day(4005 , 0.01 , 0.50 , "Sunday")
df9 = day(1133 , 0.01 , 0.50 , "Sunday")
df10 = day(4007 , 0.01 , 0.50 , "Sunday")
df11 = day(4006 , 0.01 , 0.50 , "Sunday")
df12 = day(4003 , 0.01 , 0.50 , "Sunday")
df13 = day(1106 , 0.01 , 0.50 , "Sunday")
df14 = day(1115 , 0.01 , 0.50 , "Sunday")
df15 = day(1107 , 0.01 , 0.50 , "Sunday")
df16 = day(1117 , 0.01 , 0.50 , "Sunday")
df17 = day(1135 , 0.01 , 0.50 , "Sunday")
df18 = day(1119 , 0.01 , 0.50 , "Sunday")
df19 = day(1148 , 0.01 , 0.50 , "Sunday")
df20 = day(1126 , 0.01 , 0.50 , "Sunday")
df21 = day(1140 , 0.01 , 0.50 , "Sunday")
df22 = day(4004 , 0.01 , 0.50 , "Sunday")
df23 = day(1124 , 0.01 , 0.50 , "Sunday")
df24 = day(1130 , 0.01 , 0.50 , "Sunday")
df25 = day(1127 , 0.01 , 0.50 , "Sunday")
df26 = day(1122 , 0.01 , 0.50 , "Sunday")
df27 = day(1142 , 0.01 , 0.50 , "Sunday")
df28 = day(1114 , 0.01 , 0.50 , "Sunday")
df29 = day(1145 , 0.01 , 0.50 , "Sunday")
df30 = day(1109 , 0.01 , 0.50 , "Sunday")
df31 = day(1139 , 0.01 , 0.50 , "Sunday")
df32 = day(1160 , 0.01 , 0.50 , "Sunday")
df33 = day(1128 , 0.01 , 0.50 , "Sunday")
df34 = day(1102 , 0.01 , 0.50 , "Sunday")
df35 = day(1103 , 0.01 , 0.50 , "Sunday")
df36 = day(1141 , 0.01 , 0.50 , "Sunday")
df37 = day(1132 , 0.01 , 0.50 , "Sunday")
df38 = day(1129 , 0.01 , 0.50 , "Sunday")
df39 = day(1134 , 0.01 , 0.50 , "Sunday")
df40 = day(1123 , 0.01 , 0.50 , "Sunday")
df41 = day(1113 , 0.01 , 0.50 , "Sunday")
df42 = day(4010 , 0.01 , 0.50 , "Sunday")
df43 = day(1144 , 0.01 , 0.50 , "Sunday")
df44 = day(1116 , 0.01 , 0.50 , "Sunday")
df45 = day(1136 , 0.01 , 0.50 , "Sunday")
df46 = day(1112 , 0.01 , 0.50 , "Sunday")
df47 = day(1138 , 0.01 , 0.50 , "Sunday")
df48 = day(1104 , 0.01 , 0.50 , "Sunday")
df49 = day(1101 , 0.01 , 0.50 , "Sunday")
df50 = day(1110 , 0.01 , 0.50 , "Sunday")
df51 = day(1147 , 0.01 , 0.50 , "Sunday")
df52 = day(4009 , 0.01 , 0.50 , "Sunday")
df53 = day(1111 , 0.01 , 0.50 , "Sunday")
df54 = day(1120 , 0.01 , 0.50 , "Sunday")
df55 = day(4010 , 0.01 , 0.50 , "Sunday")
dff = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55] , axis = 0)

dff[["antecedents","consequents"]].value_counts()[0:60]

# GUNLERE GORE SATISLAR
def day_satıs(sube_no):
    """
    Secilecek olan bir subenin (sube_no)  gunlere gore satıs grafıgını verir.
    
    """
    import matplotlib.pyplot as plt
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    df = satislar_load()

    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 

    df_en = df.loc[df.BranchId == sube_no] # onemli

    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute

    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    df_en

    df_yedek = df_en.copy()

    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)

    return df_yedek.groupby("day_name").agg({"InvoiceNo":"count"})

day_satıs(1146)
#GUNLERE GORE BİRLİKTELİK KURALLARI
df_mon = day(1146 , 0.01 , 0.60 , "Monday" )
df_mon
#SALI GUNUNUN BİRLİKTELİK ANALİZİ
df_tu = day(1146 , 0.01 , 0.50 , "Tuesday")
df_tu
#CARSAMBA GUNUNUN BİRLİKTELİK ANALİZİ
df_we = day(1146 , 0.01 , 0.50 , "Wednesday")
df_we
#PERSEMBE GUNUNUN BİRLİKTELİK ANALİZİ
df_th = day(1146 , 0.01 , 0.50 , "Thursday")
df_th
#CUMA GUNUNUN BİRLİKTELİK ANALİZİ
df_fr = day(1146 , 0.01 , 0.50 , "Friday")
df_fr
#CUMARTESİ GUNUNUN BİRLİKTELİK ANALİZİ
df_sa = day(1146 , 0.01 , 0.50 , "Saturday")
df_sa
#PAZAR GUNUNUN BİRLİKTELİK ANALİZİ
df_su = day(1146 , 0.01 , 0.50 , "Sunday")
df_su
df_all_day = pd.concat([df_mon , df_tu , df_we , df_th , df_fr , df_sa , df_su] , axis = 0)
df_all_day[["antecedents","consequents"]].value_counts()[0:60]

df_all_day[["antecedents","consequents"]].value_counts()[60:120]
df_all_day[["antecedents","consequents"]].value_counts()[120:180]
##############################################################################################################################################################################################################################
