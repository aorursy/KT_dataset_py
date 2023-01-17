#Kutuphanelerin yuklenmesi.
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
#Veri setinin fonksiyonla getirilmesi.
def satislar_load():
    dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
    return dff

df = satislar_load()

#Degisken isimlerinin belirlenmesi.
df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
#Veri setinin gozlem sayısı.
len(df)
#Veriden ilk 5 satır.

df.head(3)
#TUM SUBELERDEN EN FAZLA ALINAN URUNLER (URUNLER VE ALINMA MİKTARLARI)

df_yedek = df.copy()
df_yedek = df_yedek.CategoryName.str.strip(",").value_counts()[0:40]

#İLK 5 GOZLEM DEGERİ.
df_yedek.head(3)
#SECİLEN SUBEDEN EN FAZLA ALINAN URUNLER (URUNLER VE ALINMA MİKTARLARI)

df_en = df.copy()
df_en = df.loc[df.BranchId == 1146]
df_en.CategoryName.str.strip(",").value_counts()[0:3]
#SUBELERE GORE EN FAZLA SATILAN URUNLER

a = [1129, 1141, 1109, 1128, 1143, 1148, 1124, 1127, 1130, 1131, 1137,
       1102, 1104, 1110, 1111, 1115, 1118, 1120, 1136, 4010, 1103, 1122,
       1126, 1140, 1145, 1147, 1160, 1106, 1108, 1113, 1119, 1125, 1135,
       1139, 1142, 1146, 4003, 4004, 4006, 4007, 1101, 1112, 1114, 1116,
       1132, 1133, 1138, 1144, 1107, 1117, 1121, 1123, 1134, 4009, 4005]

for i in a:
    print(i , "*"*100)
    print(df.loc[df.BranchId == i].CategoryName.str.strip(",").value_counts()[0:5])
    
    
# TANIMSIZ URUNUNE BAKIS. "Tanımsız" URUNLERİN OGLEDEN SONRA DAHA FAZLA SATILDIGI GORULMEKTEDİR.
#"CategoryName" DEGİSKENİNDE "Tanımsız" İSİMLENDİRMESİ MEVCUT. 


#BU DEGİSKENİ SAAT VE GUNE GORE GRAFİKLESTİRECEGİZ.


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

#Grafigin boyutlarını ayarladım.
sns.set(rc={'figure.figsize':(11.7,8.27)})

#Grafigi cizfirdim.
sns.countplot(x="hour", hue="day", data=df_tanimsiz);
# En fazla hangi subelerden alısberis yapılmıs. Toplamda 55 sube var.
# 1146 subesinde daha cok siparis oldugu icin ilk ona bakıyorum. 48146 adet geçmiş.

df.groupby("BranchId")["InvoiceNo"].count().sort_values(ascending= False).head(3)
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
    #Kutuphanelerin yuklenmesi
    import numpy as np 
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mlxtend.frequent_patterns import apriori,  association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    #Veri setinin fonksiyonla getirilmesi.
    def satislar_load():
        dff = pd.read_csv("../input/satisss/satislar kopyas.csv" , sep = ";")
        return dff
    
    #Veri setinin degiskene atanması
    df = satislar_load()
    
    #Degisken isimlerinin belirlenmesi.
    df.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
    
    #Secilen degiskenlerin silinmesi.
    df.drop(columns= ['PosId', "StockCode", "Line",'Quantity','CategoryCode'], inplace=True) 
    
    #Fonksiyona girilen "sube_no" degerine gore , o subenin tum gozlem degerlerinin secilmesi.
    df_en = df.loc[df.BranchId == sube_no] # onemli

    #"InvoiceDate" degiskeninin yıl , ay , gun , saat ve dakika cinsinden degiskenlere donusturulmesi.
    df_en["InvoiceDate"] = df_en["InvoiceDate"].astype('datetime64[ns]')
    df_en["year"] =df_en["InvoiceDate"].dt.year
    df_en["month"] =df_en["InvoiceDate"].dt.month
    df_en["day"] =df_en["InvoiceDate"].dt.day
    df_en["hour"] =df_en["InvoiceDate"].dt.hour
    df_en["minute"] =df_en["InvoiceDate"].dt.minute
    
    #"InvoiceDate" degiskeninin icinden gun isimlerini alarak yeni degisken olusturulması.
    df_en["day_name"]= df_en.InvoiceDate.dt.day_name()
    
    #Yedek olusturulması
    df_yedek = df_en.copy()
    
    #"InvoiceDate","year" ve "month" degiskenini kullanmayacagımızdan dolayı siliyoruz.
    df_yedek = df_yedek.drop(["InvoiceDate","year","month"] , axis = 1)
    
    #Fonksiyonda secilecek gun ismine gore tum gozlem birimini getirir.
    df_yedek = df_yedek.loc[df_yedek.day_name == day]
    
    #Tum urunlerin tekil isimlerini yazdırdık.
    df_urunler = df_yedek.CategoryName.unique()

    #tum zamanlara(saat ve gunler) gore urunlerin satısının grafigi.
    
    if plot:
        for a in df_urunler:
            for i in ["hour" ,"day_name"]:
                sns.set(rc={'figure.figsize':(11.7,8.27)})
                sns.countplot(df_yedek.loc[df_yedek.CategoryName == a, "CategoryName"], hue=df_yedek[i])
                plt.show()

    # Birliktelik analizi
    
    #Yedek olusturulması
    df_genel = df_yedek.copy()
    
    # CategoryName degiskenindeki "," isaretinden kurtulduk.
    df_yedek = df_yedek.CategoryName.str.strip(",")
    
    #Birlestirme islemi.
    dff = pd.concat([df_genel.InvoiceNo , df_yedek] , axis = 1 )
   
    #Her faturanın ıcındeki urunleri yan yana virgul ile birlestirdim.Yani bir faturanın urunleri tek satırda birlesti.
    dff = dff.groupby('InvoiceNo')['CategoryName'].agg(','.join).reset_index()
    #Degiskenin silinmesi.
    dff = dff.drop("InvoiceNo" , axis = 1)
    data = list(dff['CategoryName'].apply(lambda x:x.split(",")))
    


    #Encod islemi ve uygulanması.
    tencoder = TransactionEncoder()
    te_data = tencoder.fit(data).transform(data)
    df = pd.DataFrame(te_data, columns=tencoder.columns_)
    


    #Apriori algoritmasında secilen "min_support" degerine gore birliktelik analizlerinin olusturulması.
    df1 = apriori(df, min_support=support, use_colnames=True)
   
    #Birliktelik kurallarının ayrıntılı olarak cıkarılması. Ve fonksiyonun cıktısı olan degisken.
    df_association = association_rules(df1, metric = 'confidence', min_threshold = threshold)
    
    #Gun ısmının yazılması
    print(day)
    
    return df_association.sort_values(by='confidence', ascending=False).reset_index()


#day(1147 , 0.01 , 0.51 , "Monday" )
# GUNLERE GORE SATISLARIN LİSTELENMESİ
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
#PAZARTESİ GUNUNUN BİRLİKTELİK ANALİZİ
df_mon = day(1146 , 0.01 , 0.60 , "Monday" )
df_mon.head(3)
#SALI GUNUNUN BİRLİKTELİK ANALİZİ
df_tu = day(1146 , 0.01 , 0.50 , "Tuesday")
df_tu.head(3)
#CARSAMBA GUNUNUN BİRLİKTELİK ANALİZİ
df_we = day(1146 , 0.01 , 0.50 , "Wednesday")
df_we.head(3)
#PERSEMBE GUNUNUN BİRLİKTELİK ANALİZİ
df_th = day(1146 , 0.01 , 0.50 , "Thursday")
df_th.head(3)
#CUMA GUNUNUN BİRLİKTELİK ANALİZİ
df_fr = day(1146 , 0.01 , 0.50 , "Friday")
df_fr.head(3)
#CUMARTESİ GUNUNUN BİRLİKTELİK ANALİZİ
df_sa = day(1146 , 0.01 , 0.50 , "Saturday")
df_sa.head(3)
#PAZAR GUNUNUN BİRLİKTELİK ANALİZİ
df_su = day(1146 , 0.01 , 0.50 , "Sunday")
df_su.head(3)
#Cıkarılan sonucların birlestirilmesi. Bunu yapmamın nedeni; tum gunleri birlikte degerlendirmek istememdi.
df_all_day = pd.concat([df_mon , df_tu , df_we , df_th , df_fr , df_sa , df_su] , axis = 0)

#Bu kodlar sadece gorsellestirmek amacıyla kullanılmıstır.
df_all_day[["antecedents","consequents"]].value_counts()[0:60]
df_all_day[["antecedents","consequents"]].value_counts()[60:120]
df_all_day[["antecedents","consequents"]].value_counts()[120:180]