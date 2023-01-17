import numpy as np

import pandas as ps

import seaborn as sns



df = sns.load_dataset('tips') # Seaborn kütüphanesindeki bir veri setini çağırır.
df.info()



# Veri çerçevesinin genel özetini verir.
df.shape 



# Veri çerçevesinde kaç satır ve sütunun olduğunu söyler.
df.head() 



# Veri çerçevesinin tüm sütunlarıyla ilk 5 satırı görüntüler.
df.tail() 



# Veri çerçevesinin son 5 satırını gösterir.
df.describe().T 



# Sadece sayısal verilere ait temel istatistik bilgilerini gösterir.

# Transpose ile tablonun görünümü değişti.
df.sample(10)



# Veri çerçevesi içerisindeki rastgele 10 değeri alır.
df.isna().sum()



# Değişkenlerdeki eksik değerleri sayar.
df.count()



# Her bir sütunda kaç satır olduğunu sayar.
df["tip"]



# Sadece tip sütununa bakalım.

# df["tip"][herhangi bir değer] şeklinde de çalışabilir. İstenilen değer alınabilir.
df["day"].unique()



# Kaç adet benzersiz gün var? Tüm haftanın günleri var mı?
df["sex"].unique()



# Cinsiyet beyanında bulunmayan oldu mu?
df["smoker"].unique()



# Sigara içen ve içmeyen olarak mı işaretlenmiş sadece?
df["size"].unique()



# Kaç benzersiz porsiyon var?
df["size"].nunique()



# Sayamayacağımız kadar değer olduğunda kaç benzersiz değerin olduğunu söyler.
df["time"].unique()



# Yemek zamanı kaç benzersiz türde?
df["total_bill"].mean()



# Ücretin ortalaması.
df["total_bill"].std()



# Bırakılan ücretin standart sapması.
df["total_bill"].median()



# Ücretin ortanca değeri.
df["size"].describe()



# Porsiyonların temel istatistik değerleri.
df.groupby(["size"]).mean()



# Her bir porsiyon için ödenen ücret ve bırakılan bahşişlerin ortalaması.
df.groupby(["sex"]).describe()["total_bill"]



# Cinsiyete göre ödenen ücretin temel değerleri.
df["tip"].mode()



# En sık tekrar eden bahşiş miktarı.
df.groupby("size")["tip"].apply(lambda x: np.mean(x))



# Porsiyonlara göre bırakılan bahşişlerin ortalaması. Aralarında korelasyon gözüküyor.
df.corr()["size"]["tip"]



# Tahmin edildiği gibi pozitif ve orta düzeyde bir korelasyon var.
df[(df["sex"] == "Female") & (df["tip"] > 5)]



# 5 liranın üzerinde bahşiş bırakan kadın müşteriler.
df[(df["sex"] == "Female") & (df["tip"] > 5)].sort_values("total_bill", axis=0, ascending=False)



# 5 liranın üzerinde bahşiş bırakan kadın müşteriler ödedikleri ücrete göre sıralandı.
df.sort_values("tip", axis=0, ascending=False).head()[["tip", "day", "smoker", "sex", "size"]]



# En yüksek bahşiş bırakan müşteriler miktar, cinsiyet, gün ve sigara durumuna göre sıralandı.

# Cumartesi gelen ve sigara içmeyen erkek müşterilerin en çok bahşiş verebilme ihtimali çıkartılabilir buradan.  
df_filtered = df.query('tip > 6')[["tip", "day", "smoker", "sex", "size"]].sort_values("tip", axis=0, ascending=False)

df_filtered
sns.set(rc={'figure.figsize':(8,6)})



# Grafikleri boyutlandırmak için kullandık.
sns.scatterplot(x = "total_bill", y = "tip", data = df, color = "blue")



# Hesap miktarıyla bırakılan bahşiş arasındaki dağılımı gösterir.
sns.jointplot(x = "total_bill", y = "tip", data = df, color = "purple")



# Değerlerin üst üste bindiği durumlarda aralıkların yoğunluğunu da gösterir.
df.corr()
corr = df.corr()

sns.heatmap(corr,

           xticklabels = corr.columns.values,

           yticklabels = corr.columns.values)



# Ücret, bahşiş ve porsiyon büyüklüğü arasındaki korelasyona ısı haritası uygular.

# Porsiyon ile bahşiş arasında güçlü sayılabilecek bir pozitif ilişki var. 
sns.scatterplot(x = "total_bill", y = "tip", data = df, hue = "sex")
sns.distplot(df["total_bill"], bins = 16, color = "purple")
sns.distplot(df["tip"], bins = 16, color = "purple")
sns.jointplot(x = df["total_bill"], y = df["tip"], kind = "kde", color = "purple")
sns.scatterplot(x = "total_bill", y = "tip", data = df, hue = "sex", size = "smoker")
sns.scatterplot(x = "total_bill", y = "tip", data = df, hue = "sex", size = "smoker", style = "time")
sns.lmplot(x = "total_bill", y = "tip", data = df, hue = "sex")
sns.violinplot(x = "size", data = df)
sns.violinplot(x = "total_bill", y = "smoker", data = df)
sns.violinplot(x = "tip", y = "smoker", data = df)
sns.pairplot(df, hue = "sex", palette = "Set2")
sns.barplot(x = "sex", y = "total_bill", data = df)
sns.barplot(x = "day", y = "total_bill", data = df)
sns.barplot(x = "day", y = "size", data = df)
sns.catplot(x = "day", y = "tip", data = df)
sns.countplot(x = "sex", data = df)
sns.barplot("time", "total_bill", "smoker", data = df)