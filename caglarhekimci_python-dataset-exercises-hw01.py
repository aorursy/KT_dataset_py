import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/irisdataset/iris.csv")
df.head(5)
df.shape
df.info()
df.describe().T # Standart sapma varyansın kare köküdür.
df.isna().sum()
df.corr() 
# korelasyon; 1 ile -1 arası değer alırlar. öznitelikler arası ilişkinin yönünü ve şiddetini tanımlar.
# en güçlü pozitif ilişki 0.96 ile petal.length ile petal.width arasındadır.
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
# heatmap() fonksiyonu seaborn paketinde yer almaktadır.
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df, color="green");
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="black");
# Üst üste binen gözlemlerin okunabilirliğini kolaylaştırmak için kullanılır. Hangi aralıkta ne kadar yoğun olduğunu görebiliriz
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = "variety", data = df);
# Turuncu ve yeşil renklerin ortak noktaları olduğu gibi ayrı noktalarının olduğunu ve mavi renkteki noktaların da bu iki 
# renkten bağımsız olduğunu görebiliyoruz. Noktalar görüldüğü üzere küme şeklinde ve birbirlerinden çok bağımsız değiller.
# Bu yüzden kümeleme işlemi ayırt etmek için yapılabilir.
df["variety"].value_counts()
# value_counts() ;Sütundaki NaN olmayan her bir unique değerin kaç kez kullanıldığını azalan şekilde gösteren bir seri döndürür.
# 50x3 ve Dengeli bir dağılım.
sns.violinplot(y="sepal.width", data = df);
# Çok dalgalanma olmamakla birlikte 3 değerinde artış, 3'ten küçük ve büyük değerlere doğru azalma gözüküyor.Normal bir dağılım.
sns.distplot(df["sepal.width"], color="black");
sns.violinplot(x = df["variety"], y = df["sepal.length"], data = df );
sns.countplot(x="variety", data = df);
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df);
#length 5.5-6 arası width, width 3.5-4 arasıda length en yüksek değerlere ulaşmış.
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, kind="kde", color="purple");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = 'petal.length', y = 'petal.width', hue = 'variety', data = df);
sns.lmplot(x="petal.length", y="petal.width", data = df);
#Şekilde de görüldüğü gibi pozitif yönde, doğrusal ve çok güçlü bir ilişki vardır. Ne kadar yapışık o kadar güçlü.
df[["petal.width","petal.length"]].corr()
# Birden çok elemanı olan dizilerde 2 kapalı parantez kullanılıyor
df["total.length"] = df[['petal.length','sepal.length']].sum(axis=1)
df
# a xis = 0 satırlarla, axis = 1 sütunlarla işlem yapar.
# Birden çok elemanı olan dizilerde 2 kapalı parantez kullanılıyor
df["total.length"].mean(axis=0)
# Fakat tek elemanlarda gerek yok.
df["total.length"].std(axis=0)
df["sepal.length"].max()
df[(df["sepal.length"] > 5.5) & (df["variety"] == "Setosa")]
df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")][["sepal.length", "sepal.width"]]
df.groupby(["variety"]).mean()
(df.groupby(["variety"])["petal.length"]).std()