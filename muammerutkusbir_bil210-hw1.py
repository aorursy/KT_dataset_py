import numpy as np
import seaborn as sns # kendine özgü kısaltmaları ile kütüphaneleri ekledik
import pandas as pd 
df = pd.read_csv("../input/iris.csv") #veri dosyamızı dataframe kısaltması olan df isimli değişkene taşıdık
df.head() #parantezin içine n = 10 gibi sayılar ekleyerek satır sayısını arttırabiliriz
df.shape #satır sütun sayısı yerine öznitelik ve gözlem sayısı olarak türkçeleştiriyoruz
         #shape komutunda parantez kullanmamaya dikkat ediyoruz
df.info() #boş olmayan veri kolon sayısı dahil temel bilgileri verir
df.describe() #varyans standart sapmanın karekök alınmamış halidir
              #.T ekleyerek satır ile sütunun yer değiştirmesini sağayabiliriz
df.isna().sum()#isna is null olarak açılabilir
               #tekrar sum() ekleyerek toplam eksik değer sayısını bulabiliriz
df.corr() #en güçlü ilişki petal.lenght ile petal.with arasında
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
#renk koyulaştıkça negatif yönde,beyazlaştıkça pozitif yönde korelasyon artar
df["variety"].unique()#dizi şeklinde döndürür
df["variety"].nunique()#number'in 'n'si başa gelir ve bize adet bilgisini verir
sns.scatterplot(x="sepal.width", y="sepal.length", data=df)
#kategorileştirme yapmadan noktalı olarak görselleştirdik
sns.jointplot(x=df["sepal.width"],y=df["sepal.length"],kind="kde",color="blue");
sns.scatterplot(x="sepal.width", y="sepal.length",hue="variety" , data=df)
#virginica türü sepal.lenght özelliği ile ön plana çıkarken setosa türü sepal.with özelliği ile ön planda
df["variety"].value_counts()

sns.violinplot(y="sepal.width",data=df);
#en yaygın 3.0 değerinde gözlemleniyor
sns.distplot(df["sepal.width"],bins=16,color='purple')
#merkezi limit teoremi
sns.violinplot(x="variety",y="sepal.width",data=df);
ax = sns.countplot(x="variety", data=df)
g = sns.jointplot(x="sepal.length", y="sepal.width", data=df)
g = sns.jointplot(x="sepal.length", y="sepal.width",kind = "kde", data=df)
sns.scatterplot(x="petal.length",y="petal.width",data=df)
sns.scatterplot(x="petal.length",y="petal.width",hue = "variety",data=df)
sns.lmplot(x="petal.length",y="petal.width",data=df)
df.corr()["petal.width"]["petal.length"]
df["total.length"]=df["sepal.length"]+df["petal.length"]

df["total.length"].mean()
df["total.length"].std()
df.max()["petal.length"]
#df[ (df["variety"] == "setosa") & ( df["sepal.length" > 5.5] ) ]

df.sort_values('sepal.length', axis = 0, ascending = False).head(1)
#df_filtered = df.query("petal.length<5 & variety==virginica")[["sepal.length","sepal.width"]]

#&(df["sepal.length > 5.5"]
df[(df["variety"] == "setosa") & (df["sepal.length"] > 5.5)]

df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]