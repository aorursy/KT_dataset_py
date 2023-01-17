import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iris.csv")
df.head(3)
df.shape # 150 x 5 bir matris var yani 150 adet kayıt var ama aşarıda asıl yöntemi yazdım 
df.axes #5 adet öznitelikten oluşuyor , ki burada da start = 0 ve stop = 150 yazıyor yani 150 kayıt var 
df.info() 

df.groupby(["variety"]).std()
df.groupby(["variety"]).mean()
df.groupby(["variety"]).median()
df.groupby(["variety"]).describe()["sepal.length"] 
df.groupby(["variety"]).describe()["sepal.width"] 
df.groupby(["variety"]).describe()["petal.length"] 
df.groupby(["variety"]).describe()["petal.width"] 
df.isna().sum() # hiç bir özelliğin eksik değeri yok
df.corr() #petal.with ile  petal.length arasındaki ilişki en güçlüdür
corr = df.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values);

df["variety"].value_counts() #benzersiz değerlerin frekanslarını görüntülemek için
df["variety"].nunique() #unique ve benzersiz kaç değer olduğunu öğrenmek için
df["variety"].unique() #Benzersiz değerleri görüntülemek için
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width" , y = "sepal.length", data = df, color = "blue" );
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = "variety" ,data = df);
df["variety"].value_counts() #hepsinden 50şer tane var
sns.violinplot(y = "sepal.width", data = df); 
sns.distplot(df["sepal.width"], bins=16, color="purple");
sns.violinplot(x = "variety", y = "sepal.length",  data = df); 
sns.countplot(x = "variety", data = df);
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], color = "red");
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"],kind = "kde", color = "red");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df); 
sns.scatterplot(x = "petal.length", y = "petal.width", hue = "variety", data = df);
sns.lmplot(x = "petal.length", y = "petal.width", data = df); #dağılımı incelediğimizde doğrumuza yakın noktalarda dağılım göstermişler
df.corr()["petal.length"]["petal.width"]
df['total.length'] = df['petal.length'] + df['sepal.length']
df["total.length"].mean()
df["total.length"].std()
df.sort_values('sepal.length', axis = 0, ascending = False).head(1) #131 numara en büyük değere sahipmiş
df[(df["variety"] == "setosa") & (df["sepal.length"] > 5.5)]
df[(df['petal.length']<5) & (df['variety'] == "virginica")][["sepal.length","sepal.width"]] 
df.groupby(["variety"]).mean()
df.groupby("variety")["petal.length"].std()
