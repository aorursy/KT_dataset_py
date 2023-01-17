import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iriscsv/iris.csv")
df
df.head(5)
df.shape
print("öznitelik : ", df.shape[0]) # veri matrisinin satırlarını almış olduk
print("gözlem : ", df.shape[1])    # veri matrisinin sütunlarını almış olduk
df.info()
# veri türlerimiz hakkında bilgileri almış olduk
df.describe()

# verilerin standart sapmasına bakıldığında speal.lendth için pozitif güçlü bir bağlantı var 
# verilerimiz ortalmaa değeri min ve max değer arasında ve çok fark yok bu verilerimizin çok dağınık olmadığını gösterir
df.isna().sum()
#bütün değerler tam girilmiş
sepal = df.corr()["sepal.length"]["sepal.width"]
print("sepal : ", sepal)
petal = df.corr()["petal.length"]["petal.width"]
print("petal : ", petal)

# sepal için zayıf  negatif bir ilişki olduğu söylenebilir 
# petal için güçlü pozitif bir ilişki olduğu söylenebilir
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values);

# haritadan verilerimizden pozitif güçlü ve negatif güçlü verilerimizi çok daha rahat gözlemleyebiliyoruz
df["variety"].unique()

#burada variety kolonundaki bütün değişkenlerimizi görmüş oluyoruz
#not bunlardan başka değişken yoktur
#tabiki bunlardan birden fazla defa kullanılmış olabilir
dflen = df["variety"].unique()
len(dflen)

# bir önceki denemede zaten değişkenlerimizi görmüştük
# burdada kaç adet olduklarını sayısal olarak göstermiş olduk
sns.scatterplot(x = "sepal.length", y = "sepal.width",data=df, color="blue")

# verilerin üst ve altta iki farklı çizelgede olduğunu net bir şekilde görebiliriz
sns.jointplot(x = "sepal.width", y = "sepal.length", data=df, color="red")

# yoğunlukların nerelerde olduğunu daha net görmüş olduk
sns.scatterplot(x = "sepal.length", y = "sepal.width",hue="variety",data=df, color="blue")

# burada bu üç değişkenle göstermeyi başardık ve çok daha okunabilir oldu
# örneğin versicolor sepal.widht olarak üst değerinin ortalamadan yüksek olduğu ve düzgün dağılmadığını net bir şekilde görebiliriz
sl = df['sepal.length'].value_counts(normalize=False)
sw = df['sepal.width'].value_counts(normalize=False)
pl = df['petal.length'].value_counts(normalize=False)
pw = df['petal.width'].value_counts(normalize=False)
print(sl, sw, pl, pw)

sns.violinplot(y="sepal.width", data=df, color="aqua")

## verilerimizin 30 genişliğindeki yoğunluğu çok açık bir şekilde gözlemlenebiliyor
## bu durumda bunun doğal ve düzgün bir dağılım olmadığını söylemek yanlış olmaz
sns.distplot(df['sepal.width'], bins=16, color="red")

### buradan da görüldüğü gibi peek noktası çok yüksek veri sürekliliğini bozuyor
# verilerimizin düzgün dağılmadığını ortalama değerin min ve max değere uzaklığının fazla olduğunu söyleyebiliriz
sns.violinplot(x = "sepal.length", y = "variety", data=df, )

# verilerimizden setosa'nın en az düzensizliğe
# virginica'nin ise en farklı ve yayılımlı verilere sahip olduğunu söyleyebiliriz
# burada unutulmamalıdır ki sepal.lengt özelliğ en çok değişkenlik gösteren virginica dır
sns.countplot(x="variety", data=df)

# verilerimiz dengeli olduğunu görebiliyoruz
sns.jointplot(x= "sepal.length", y="sepal.width", data=df)

# dağılımımızda frenkansın yüksek oldığı bölgeler
# sepal.length 3.0-3.5 sepal.width 4.00-5.50 arasındaki değerler
# sepal.length 2.5-3.5 sepal.width 5.50-7.00 arasındaki değerler

sns.jointplot(x= "sepal.length", y="sepal.width", kind="kde", data=df)

# yoğunluk bir öncekine göre çok daha rahat gözüküyor
# yoğunluk bölgsindeki değerlerin yine benzer aralık olduğunu göstermiş olduk
sns.scatterplot(x="petal.length", y="petal.width", data=df)

# verilerin ortalamadan nereden ayrılık gösterdiğini daha net görebiliriz

sns.scatterplot(x="petal.length", y="petal.width", hue="variety", data= df)

# ortalamadan verilerin hangi türde daha çok ayrıldığını görebiliyoruz
# mesela virginica
sns.lmplot(x = "petal.length", y = "petal.width", data = df);

# burada ilişkinin güçlü olduğu arakıklardan bahsedilebilir
# mesela alt kısımda ilişki yüksektir ama alt tarafa gelindiğinde çizgiden sapan veriler ilişkiyi negatif etkiliyor
df.corr()['petal.length']['petal.width']

# burada positif yönlü güçlü bir ilişki vardır
# verilerimiz ortalamadan az sapmıştır diyebiliriz
# grafikten de bu görülebilir
df['petal.length'].add(df['petal.width'])
df['petal.length'].add(df['petal.width']).mean()
df['petal.length'].add(df['petal.width']).std()

# standart sapma değerimizin bize iki verinin çok ortalamadan çok uzak olmadığını ama tam bir güven de vermediğini söyleyebliriz
df['sepal.length'].max()

# maximum değerimizin yanına min değeri ve ortalama değeri de yazıdırarak veri düzenini görebiliriz
df[(df['sepal.length']>5) & (df['variety'] == "Setosa")]
df[(df['petal.length']<5) & (df['variety']=="Virginica")]

# virginica için petal.length değerlerinden 5 ten büyük olanlar
df.groupby(['variety']).mean()
df.groupby(['variety'])['petal.length'].std()

# burada her tür için farklı standart sapmalar bulmuş olduk .
# türlerden en güven vereni (ortalamaya en yakın değerler taşıyan) setosa dır