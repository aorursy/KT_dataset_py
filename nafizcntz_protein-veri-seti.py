import pandas as pd

import numpy as np

import seaborn as sns
protein = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")

df = protein.copy()

df.head()
df.info()
df.dtypes
df.isnull().sum()
dfkat = df.select_dtypes(include = ["object"])

dfkat
dfkat.columns
dfkat["structureId"].value_counts()
dfkat["classification"].value_counts()
dfkat["experimentalTechnique"].value_counts()
dfkat["macromoleculeType"].value_counts()
dfkat["crystallizationMethod"].value_counts()
dfkat["pdbxDetails"].value_counts()
dfkat.isnull().sum()
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(

    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(dfkat)
dfkat.isin(["?", "-", ",", "*", "/"]).sum()
dfnum = df.select_dtypes(include = ["int64", "float64"])

dfnum
dfnum.columns
dfnum.describe().T
dfnum.count()
dfnum.isnull().sum()
dfnum.isin(["?", "-", ",", "*", "/"]).sum()
eksik_deger_tablosu(dfnum)
df["publicationYear"] = df['publicationYear'].astype('datetime64[ns]')
df.info()
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import missingno as msno
df1 =  df[(df['classification'] == "HYDROLASE") | (df['classification'] == "TRANSFERASE")

           | (df['classification'] == "OXIDOREDUCTASE") | (df['classification'] == "LYASE") 

           | (df['classification'] == "IMMUNE SYSTEM") | (df['classification'] == "TRANSCRIPTION") 

           | (df['classification'] == "TRANSPORT PROTEIN")]
df1["classification"].value_counts().plot.barh() 

# Bagimli degisken olarak gordugumuz Classification degiskeni ile gorsellestirmeye basliyoruz
sns.barplot( x = "classification", y = df1.classification.index, data = df1 )
df1.groupby(["classification"])["structureMolecularWeight"].std()



# Bu group by isleminin amaci ise bir sonraki grafikte gorecegimiz barplot grafigindeki ince sapmalari belirten cubuklarin daha net anlasilmasi icindir.
plt.figure(figsize=(12,8))

ax = sns.barplot( x = "classification", y = "structureMolecularWeight", data = df1 )



# Classification degiskenine ait siniflarin Molekul agirligina gore sirlamasini gorebiliyoruz

# Yani en fazla molekul agirlina sahip olan sinif OXIDOREDUCTASE olarak gozukmekte

# Bu dagilimda en fazla standart sapmaya sahip sinif ise TRANSPORT PROTEIN olarak gozlenmekte 
df1.groupby(["classification","macromoleculeType"])["structureMolecularWeight"].mean()
plt.figure(figsize=(15,9))

ax = sns.barplot( x = "classification", y = "structureMolecularWeight",hue="macromoleculeType", data = df1 )



# Bir onceki grafikten farkli olarak bu grafikte ele olay ikinci bir kirilimi eklemis olmak

# Protein siniflari X eksinin de iken Y eksenin de proteinlerin molekuler agirliklari var

# Ve ayni zaman da bu siniflarin kendi icerisinde dagilimlari var ve  bu proteinlerin molekul tiplerine gore gozlenmekte

# Simdi Yukari daki GROUP BY islemine baktimiz da aslinda bu dagilimlarin saglamasi da gozukmekte 
(sns

 .FacetGrid(df1,

            hue= "classification",

           height= 7.5,

           xlim= (0, 300000))

 .map(sns.kdeplot, "structureMolecularWeight",shade = True)

 .add_legend())



# Proteinlerin ait olduklari siniflara da molekul tipine gore frekanslarini gorebiliyoruz

# Molekul agirliginin siniflara gore frekanslarinin daha farkli oldugunu anlayabiliyoruz

# Ornegin sinif TRANSCRIPTION olan proteinlerin molekul agirliklari 0-5000 arasinda iken frekans yogunlugu tavan yapmis durumda

# Daha sonra 5000-10000 molekul agirliginda ise IMMUNE SYSTEM sinifina ait proteinler frekans sikligi olarak en fazla olan duruyor

# Bir yerden sonra molekul agirli artan bir trende burundugunde artik tum siniflarin her molekul agirliginda neredeyse ayni frekans

## yogunluguna sahip oldugunu gorebiliyoruz. 
sns.pairplot(df1,kind="reg")



# Bu gorsel ile suna deginmis olduk Scatter Plot bilindigi uzere sacilim ve bize korelasyon hakkinda bilgi vermekte.

# Ayni zamanda degiskenlerin birbiri ile olan iliskisini de gostermek acisindan bir regresyon egrisi ciziyoruz. 

# Ortaya cikan grafikte cok fazla detay oldugu icin bu grafiksel sonuclarin birde istatistiksel kismini yani korelasyon kismini ele alip yorum yapalim.
df1.corr()



# Olayin istatistiksel kismina gecildiginde iki sonuc goze carpmakta 0.349965 ve 0.990821

# Iste bu iki sonuc residueCount - resolution arasinda pozitife yakin bir iliski oldugunu gostermekte.

# Kalintilar ile cozulme arasinda artan bir iliski gorulmekte ama fazla denemez



# Daha sonra structureMolecularWeight - residueCount arasinda yuksek pozitif yonlu bir iiski oldugu gozlenmekte

# Yani molekul agirligi artarken cozulme degeri de artmakta.  
plt.figure(figsize=(12,6))

ax = sns.scatterplot(x = "structureMolecularWeight", y = "residueCount" ,hue="classification",data=df1)



# Bu 0,99 iliskiye bakarken bir de fazladan bilgi almak adina HUE ile Classification yani siniflari da ekliyoruz.

# Molekul siniflarina gore molekul agirli ve kalinti(tortular) arasindaki iliskinin neye gore arttigi hakkinda biraz daha bilgi sahibi olmus olduk.
plt.figure(figsize=(12,6))

ax = sns.scatterplot(x = "resolution", y = "residueCount" ,hue="classification",data=df1)



# Ikinci yuksek sayilabilecek 0,34 iliski incelendigin de ise bunlarin hangi siniflar bazli oldugu ve yayilimda yine hangilerinin etkili oldugunu anlayabiliriz.

# Genel olarak bir yere kadar iki degisken arasindaki iliski birlikte artarken ardindan bu trend durmus.

# Bunun arkasindaki detaylar ise biraz da is bilgisine dayanmaktadir. 

df.isnull().sum()
msno.bar(df) 

#Grafikteki barların yüksekliği total veriye oranla orada olan sayıyı veriyor. Örneğin structerId de boş değer yok diyebiliriz.
msno.heatmap(df)
baz = df1[df1["phValue"]  > 7 ]["phValue"]
notr = df1[df1["phValue"]  == 7 ]["phValue"]
asit = df1[df1["phValue"]  < 7 ]["phValue"]
sozluk = {"Asit": asit,

          "Notr": notr,

          "Baz": baz

         }
phtablo = pd.DataFrame(sozluk)
phtablo.count()
phtablo.count().plot.barh()
from sklearn.model_selection import train_test_split
x = df.drop("classification", axis = 1)

y = df[["classification"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
columns = x_train.select_dtypes(["int64", "float64"])

columns
lower_and_upper = {}

x_train_copy = x_train.copy()

for col in columns.columns:

    q1 = x_train[col].describe()[4]

    q3 = x_train[col].describe()[6]

    iqr = q3 - q1

    lowerbound = q1 - (1.5*iqr)

    upperbound = q3 + (1.5*iqr)

    lower_and_upper[col] = (lowerbound, upperbound)

    x_train_copy.loc[(x_train_copy.loc[:,col] < lowerbound), col] = lowerbound * 0.75

    x_train_copy.loc[(x_train_copy.loc[:,col] > upperbound), col] = upperbound * 1.25
lower_and_upper
x_test_copy = x_test.copy()

for col in columns.columns:

    x_test_copy.loc[(x_test_copy.loc[:,col] < lower_and_upper[col][0]), col] = lower_and_upper[col][0] * 0.75

    x_test_copy.loc[(x_test_copy.loc[:,col] > lower_and_upper[col][1]), col] = lower_and_upper[col][1] * 1.25
sns.boxplot(x_test_copy.residueCount)
sns.boxplot(x_train_copy.residueCount)
sns.boxplot(x_train_copy.resolution)
dfnum = df.select_dtypes(["int64", "float64"])
dfnum.drop(columns=['phValue'], inplace = True)
dfnum.isnull().sum()
(dfnum.isnull().sum(axis = 1) > 3).sum()
dfnum.loc[dfnum.isnull().sum(axis = 1) > 3]
dfnum = dfnum.loc[dfnum.isnull().sum(axis = 1) <= 3]
df.isnull().sum()
df["resolution"].isnull().sum()
df["resolution"].describe()
df['resolution'].fillna(df.resolution.mean(), inplace = True)
df["resolution"].isnull().sum()
df["crystallizationTempK"].isnull().sum()
df["crystallizationTempK"].describe()
df['crystallizationTempK'].fillna(df.crystallizationTempK.mean(), inplace = True)
df["crystallizationTempK"].isnull().sum()
df["crystallizationTempK"].describe()
df["densityMatthews"].isnull().sum()
df["densityMatthews"].describe()
df['densityMatthews'].fillna(df.densityMatthews.mean(), inplace = True)
df["densityMatthews"].isnull().sum()
df["densityMatthews"].describe()
df["densityPercentSol"].describe()
df['densityPercentSol'].fillna(df.densityPercentSol.mean(), inplace = True)
df["densityPercentSol"].isnull().sum()
df["densityPercentSol"].describe()
df.isnull().sum()
df