import pandas as pd

import numpy as np

import seaborn as sns
protein = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")
df = protein.copy()

df.head()
df.info()
df.dtypes
df.isnull().sum()
df.columns
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
dfnum["phValue"].isnull().sum()
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
df1 = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")
df2 =  df1[(df1['classification'] == "HYDROLASE") | (df1['classification'] == "TRANSFERASE")

           | (df1['classification'] == "OXIDOREDUCTASE") | (df1['classification'] == "LYASE") 

           | (df1['classification'] == "IMMUNE SYSTEM") | (df1['classification'] == "TRANSCRIPTION") | (df1['classification'] == "TRANSPORT PROTEIN")]
df2["classification"].value_counts().plot.barh()



# Bagimli degisken olarak gordugumuz Classification degiskeni ile gorsellestirmeye basliyoruz
sns.barplot( x = "classification", y = df2.classification.index, data = df2 )
df2.groupby(["classification"])["structureMolecularWeight",].std()



# Bu group by isleminin amaci ise bir sonraki grafikte gorecegimiz barplot grafigindeki ince sapmalari belirten cubuklarin daha net anlasilmasi icindir.
plt.figure(figsize=(12,8))

ax = sns.barplot( x = "classification", y = "structureMolecularWeight", data = df2 )



# Classification degiskenine ait siniflarin Molekul agirligina gore sirlamasini gorebiliyoruz

# Yani en fazla molekul agirlina sahip olan sinif OXIDOREDUCTASE olarak gozukmekte

# Bu dagilimda en fazla standart sapmaya sahip sinif ise TRANSPORT PROTEIN olarak gozlenmekte 
df2.groupby(["classification","macromoleculeType"])["structureMolecularWeight"].mean()
plt.figure(figsize=(15,9))

ax = sns.barplot( x = "classification", y = "structureMolecularWeight",hue="macromoleculeType", data = df2 )



# Bir onceki grafikten farkli olarak bu grafikte ele olay ikinci bir kirilimi eklemis olmak

# Protein siniflari X eksinin de iken Y eksenin de proteinlerin molekuler agirliklari var

# Ve ayni zaman da bu siniflarin kendi icerisinde dagilimlari var ve  bu proteinlerin molekul tiplerine gore gozlenmekte

# Simdi Yukari daki GROUP BY islemine baktimiz da aslinda bu dagilimlarin saglamasi da gozukmekte 
(sns

 .FacetGrid(df2,

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
sns.pairplot(df2,kind="reg")



# Bu gorsel ile suna deginmis olduk Scatter Plot bilindigi uzere sacilim ve bize korelasyon hakkinda bilgi vermekte.

# Ayni zamanda degiskenlerin birbiri ile olan iliskisini de gostermek acisindan bir regresyon egrisi ciziyoruz. 

# Ortaya cikan grafikte cok fazla detay oldugu icin bu grafiksel sonuclarin birde istatistiksel kismini yani korelasyon kismini ele alip yorum yapalim.
df2.corr()



# Olayin istatistiksel kismina gecildiginde iki sonuc goze carpmakta 0.349965 ve 0.990821

# Iste bu iki sonuc residueCount - resolution arasinda pozitife yakin bir iliski oldugunu gostermekte.

# Kalintilar ile cozulme arasinda artan bir iliski gorulmekte ama fazla denemez



# Daha sonra structureMolecularWeight - residueCount arasinda yuksek pozitif yonlu bir iiski oldugu gozlenmekte

# Yani molekul agirligi artarken cozulme degeri de artmakta.  
plt.figure(figsize=(12,6))

ax = sns.scatterplot(x = "structureMolecularWeight", y = "residueCount" ,hue="classification",data=df2)



# Bu 0,99 iliskiye bakarken bir de fazladan bilgi almak adina HUE ile Classification yani siniflari da ekliyoruz.

# Molekul siniflarina gore molekul agirli ve kalinti(tortular) arasindaki iliskinin neye gore arttigi hakkinda biraz daha bilgi sahibi olmus olduk.
plt.figure(figsize=(12,6))

ax = sns.scatterplot(x = "resolution", y = "residueCount" ,hue="classification",data=df2)



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
phtablo["Notr"].value_counts()
phtablo.count()
phtablo.count().plot.barh()
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns
data = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")

data.head(2)
y = data[["classification"]]

y
x = data.drop("classification", axis = 1)

x.head(2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
columns = x_train.select_dtypes(["int64", "float64"])

columns
del columns["phValue"]

del columns["publicationYear"]

columns
columns.columns
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
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
df = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")
df.info()
df.describe().T
dfnum = df.select_dtypes(include = ["int64", "float64"])

#dfnum = dfnum.dropna()
y = dfnum.densityMatthews

X = dfnum.drop('densityMatthews',axis=1)
y
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 123)
sns.boxplot(x = "crystallizationTempK",data = X_train)

sns.boxplot(x ="structureMolecularWeight",data = X_train)
sns.boxplot(x ="residueCount",data = X_train)
Q1 = X_train.quantile(0.25)

Q3 = X_train.quantile(0.75)

IQR = Q3 - Q1 
Q1
Q3
IQR
alt_sinir = Q1 - 1.5 * IQR

ust_sinir = Q3 + 1.5 * IQR

alt_sinir
ust_sinir
residueCount = X_train["residueCount"]
(residueCount < alt_sinir.residueCount) | (residueCount > ust_sinir.residueCount)
RCount_upper = (residueCount > ust_sinir.residueCount)
RCount_lower = (residueCount < alt_sinir.residueCount)
# Ust siniri dolduruyoruz



X_train["residueCount"][RCount_upper] = ust_sinir.residueCount

X_train["residueCount"][RCount_upper]



# Yeni "residueCount" degiskenimiz de ust sinirdaki tum degerleri doldurduk 
# Ayni sekilde ust siniri da baskilama yontemi ile dolduruyoruz



X_train["residueCount"][RCount_lower] = alt_sinir.residueCount

X_train["residueCount"][RCount_lower]



# Alt sinirin altinda deger olmadigi icin bos bir seri goruyoruz 
sns.boxplot(x ="residueCount",data = X_train)
# Bir diger degiskene geciyoruz "crystallizationTempK"



crystallizationKelvin = X_train["crystallizationTempK"]
(crystallizationKelvin < alt_sinir.crystallizationTempK) | (crystallizationKelvin > ust_sinir.crystallizationTempK)
upper = (crystallizationKelvin > ust_sinir.crystallizationTempK)

lower = (crystallizationKelvin < alt_sinir.crystallizationTempK)
X_train["crystallizationTempK"][upper] = ust_sinir.crystallizationTempK

X_train["crystallizationTempK"][upper]
X_train["crystallizationTempK"][lower] = alt_sinir.crystallizationTempK

X_train["crystallizationTempK"][lower]
sns.boxplot(x = "crystallizationTempK",data = X_train)
MolekularWeight = X_train["structureMolecularWeight"]

MolekularWeight
(MolekularWeight < alt_sinir.structureMolecularWeight) | (MolekularWeight > ust_sinir.structureMolecularWeight)
upper = (MolekularWeight > ust_sinir.structureMolecularWeight)

lower = (MolekularWeight < alt_sinir.structureMolecularWeight)
X_train["structureMolecularWeight"][upper] = ust_sinir.structureMolecularWeight

X_train["structureMolecularWeight"][upper]
X_train["structureMolecularWeight"][upper] = alt_sinir.structureMolecularWeight

X_train["structureMolecularWeight"][upper]
sns.boxplot(x = "structureMolecularWeight",data = X_train)
alt_sinir
ust_sinir
# Test kismindaki degsikenler icinde TRAIN de gecerli olan ALT ve UST sinir degerlerini kullanarak 3 adet degiskenin OUTLIER degerlerini dolduralim



X_test
sns.boxplot(x = "crystallizationTempK",data = X_test)
sns.boxplot(x = "structureMolecularWeight",data = X_test)
sns.boxplot(x = "residueCount",data = X_test)
crystallizationKelvin = X_test["crystallizationTempK"]
(crystallizationKelvin < alt_sinir.crystallizationTempK) | (crystallizationKelvin > ust_sinir.crystallizationTempK)
upper = (crystallizationKelvin > ust_sinir.crystallizationTempK)

lower = (crystallizationKelvin < alt_sinir.crystallizationTempK)
X_test["crystallizationTempK"][upper] = ust_sinir.crystallizationTempK

X_test["crystallizationTempK"][upper]
X_test["crystallizationTempK"][lower] = alt_sinir.crystallizationTempK

X_test["crystallizationTempK"][lower]
sns.boxplot(x = "crystallizationTempK",data = X_test)

MolekulAgirligi = X_test["structureMolecularWeight"]
(MolekulAgirligi < alt_sinir.structureMolecularWeight) | (MolekulAgirligi > ust_sinir.structureMolecularWeight)
upper = (MolekulAgirligi > ust_sinir.structureMolecularWeight)

lower = (MolekulAgirligi < alt_sinir.structureMolecularWeight)
X_test["structureMolecularWeight"][upper] = ust_sinir.structureMolecularWeight

X_test["structureMolecularWeight"][upper]
X_test["structureMolecularWeight"][upper] = alt_sinir.structureMolecularWeight

X_test["structureMolecularWeight"][upper]
sns.boxplot(x = "structureMolecularWeight",data = X_test)
residueCount = X_test["residueCount"]
(residueCount < alt_sinir.residueCount) | (residueCount > ust_sinir.residueCount)
upper = (residueCount > ust_sinir.residueCount)

lower = (residueCount < alt_sinir.residueCount)
X_test["residueCount"][upper] = ust_sinir.residueCount

X_test["residueCount"][upper]
X_test["residueCount"][lower] = alt_sinir.residueCount

X_test["residueCount"][lower]
sns.boxplot(x = "residueCount",data = X_test)
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns

pd.options.mode.chained_assignment = None
data = pd.read_csv("../input/protein-data-set/pdb_data_no_dups.csv")

data.head(2)
df = data.select_dtypes(["int64", "float64"])
df.drop(columns=['publicationYear', 'phValue'], inplace = True)
df.isnull().sum()
(df.isnull().sum(axis = 1) > 3).sum()
df.loc[df.isnull().sum(axis = 1) > 3]
df = df.loc[df.isnull().sum(axis = 1) <= 3]
df
df.isnull().sum()
df["resolution"].isnull().sum()
df["resolution"].describe()
df['resolution'].fillna(df.resolution.mean(), inplace = True)
df["resolution"].isnull().sum()
df["resolution"].describe()
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
df["densityPercentSol"].isnull().sum()
df["densityPercentSol"].describe()
df['densityPercentSol'].fillna(df.densityPercentSol.mean(), inplace = True)
df["densityPercentSol"].isnull().sum()
df["densityPercentSol"].describe()
df.isnull().sum()