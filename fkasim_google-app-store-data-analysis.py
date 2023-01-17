
import numpy as np                 # linear algebra
import pandas as pd                # data processing
import matplotlib.pyplot as plt    # visualization tool
import seaborn as sns              # visualization tool

import plotly.offline              # visualization tool
import plotly.graph_objs as go


from pylab import rcParams         # figure size in inches


import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("../input/googleplaystore.csv") # read .csv file
dataset.sample(10) 
dataset.info() # information about data set
dataset.describe() # statistical values for numerical columns
dataset.columns  #show features
print('Number of apps in the dataset : ' , len(dataset))
dataset.sample(7)  #give the random 7 sapmle.
print(dataset.shape)
dataset.info()
#missing data
total = dataset.isnull().sum().sort_values(ascending=False)  
percentage = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False) 
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage']) 
missing_data
#Remove missing data
dataset.dropna(how ='any', inplace = True)
total=dataset.isnull().sum().sort_values(ascending=False)
percentage=(dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])
missing_data
print(dataset.shape)
app=dataset.App.unique()
len(app)
#aynı uygulamadan bir kaç tane varsa onları kaldırmak.
dataset.drop_duplicates(subset="App",keep="first",inplace=True) 
#subset=feature ı seçer.Defaultu firsttir().,nplace kalıcı olarak datasete eşitler.
print(dataset.shape)
#Installs kısmını düzenleme: remove "+" and "," and convet to int
dataset.Installs.unique()
#burda sayılar arasındaki virgülü ve sonundaki artyı kaldırma işlemi yapılacak!
dataset["Installs"] = dataset["Installs"].apply(lambda x: x.replace(",","") if "," in str(x) else x) #sayılar arasındaki virgül
dataset["Installs"] = dataset["Installs"].apply(lambda x: x.replace("+","") if "+" in str(x) else x) #sayıların sonundaki +
dataset["Installs"] = dataset["Installs"].apply(lambda x: int(x))                                       #sayıları int'a çevirme
dataset.Installs.unique()

dataset.loc[1:10,["App","Installs"]]
#Şimdi size kısmını düzeltme
dataset.Size.unique()
#burda hepsini mb cinsinden float bir değere cevirmeliyiz
dataset['Size'] = dataset['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

dataset['Size'] = dataset["Size"].apply(lambda x: str(x).replace(",","") if 'M'or "m" in str(x) else x) #burda , leri kaldırma

def mb(x):    #M leri kaldırma fonks.
    if "M"or"m" in str(x):
        x=x.replace("M"or"m","")
        return x
    else:
        return x
dataset["Size"] = dataset["Size"].apply(mb)

def convert_mb(x):    #kb olanları mb a cevirme fonk.
    if "k" in str(x):
        x=x.replace("k","")
        x=float(x)
        x=x/1024
        return x
    else:
        return x
dataset['Size'] = dataset["Size"].apply(convert_mb)

dataset["Size"]=dataset["Size"].astype("float") #Float değerine cevirme fonks.
dataset.loc[1:10,["App","Size"]]
#burda da görüldüğü gibi Nan değerleri yaratmış olduk.
dataset["Size"].value_counts(dropna=False)
#nan olanları fillna ile ortalamaları yazdırma
dataset["Size"].fillna(np.mean(dataset.Size),inplace=True)

dataset["Size"] = dataset["Size"].apply(lambda x: round(x,2))
dataset["Size"].value_counts(dropna=False)
#Şimdi de price kısmını düzeltme de 
print(dataset.dtypes[["Price"]]) #görüldüğü gibi price str gözüküyor.
dataset.Price.unique()
dataset["Price"] = dataset["Price"].apply(lambda x: x.replace("$","") if "$" in str(x) else x)  #$ işaretini kaldırma
dataset["Price"] =dataset["Price"].astype("float")  #ve float a çevirme

dataset.loc[1:5,["App","Price"]]
#Review str bir değer gözüküyor onu int çevirme
dataset.Reviews.unique()
dataset['Reviews'] = dataset['Reviews'].apply(lambda x: int(x))
print(dataset.dtypes[["Reviews"]])
#sütun kaldırma işlemi
dataset.drop(labels=['Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)  
## Cleaning Categories into integers
category_string = dataset["Category"]
category_val = dataset["Category"].unique()
category_valcount = len(category_val)
category_dict = {}
for i in range(0,category_valcount):
    category_dict[category_val[i]] = i
print(category_dict)
dataset["category_int"] = dataset["Category"].map(category_dict).astype(int)
dataset.head()
#Converting type classification into binary
def type_binary(x):
    if x == 'Free':
        return 0
    else:
        return 1

dataset['Type'] = dataset['Type'].map(type_binary)
dataset.head()
dataset['Rating'].describe()
# 2.rating dağılımı
rcParams['figure.figsize'] = 12,8
g = sns.kdeplot(dataset.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)
plt.show()
print('Average app rating = ', np.mean(dataset['Rating']))
#indeksi düzenleme!! unutma
dataset.tail(10)
dataset.shape
"""
R codeları csv yi arff ye cevirme!! Belgelere kaydet csv yi

library("foreign")
data=read.csv("example.csv",header=TRUE)
write.arff(x=data ,file= "file.arff")
"""

index = list(range(1,8191))
dataset.index= index
#dataset.reset_index()   -- bunu da kullanabilirz.
dataset.tail()
#ya da set index iler baska sütunu index yapabilirsin.

paid_apps = dataset[dataset.Price>0]
figure = sns.jointplot("Price","Rating",paid_apps) #compare to price and rating of paid apps

#print('Junk apps priced above 350$')
dataset[['Category', 'App']][dataset.Price > 200]
#1
number_of_apps_in_category=dataset.Category.value_counts().sort_values(ascending=False)
number_of_apps_in_category
#CAtegory sutunundaki olan uygulamaları say ve artan ve ya azalana göre sırala

#buna göre Pie chart yapma
data = [go.Pie(
    labels = number_of_apps_in_category.index,
    values = number_of_apps_in_category.values,
    hoverinfo = 'label+value'
)]

plotly.offline.iplot(data,filename = "Actvie_Category") #buna png olarak erişmemizi saglayacak

subset_df = dataset[dataset.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE',
                                 'LIFESTYLE','BUSINESS'])]
#subset_df ->alt küme bir dataframe olustururuz onuda ana datasetten bu categorydeki uygulamalar alınır
sns.set_style('darkgrid')
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
title = ax.set_title('App pricing trend across categories')
p = sns.stripplot(x="Price", y="Category", data=subset_df, jitter=True, linewidth=1) #burda jitter kümelenme gibi birsey
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
subset_df_price = subset_df[subset_df.Price<100]
p = sns.stripplot(x="Price", y="Category", data=subset_df_price, jitter=True, linewidth=1)
title = ax.set_title('App pricing trend across categories - after filtering for junk apps')

#Bilgi

#ana datasetten sütunları gruplayarak yeni dataset oluşturma.App diye sutun olustur ve o categoride kaç tane var yaz!
new_dataset = dataset.groupby(['Category', 'Type']).agg({'App' : 'count'}).reset_index()
#print(new_dataset)  



rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(dataset.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)
plt.figure(figsize = (10,10))
g = sns.jointplot(x="Reviews", y="Rating",color = 'orange', data=dataset,size = 8);
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=dataset[dataset['Reviews']<1000000]); #regrasyon çizgisi.
plt.title('Rating VS Reveiws',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'teal',data=dataset);
plt.title('Rating VS Installs',size = 20)
f,ax=plt.subplots(figsize=(18,18)) #yani f burda figure dur.bu tamamen çıkan görselin boyutunu belirler.18e 18
sns.heatmap(dataset.corr(),annot=True,linewidth=.8,fmt=".1f",ax=ax) #görsel için seaborn kütüphanesinin heatmap() metodu kullanılır.
#data.corr() tabloyu alır,annot=True korelasyon sayılarının gözükmesi demek,linewidth çizgi kalınlıgı,fmt= virgülden sonraki basamak sayısı
plt.show()
