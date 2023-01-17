# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
df=data.copy()
df.head()
df=df.select_dtypes(include=['float64','int64'])
df.head()
df=df.iloc[:,1:]
df.head()
df_table=df["table"].copy()
df.isnull().sum()
sns.boxplot(x=df_table)
Q1=df_table.quantile(0.25)
Q3=df_table.quantile(0.75)
IQR=Q3-Q1

alt_sinir=Q1-1.5*IQR
ust_sinir=Q3+1.5*IQR
print(alt_sinir)
print(ust_sinir)
(df_table<(alt_sinir))|(df_table>(ust_sinir))
#aykiri degerleri true olarak verir tum datadan
#BUNA VEKTOR DUZEYINDE AYKIRI DEGER SORGULAMASI DENIR 
df_table<alt_sinir
aykiri_tf=df_table<alt_sinir
aykiri_tf[0:13]
aykirilar=df_table[aykiri_tf]
aykirilar.index
#Burda aykirilarin kendisi geliyor
#aykiri gozlemlerin indexini yakaladik
import pandas as pd
display(df_table.head())
display(type(df_table))
display(df_table.shape)
#clean_df_table=df_table[~((df_table<(alt_sinir))|(df_table>(ust_sinir))).any(axis=1)]
#temiz_df_table = df_table[~((df_table < (alt_sinir)) | (df_table > (ust_sinir))).any(axis = 1)]
#temiz_df_table.shape
df_table=df["table"].copy()
sns.boxplot(x=df_table)
df_table[aykiri_tf]
df_table.mean()
df_table[aykiri_tf]=df_table.mean()

df_table[aykiri_tf]#tekrar cagirdigimizda ortalamayla dolduruldugunu goruyoruz
aykiri_tf=(df_table<(alt_sinir))|(df_table>(ust_sinir))
df_table[aykiri_tf].head()
#aykiri degerleri tek tarafli bakmistik ,bir de 2 tarafli bakalim
df_table.describe() #onceki describe hali
df_table[aykiri_tf] = df_table.mean() #tum 2 tarafli olan aykiri degerler ortalama ile dolduruldu.
df_table.describe() #sonraki describe hali.std ve mean kuculdu
#bunu silmek istemedigimiz icin yapiyoruz cunku onemli olabilecegini dusunuyoruz.
df_table = df["table"].copy()

aykiri_tf = df_table < (alt_sinir)
df_table[aykiri_tf]
df_table[aykiri_tf] = alt_sinir 
#aykiri olan degerleri alt sinir degerine esitledik.Bunu ust sinir icin de yapabiliriz ayri
df_table[aykiri_tf]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
#Kendimiz suni sekilde bir yapi olusturuyoruz.Grafikle ele alabilmek icin yaptik.
np.random.seed(42) #olusturulacak sayilarin sabit olmasini sagliyoruz.
X_inliers = np.random.normal(70, 3, (100, 2))#normal bir dagilimdan bunu olusturduk.
#ortalamasi 70,std si 3 olan(varyansi) 2 boyutlu veri seti olusturuyoruz

X_inliers = np.r_[X_inliers + 10, X_inliers - 10] 

print(X_inliers.shape)
print(X_inliers[:3,:2])
X_outliers = np.random.uniform(low=15, high=130, size=(20, 2))
#en dusuk 15 en yuksek 130 olan 2 degiskenden olusan outlier olusturuyoruz.
X_outliers
X = np.r_[X_inliers, X_outliers]  #2 suni veriyi birlestiriyoruz.
X[0:3,:]
#LOF SKORLARININ HESAPLANMASI
LOF = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
LOF.fit_predict(X)
X_score = LOF.negative_outlier_factor_  
#butun gozlemleri bu skorlama islemine tabi tutma islemi
X_score[0:3]
X_score.mean()
X_score.std()
np.sort(X_score)[0:10]
plt.hist(X_score, bins = "auto", density = True)
plt.show
#AYKIRI GOZLEMLERIN GORSELLESTIRILMESI
plt.scatter(X[:,0], X[:,1], color = "k", s = 3, label = "Gözlem Birimleri");
radius = radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
plt.scatter(X[:,0], X[:,1], color = "k", s = 3, label = "Gözlem Birimleri");

plt.scatter(X[:, 0], X[:, 1], s = 1000 * radius, edgecolors='r', 
            facecolors='none',label='LOF Skorları') #facecolor halkalarin icini bos birakmak analami

plt.xlim((10,100))
plt.ylim((10,100))

legend = plt.legend(loc = "upper left")

legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [30]  #2 degisken varmis gibi grafik olustu.
#cok degisken oldugunda boyut indirgemesi yapip(2 ye indirip) boyle grafiklendirilebilir
#AYKIRI GOZLEMLERIN YAKALANMASI
X[0:3]
np.sort(X_score)[0:9]
esik_deger = np.sort(X_score)[9]
esik_deger
(X_score > esik_deger)[200:220]
tf_vektor = (X_score > esik_deger)

X[X_score < esik_deger]  #aykiri gozlemlerin kendisine ulasmis oluruz.
# > yaparsak aykiri gozlemlerden arinmis seklini gormus oluruz 
X[~tf_vektor]  #aykiri gozlemlere boyle de erisebiliriz.
X[X_score < esik_deger]
X[200:220] #ornegin bunun icinde bazi aykiri degerleri yakalabiliriz,gorebiliriz.
#belki o degisken tek basina aykiri deger degil ama baska degiskenle beraber 
#degerlendirildiginde aykiri gozlem olmus olabilir
df = X[X_score > esik_deger]

df[0:10]
#AYKIRI GOZLEMLERI ORTALAMA ILE DOLDURMA.
df_X = X.copy()
np.mean(df_X[0])
np.mean(df_X[1])
df_X[~tf_vektor] #aykiri gozlemlerimiz
aykirilar = df_X[~tf_vektor]
aykirilar[:,:1]
aykirilar[:,:1] = np.mean(df_X[0]) #1.degisken icin yaptik
aykirilar[:,1:2] = np.mean(df_X[1]) #2.degisken icin yaptik
aykirilar #cagirdigimiz da 2 si icin de degistigini gormus oluruz
df_X[~tf_vektor] = aykirilar #bu islem ile doldurmus olduk
df_X[~tf_vektor]
#AYKIRI DEGERLERI BASKILAMA ILE DEGISTIRMEK(LOF UN VERDIGI SKORLARA GORE)
df_X = X.copy()

df_X[~tf_vektor]
df_X[X_score == esik_deger]
df_X[~tf_vektor] = df_X[X_score == esik_deger] #aykiri degerleri esik degere esitlemis olduk
df_X[~tf_vektor]
import numpy as np
import pandas as pd

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df
df.isnull().sum()
df.dropna()
df#degismedi inplace ya da bir degiskene atanabilir
dff = df.dropna()
dff.isnull().sum()
df["V1"].mean()
df["V1"].fillna(df["V1"].mean())
df["V1"].fillna(0)
df.apply(lambda x: x.fillna(x.mean()), axis = 0)
import numpy as np
import pandas as pd

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df
df.shape
df.describe()
df.dtypes
df.notnull().sum()  #nan olmayan deger sayilari tum degiskenlerde
df.isnull().sum() #nan sayisi tum degiskenlerde
df.isnull().sum().sum() #tum toplam nan sayisi
df.isnull() #true fals lu gosterir
df[df.isnull().any(axis = 1)] #kendisinde en az 1 eksik veri olan gozlem birimlerini getirdi
df[df.notnull().all(axis = 1)] #tum degerleri tam olan gozlem birimleri
df[df["V1"].notnull() & df["V2"].notnull() & df["V3"].notnull()]
#tum degerleri tam olan gozlem birimleri(baska yontem-amele)
import missingno as msno
msno.bar(df);
df.isnull().sum()
import seaborn as sns
sns.heatmap(df.isnull(), cbar = False);
msno.matrix(df)
#koralasyon heatmap
msno.heatmap(df); 
#kendilerinde eksik olanlarin birbirini etkileme durumlari.Mesela ayni anda veriler bos olabilir,birbiri arasinda koralasyon var mi ona bakiyoruz
null_pattern = (np.random.random(1000).reshape((50, 20)) > 0.5).astype(bool)

null_pattern = pd.DataFrame(null_pattern).replace({False: None})

msno.matrix(null_pattern.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ');
#missigno kutuphanesi ornegi:x ekseni birimler,y ekseni tarihler.Elimizde zaman serisi varsa bu sekilde yapilabilir
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df
df.dropna()
df.dropna(how = "all") #tum satiri NaN olanlari siler.
df.dropna(axis = 1) #sutuna bakar en az 1 eksik veri varsa,sutunu siler.
df["V1"][[3,6]] = 99 #fancy index kullanarak 99 atama yapiyoruz
df.dropna(axis = 1)  #v1 sutununda NaN veri olmadigi icin kaldi
df.dropna(axis = 1, how = "all") #ayni anda bir sutunda hepsi  NaN olanlari silmek icin.
df
df.dropna(axis = 1, how = "all", inplace = True)
df
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df
df["V1"].fillna(0)  # 0 ile doldurduk 
df["V1"].fillna(df["V1"].mean())  #v1 sutunindaki NaN degerleri v1 in ortalamasiyla doldurduk
df.apply(lambda x: x.fillna(x.mean()), axis = 0 ) #tum gozlemleri kendi ortalamalariyla doldurma
df.fillna(df.mean()[:]) #tum gozlemleri kendi ortalamalariyla doldurma(kisa yolu)
df.fillna(df.mean()["V1":"V2"]) #ilk iki degiskeni ortalamayla doldurmak istiyoruz
df.fillna(df.median()["V3"]) #Diyelimki dagilim carpik oldugu icin boyle tercih ettik.
df.where(pd.notna(df), df.mean(), axis = "columns")#pandas in baska bir yolu  ?????
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame(
        {"maas" : V1,
         "V2" : V2,
         "V3" : V3,
        "departman" : V4}        
)

df
df.groupby("departman")["maas"].mean() 
#biz eger maas daki nan degerlere onun ortalamasini atarsak hata yapmis oluruz ,
#kategorik degiskenlerin kirilimina bakmamiz lazim,mesela departman kismi
#siniflarini da gozonunde bulundurup atama yaparsak daha isabet olur.
df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))
#departman degerinde,kiriliminda degerleri basmis olduk,sonuna implace koyarsak kalici hale gelir
import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT",np.NaN,"IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame(
        {"maas" : V1,
         "V2" : V2,
         "V3" : V3,
        "departman" : V4}        
)

df
df.isnull() #eksik bilgi yok gozukuyor ama kucuk harflerle nan yazilmis
df.groupby("departman")["departman"].count() #burada yakalayabiliriz
df.departman.loc[df.departman == "nan"] = "IK" #IK atadik
df
df.departman[0] = df.V3[0]
df
df.groupby("departman")["departman"].count()
df.departman.fillna(df["departman"].mode()) #mode, yani en cok tekrar edeni atamis oldu
#baska bir yontem: kendisinden bir onceki veya sonraki degerlerle de doldurabiliriz.
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame(
        {"maas" : V1,
         "V2" : V2,
         "V3" : V3,
        "departman" : V4}        
)

df
df["maas"].interpolate() #nan degeri kendisinden bir onceki ve bir sonraki degerin ortalamasiyla dolduruyor
df["maas"].fillna(method = "bfill")#bir sonraki degerle dolduruyor
import seaborn as sns
df = sns.load_dataset('planets').copy()
df = df.select_dtypes(include = ['float64', 'int64'])
print(df.isnull().sum())
msno.matrix(df);
!pip install fancyimpute
from fancyimpute import KNN
import pandas as pd
var_names = list(df) #kolon isimleri silinecek ondan sakliyorum isimlerini
knn_imp = KNN(k = 5).fit_transform(df);
knn_imp[0:1] #array bu
dff = pd.DataFrame(knn_imp)# dataframe donusturduk
dff.head()
dff.columns = var_names#kolonlarini yazdik
dff.head()
dff.isnull().sum()
#YCIMPUTE YONTEMI
!pip install ycimpute
from ycimpute.imputer import knnimput
var_names = list(df)
n_df = np.array(df) #nanpy array e donusturduk bu programda kullanicidan istiyor cevirmeyi ama baska programlarda kendisi cevirebiliyor
n_df.shape
dff = knnimput.KNN(k=4).complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.head()
dff.isnull().sum()
import seaborn as sns
df = sns.load_dataset('planets').copy()
df = df.select_dtypes(include = ['float64', 'int64'])
print(df.isnull().sum())
msno.matrix(df);
from ycimpute.imputer import iterforest
var_names = list(df)
n_df = np.array(df)
dff = iterforest.IterImput().complete(n_df)
df.head()
from ycimpute.imputer import EM
var_names = list(df)
n_df = np.array(df)
dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.isnull().sum()
import numpy as np
import pandas as pd

V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)



df = df.astype(float)
df
from sklearn import preprocessing
preprocessing.scale(df) #!!! ORTALAMASI 0 VE STANDART SAPMASI 1 OLACAK SEKILDE STANDARTLASTIRILDI
preprocessing.normalize(df)
scaler = preprocessing.MinMaxScaler(feature_range = (10,20))
scaler.fit_transform(df)
binarizer = preprocessing.Binarizer(threshold = 5).fit(df) #esik deger 5 girildi
#fit_transform da olur fit de olur
binarizer.transform(df)
import seaborn as sns
tips = sns.load_dataset('tips')
df = tips.copy()
df_l = df.copy()
df_l.head()
df_l["yeni_sex"] = df_l["sex"].cat.codes #yeni degisken ekleyip 0 ve 1 lerden olusturuyor
df.head()
df["day"].str.contains("Sun")
import numpy as np 
df["yeni_day"] = np.where(df["day"].str.contains("Sun"), 1, 0)
df
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit_transform(df["day"])
#cok dikkat
df.head()
df_one_hot = pd.get_dummies(df, columns = ["sex"], prefix = ["sex"])
df_one_hot.head()
pd.get_dummies(df, columns = ["day"], prefix = ["day"]).head()


