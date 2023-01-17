pd.read_csv('../input/tabletpc-priceclassification/tablet.csv')
df=pd.read_csv('../input/tabletpc-priceclassification/tablet.csv')
df.head()
df.info()
df.describe().T # sadece sayısal verileri açıklar.
df.tail() #Son 5 gözlem
df.sample(3) # rastgele 3 gözlem
df.shape #2000 gözlem ve 20 değişken
df.isna().sum() #RAM değişkeninde 12 tane eksik gözlem, onkamera değişkeninde 5 tane eksik gözlem var.
df.count() #herbir değişkende kaç adet değer var?
df["DahiliBellek"] #sadece dahili bellek değişkenini gösterelim
df["Renk"].unique() #Renk değerlerini görüntüleyelim
df["RAM"].unique() #RAM değerlerini görüntüleyelim.
df["RAM"].nunique() #1554 tane RAM değeri olduğunu öğrendik
df["ArkaKameraMP"].nunique()  # 12 tane değer var. Sadece sayıyı görüntüledik.
df["BataryaGucu"].mean() # batarya gücü ortalama değer.
df["RAM"].mean() #Ağırlık ortalama değer
df["RAM"].std() #Kalınlık standart sapma
df["RAM"].median() #Batarya ömrü medyan değeri
df["CekirdekSayisi"].describe().T # cekirdek sayisinin temel istatistik değerleri
df.groupby(["RAM"]).mean() #RAM göre diğer değişkenlerin ortalamasını içeren tablo
df.groupby(["CozunurlukYükseklik"]).mean() #Çözünürlük yüksekliğine göre değişkenlerin ortalaması
df["CekirdekSayisi"].mode() #En sık tekrar eden çekirdek sayısı
df.groupby('Agirlik')["Kalinlik"].apply(lambda x: np.mean(x))
df.corr()['Agirlik']['Kalinlik'] #Tahmin ettiğimiz gibi kalınlık ve ağırlık arasında güçsüz bir korelasyon var.
df[(df["4G"] == "Var") & (df["FiyatAraligi"] == "Normal")]
df[(df["Dokunmatik"] == "Var") & (df["FiyatAraligi"] == "Normal")]
df.sort_values('BataryaGucu', axis = 0, ascending = False).head()[["BataryaGucu","BataryaOmru", "FiyatAraligi"]]
df.head()
df["FiyatAraligi"].value_counts().plot.barh(); #Değişkenimiz dengeli dağılmış.
df["Bluetooth"].value_counts().plot.barh(); #Dnegeli
df["4G"].value_counts().plot.barh(); #Dengeli sayılabilir
df["CiftHat"].value_counts().plot.barh(); #Dengeli sayılabilir
sns.catplot(x="FiyatAraligi" , y="BataryaGucu" , data=df);
sns.catplot(x="FiyatAraligi" , y="DahiliBellek" , data=df);
sns.barplot(x="FiyatAraligi" , y="RAM" , hue="Renk" , data=df);
#RAM yaklaşık 2800 den yüksek olan kısımlar hep pahalı olduğunu görüyoruz.
sns.barplot(x="BataryaGucu" , y="Agirlik" , data=df);
sns.distplot(df.RAM ,bins=100, kde=False) #RAM değişkeninin dağılımına bakalım.
sns.distplot(df.RAM ,bins=100) #Yoğunluğa bakalım.
sns.scatterplot(x = "MikroislemciHizi", y = "CekirdekSayisi", hue="Bluetooth",  data = df);
sns.scatterplot(x = "Kalinlik", y = "Agirlik", data = df);
sns.lineplot(x="BataryaGucu", y="BataryaOmru", data=df)
sns.scatterplot(x = "FiyatAraligi", y = "MikroislemciHizi", data = df);
sns.jointplot(x = "BataryaOmru", y = "RAM", data = df, color="yellow");
df.cov() #Her değişkenin kendisi ile ve diğer değişkenlerle olan ilişkisine bakalım
sns.barplot(x ="BataryaGucu" , y = "FiyatAraligi" , data = df);
#1200 değerinden sonraki tabletlerin nereydeyse hep pahalı olduğunu söyleyebiliriz.
#Yaklaşık 1100 değerinden sonra tabletlerin ucuz olmadığını söyleyebiliriz.
# RAM yüksek olduğunda tabletin fiyatınında pahalı olduğunu görüyoruz.
sns.scatterplot(x = "MikroislemciHizi", y = "RAM", hue = "FiyatAraligi",  data = df);
sns.scatterplot(x = "ArkaKameraMP", y = "RAM", hue = "FiyatAraligi",  data = df);
sns.scatterplot(x = "CozunurlukYükseklik", y = "DahiliBellek", hue = "FiyatAraligi",  data = df);
sns.distplot(df["RAM"], bins=16, color="purple");
import pandas as pd               # dataframe manipülasyon işlemleri için kullanacağız.
import numpy as np                # vektörel ve matris işlemleri için kullanacağız.
import seaborn as sns             # görselleştirme yapmak için kullanacağız.
from sklearn import preprocessing   # ön işleme aşamasında label encoding vb. için dahil ettik.
import re                         # regular expression yani düzenli ifadeler kullanmak için dahil ettik. 
df = pd.read_csv('../input/tabletpc-priceclassification/tablet.csv').copy()
df.head()
df.shape #Veri setindeki gözlem ve değişken sayısına bakalım.
df.dtypes #Bu değişkenlerin tipleri nelerdir?
df.info()
df.isnull().sum() #Toplamda 17 tane eksik değerimiz var.
df.describe().T #Tüm sayısal değişkenlerin istatistik değerlerini görelim
corr = df.corr()
corr
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
sns.countplot(df["FiyatAraligi"]);
df.isnull().sum().sum() #Kaç tane eksik gözlem değeri var?
df.isnull().sum() #Bu gözlem değerleri hangi değişkenlere ait ve kaç taneler?
df["RAM"].mean()
df["RAM"]=df["RAM"].fillna(df["RAM"].mean())
df.isnull().sum() #Görüldüğü gibi eksik ram verilerini doldurduk.
df["OnKameraMP"].mean()
df["OnKameraMP"]=df["OnKameraMP"].fillna(df["OnKameraMP"].mean())
df.isnull().sum() #Görüldüğü gibi eksik ram verilerini doldurduk.
df["OnKameraMP"].unique() #Bilinmiyor değeri sorunsuz bir şekilde atanmış mı görelim.
df["RAM"].unique()
#Fiyat aralığı değerini sıraladık.
df["FiyatAraligi"]=pd.Categorical(df["FiyatAraligi"],categories=["Çok Ucuz","Ucuz","Normal","Pahalı"],ordered=True)

df["FiyatAraligi"]
label_encoder = preprocessing.LabelEncoder()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])
df.head()
#YUKARDIDA YAZDIĞIMIZ KOD İLK DEĞERİ 1 YAPAR DİĞER DEĞERİ 0. 
#Bluetooth YOK:1 VAR:2
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat']) #VAR:0 YOK:1
df['4G'] = label_encoder.fit_transform(df['4G']) #VAR:0 YOK:1
df['3G'] = label_encoder.fit_transform(df['3G']) #VAR:1 YOK:0
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik']) #VAR:0 YOK:1
df['WiFi'] = label_encoder.fit_transform(df['WiFi']) #VAR:0 YOK:1
df.head()
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
x= [['Beyaz', 1], ['Pembe', 2], ['Mor', 3],['Turuncu',4],['Gri',5],['Sarı',6],['Mavi',7],
   ['Turkuaz',8],['Kahverengi',9],['Yeşil',10],['Kırmızı',11],['Siyah',12]]
df["Renk"]=encoder.fit_transform(df[["Renk"]]).toarray()
df.head()
df.head()
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
#Gerekli kütüphaneleri yükledik.
y = df['FiyatAraligi']
X = df.drop(['FiyatAraligi'], axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25) 
#Bağımlı ve bağımsız değişkenleri orana göre ayırdık.
nb = GaussianNB()
nb_model = nb.fit(X_train,y_train)
nb_predict = nb_model.predict(X_test)
nb_model.score(X_test,y_test) #Score bakalım.
print(classification_report(y_test,nb_predict))
confusion_matrix(y_test,nb_predict) #Karmaşıklık matrisini görüntüleyelim.
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import confusion_matrix as cm

filterwarnings('ignore')
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy') #Ekip lideri entropy yazarsak daha iyi
cart_model = cart.fit(X_train, y_train)             #sonuç alacağımızı söylemişti.
cart_model
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred) #Karmaşıklık matrisimizi görüntüleyelim.
print(classification_report(y_test, y_pred)) #Değerleri görüntüleyelim
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X)
accuracy_score(y, y_pred)
confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))
score_list = [] 
#Ekip lideri komşu sayısının çok kritik olduğunu söylemişti.
#Range 2 ve 15 yazdık.
for each in range(2,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test, y_test))
    
plt.plot(range(2,15),score_list)
plt.xlabel("k değerleri")
plt.ylabel("doğruluk skoru")
plt.show()
#Komşu sayısının değişimi ile modelin skorunun da değişeceğini ve her komşu sayısına tekabül eden model skorunu bir plot çizdirerek
#görselleştirme yapılması gerektiğini söylemişti.