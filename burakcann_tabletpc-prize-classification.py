

import numpy as np
import seaborn as sns
import pandas as pd
import missingno           

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot 
import os
from subprocess import check_output
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from warnings import filterwarnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn import ensemble
filterwarnings('ignore')

sns.set(rc={'figure.figsize':(10,8)})


df = pd.read_csv('../input/tabletpc-priceclassification/tablet.csv')



df  
df.shape #2000 x 20 lik bir matrisimiz var. 2000 satır ve 20 sütundan oluşuyor.  20 bağımlı ya da bağımsız öznitelikte 2000 tablet cihazı bulunuyor.
df.count() #2000 tane gözlemden 5 tanesinin OnKameraMP, 12 tanesinin RAM öznitelik bilgisi yok.
df.columns 
df.head()
df.dtypes #Bu 20 değişkenin veri tipi

df.info() #Gözlemlerin öznitelikleri genel olarak sayısal tipte. 3 tane veri tipi(int,object,float) bulunuyor. Hiçbiri için null değeri atanmamış.
df["FiyatAraligi"].nunique()
df["FiyatAraligi"].unique() #4 tane fiyat aralığı tipimiz var.
df.FiyatAraligi.value_counts() #Her bir fiyat değerinden toplam 500 gözlem var.
sns.countplot(x = "FiyatAraligi", data = df);

df.hist()#Genel olarak bakıldığında çoğu öznitelik için gözlemler yakın şekilde dağılmış.
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", hue = "FiyatAraligi", data = df); #Örneğin ön kamera ve arka kamera arasındaki fiyat aralığı dağılımına bakalım. Değerler üçgen şeklinde artarak seyrediyor. Arka kamera çözünürlüğü arttıkça ön kamera çözünürlüğü de artıyor. Doğru orantısız değerler de var.

sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik", hue = "FiyatAraligi", data = df); #Çözünürlük genişlik ve yükseklik arasında da genel olarak doğrusal bir ilişki var.
sns.scatterplot(x = "BataryaGucu", y = "BataryaOmru", hue = "FiyatAraligi", data = df); #Batarya gücünün batarya ömrüyle özel bir ilgisi yok. Hepsi değişken değerdeler.
sns.violinplot(x = "RAM", data = df); #Ram değeri fiyatı etkileyen çok önemli bir özellik. Dağılım olarak da düzgüne yakın bir şekilde.
sns.distplot(df["RAM"], bins=32, color="red"); #Bu grafikten de adet olarak RAM için  her değerden birbirine yakın dağılım olduğu net anlaşılıyor.

sns.violinplot(x = "FiyatAraligi", y = "RAM", data = df); #Görüldüğü gibi Ram değeri yüksek olan değerlerin fiyat aralığı buna bağlı olarak değişiyor. Ram değeri ne kadar yüksek olursa fiyat da  artıyor. Buradan genel bir ilişki kurabiliriz.

sns.jointplot(x = df["OnKameraMP"], y = df["ArkaKameraMP"], kind = "kde", color = "brown")

plot.hist(df['RAM'], bins=15)#Ram ile ilgili histogram çizdirdiğimizde;
sns.relplot(x='OnKameraMP', y='ArkaKameraMP', hue='FiyatAraligi', size='FiyatAraligi', col='FiyatAraligi', data=df)
df.isna().sum() #OnKameraMPde 5 RAMde 12 eksik gözlem var.
df.isnull().sum().sum() #Toplam kaç eksik değer olduğunu gözlemliyoruz.
missingno.matrix(df,figsize=(20, 10)); 
missingno.heatmap(df, figsize= (20,10));
def eksik_deger_tablosu(df): 
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return eksik_deger_tablo_son
eksik_deger_tablosu(df) #Görüldüğü üzere eksik veriler gözlemlerin yüzde olarak çok küçük bir kısmını oluşturuyorlar ve aralarında bir ilişki yok. Yüzde olarak 1'in altındalar. Yani bir çıkarımda bulunmak için ciddiye alınacak bir yüzdede değiller. Fakat eksik olmalarının diğer bağımlı öznitelikler ile alakalı mı değil mi bunu irdeleyebiriz.
df.groupby("FiyatAraligi").mean() # Fiyat aralığına göre eksik ram ve on kamera değerlerini en uygun şekilde doldurmak için groupbyı kullanalım.
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean() 
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);  #Grafikte görüldüğü gibi bir alakası var.
df[(df["FiyatAraligi"] == "Ucuz") & (df["OnKameraMP"].isnull())]
df[(df["FiyatAraligi"] == "Normal") & (df["OnKameraMP"].isnull())]
df[(df["FiyatAraligi"] == "Pahalı") & (df["OnKameraMP"].isnull())]
df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())] # Görüldüğü üzere sadece çok ucuz olan fiyat aralığındaki ön kamera bilgileri nan değerinde. Yani eksik verinin olmasının sebebi fiyat aralığının çok ucuz olmasıyla alakalı.
cok_ucuz_OnKameraMP = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index
cok_ucuz_OnKameraMP
df.loc[cok_ucuz_OnKameraMP,"OnKameraMP"] = 4 # Eksik değerleri 4 ile dolduralım. Çünkü fiyat aralıklarına göre çok ucuz ortalama değeri 4 değerine yakın
df.isna().sum()["OnKameraMP"] # Görüldüğü üzere artık hiçbir ön kamera çözünürlüğünde eksik değer yok
df.groupby("FiyatAraligi")[["RAM"]].mean() 
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]); #Görüldüğü üzere pahalı fiyat aralığındaki cihazların RAM bilgisi eksik. Burdan bir çıkarım yapabiliriz.

df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())] #Toplam 12 eksik değer de pahalı olan fiyat aralığında

pahali_RAM = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index
pahali_RAM
df.loc[pahali_RAM,"RAM"] = 3450 # Pahalı fiyat aralığındaki eksik RAM bilgilerini 3450 yaptık çünkü ortalama olarak 3450. 
df.isna().sum()["RAM"] #Şimdi kontrol edelim. Görüldüğü üzere artık hiçbir eksik değer kalmadı.
le = preprocessing.LabelEncoder()
df["Bluetooth"] = le.fit_transform(df["Bluetooth"])
df.head()
#Yok =1, var =0.Burda kütüphane değerlerin numaralandırmasını kendi seçtiği için doğruluk olarak vara 1 yoka 0 veremiyoruz. Zaten yaptığımız işlemlerde onun doğruluğuna göre değil numaralandırmış olarak var yok olduğunu tutuyoruz. Fakat bunun için dummies formüllerini kullanabiliriz.

df["CiftHat"] = le.fit_transform(df["CiftHat"])
df.head() #Yok =1, var =0

df["4G"] = le.fit_transform(df["4G"])
df.head() #Yok =1, var =0
df["Dokunmatik"] = le.fit_transform(df["Dokunmatik"])
df.head() #Yok =1, var =0
df["3G"] = le.fit_transform(df["3G"])
df.head()
df["WiFi"] = le.fit_transform(df["WiFi"])
df.head()
df["Renk_Encoded"] = le.fit_transform(df["Renk"])
df.head() #Renk değeri çok fazla tip içerdiği için onu encoded olarak neye eşit olduğunu anlamak için farklı bir öznitelik tanımladık.
df["FiyatAraligi_Encoded"] =  le.fit_transform(df["FiyatAraligi"])
df.head()
df["Renk"].unique()
df["Renk"].nunique() #12 tane benzersiz renk değerimiz var


df.describe().T
df.groupby(["FiyatAraligi"]).mean()  #Fiyat aralığına göre ortalama değerlerine baktığımızda pahalı değerlerin çoğu öznitelik için daha fazla olduğunu görebiliriz.

df.groupby(["FiyatAraligi"]).std() #Standart sapma değerlerine baktığımızda da genel olarak  fiyat aralıklarının değerleri birbirine yakın değerde. Herhangi bir önçıkarımda bulunmak zor.
df.groupby(["FiyatAraligi"]).max() #Max değerlerine baktığımızda ise  belirleyici öznitelikler dışında değerler birbirine eşit. Veri seti maksimum alabilecekleri değerde.
df.corr() #Korelasyon matrisi 20x20 olmak üzere genel olarak ilişkilendirmelere bakıldığında çoğunun negatif yönde bir ilişkisi olduğu görülüyor. 1'e yakın pozitif yönde ilişkiler çok az.
df.corr()["ArkaKameraMP"]["OnKameraMP"] #En güçlü pozitif ilişki 
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values) #Genel olarak ısı haritasına baktığımızda negatife doğru ilişkilerin çoğunluğunu görüyoruz. Her bir öznitelik arasında neredeyse negatif bir ilişki var.

def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) 
 
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plot.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plot.suptitle('Scatter and Density Plot')
    plot.show()
    
plotScatterMatrix(df,20,10) #Sadece sayısal özniteliklerden ve birden fazla benzersiz değerli özniteliklerden oluşan çekirdek yoğunluğu grafiğimiz. Yoğunluğun bazı yerlerde çok bazı yerlerde az olduğunu gözlemliyoruz. Bu korelasyon matrisine bağlı olarak gelişen pozitif ve negatif ilişkileri gösteren yoğunluk
linear_regresyon = LinearRegression() 

X = df.drop(["FiyatAraligi_Encoded","FiyatAraligi","Renk"], axis = 1) #Renk özniteliğini düşürdük çünkü renk object değer içerdiği için onu float türüne dönüştüremiyorduk. O yüzden numeration yaptığımız encodedı kullandık. 

y = df["FiyatAraligi_Encoded"] 
X# Bağımlı değişkenimizi etkileyen öznitelikler. Bağımsız değişkenler
y #Bağımlı değişkenimiz. Fiyat aralığımız cihazın diğer özelliklerine bağlı olarak değişiyor.

linear_regresyon.fit(X, y)


linear_regresyon.predict([[1233,1,1,1,2,0,50,0.1,146,1,10,499,695,2000,2,0,1,1,0,]]) 
df["model1_tahmin"] = linear_regresyon.predict(X)#Oluşturulan tahmin modeli yaklaşık olarak gerçek değere yakınsıyor.
df
X 

y
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 42)
X_train
X_test
y_train
y_test
NB = GaussianNB()
NB_model = NB.fit(X_train,y_train)
NB_model


dir(NB_model)
X_test[0:20]
NB_model.predict(X_test)[0:20]
y_test[0:20]
y_pred = NB_model.predict(X_test)
y_pred
y_test
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
karmasiklik_matrisi #4 x 4lük bir karmaşıklık matrisimiz var.
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] +  karmasiklik_matrisi[1][0] + karmasiklik_matrisi[0][1])
cross_val_score(NB_model, X_test, y_test, cv = 20)
cross_val_score(NB_model, X_test, y_test, cv = 20).mean() #Ortalamalarını alalım.
print(classification_report(y_test, y_pred))#Tablo 0.71 ile 0.93 arasında yaklaşık olarak değişkenlik gösteriyor.
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore #Kesinlik skoru
RecallScore = recall_score(y_test, y_pred, average='weighted')
RecallScore #Yakamala, hassaslık skoru
F1Score = f1_score(y_test, y_pred, average = 'weighted')  
F1Score #F1 skoru
#cart = DecisionTreeClassifier(random_state = 42)
cart = DecisionTreeClassifier(random_state = 42, criterion='gini')
cart_model_gini = cart.fit(X_train, y_train)
cart_model_gini #Karar ağacı kütüphanesinin parametreleri
y_pred_gini = cart_model_gini.predict(X_test)

accuracy_score(y_test, y_pred_gini)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred_gini)
print(karmasiklik_matrisi)#Değerler sayısal olarak birbiriyle çok dengesiz.
cross_val_score(cart_model_gini, X, y, cv = 20)# 20 tane değer için baktığımızda sayılar birbirine çok yakın mutlak olarak sayıların arasındaki fark birbirine yakın değerlerde
cross_val_score(cart_model_gini, X, y, cv = 20).mean()#Ortalamaları
print(classification_report(y_test, y_pred_gini))#Genel olarak skor değerlerine baktığımızda 0.10-0.90 arasında değişkenlik gösteriyor. 3 için neredeyse skor değerleri aynı.
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')
cart_model = cart.fit(X_train, y_train)
cart_model #Information Gain hesaplamak için criterionu entropy olarak girdik.

df.columns
y_pred2 = cart_model.predict(X_test)
accuracy_score(y_test, y_pred2)#Accuracy score değeri criterion  gini iken 0.816 değerini göstermişti. Fakat entropy için 0.852 çıktı. Yaklaşık 0.036lık bir artış var. Bu değer daha başarılı bir sonuç veriyor.


karmasiklik_matrisi = confusion_matrix(y_test, y_pred2)
print(karmasiklik_matrisi)#Veriler birbirleriyle dengesiz. Fakat simetrik olarak baktığımızda birbirlerine yakın.
cross_val_score(cart_model, X, y, cv = 20)
cross_val_score(cart_model, X, y, cv = 20).mean()#Gini'deki ortalama 0.8440000000000001'di. Aralarında yaklaşık olarak 0.006lik bir fark var.
print(classification_report(y_test, y_pred2)) #Skor değerleri 0.79-0.90 arasında değişken. 

from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
graph = Source(tree.export_graphviz(cart, out_file = None, feature_names = X.columns, filled = True))
display(SVG(graph.pipe(format = 'svg'))) #Görüldüğü üzere karar ağacının kök düğümüne baktığımızda RAM özniteliğine dayanarak karar verdiğini görüyoruz.
ranking = cart.feature_importances_
features = np.argsort(ranking)[::-1][:20]
columns = X.columns

plot.figure(figsize = (16, 9))
plot.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)
plot.bar(range(len(features)), ranking[features], color="lime", align="center")
plot.xticks(range(len(features)), columns[features], rotation=80)
plot.show()#Tabloya baktığımızda anlatılmak istenen aslında fiyat aralığının en çok ram değerine göre değiştiğini gösterir. Fiyat biçilmek istenen yeni bir cihazda en çok ram değerine bakılarak bir fiyat biçilebilir.

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred3 = knn_model.predict(X)
accuracy_score(y, y_pred3)#Tüm öznitelikler için doğruluk skorunu görüntüledik. SAdece test için yapmadık.
karmasiklik_matrisi = confusion_matrix(y, y_pred3)
print(karmasiklik_matrisi) #Matrisin köşegeni 
cross_val_score(knn_model, X_test, y_test, cv = 20)
cross_val_score(knn_model, X_test, y_test, cv = 20).mean()

print(classification_report(y, y_pred3))#Değerler diğer eğitimlerin raporlarına göre daha yüksek oranda. Bu modelin daha başarılı olduğunu söyleyebiliriz.
knn_params = {"n_neighbors": np.arange(2,15)}
knn_params
knn_komsu = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_komsu, knn_params, cv = 3)
knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(9)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test) 
accuracy_score(y_test, y_pred) #Test ve ayarlanmış değerler için  doğruluk skoru. En iyi skor ile aynı değere sahip.
skor_listesi = []

for each in range(2,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    skor_listesi.append(knn2.score(X_test, y_test))
    
plot.plot(range(2,15),skor_listesi)
plot.xlabel("k değerleri")
plot.ylabel("doğruluk skoru")
plot.show()    #Biz en iyi skor değerini arrayde 9 için bulmuştuk. Grafikte de 9'un denk geldiği yer sonucunu bulduğumuz gibi 0.930 ve 0.935 arasında. Yani bizim asıl noktamız 9'a konum olarak yakın bir yerde. En iyi komşu olarak seçmesi ve knn modeline göre algoritmik olarak nokta 9'a yakındır.
cross_val_score(knn_tuned, X_test, y_test, cv = 20)


cross_val_score(knn_tuned, X_test, y_test, cv = 20).mean() #Önceki normal cross val score u normal gözlemler için 0.8870898984268549 iken komşu sayılarıyla birlikte tekrar gözlemlediğimizde 0.9047537160906728 çıktı. Denediğimiz komşu sayıların gözlemleri için daha yüksek.
 