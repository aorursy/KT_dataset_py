!pip install missingno
import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#Veri setini okutup ilk 5 satırını gözlemliyoruz.
data = pd.read_csv("Hitters.csv")
data.head()
import warnings
warnings.simplefilter(action = "ignore")
#Veri setinini boyutunu inceliyoruz. 322 adet gözlem ve 20 değişken bulunmaktadır.
data.shape
#Veri setine istatiktiksel olarak göz atıyoruz.
data.describe().T
#Kategorik değişken olup olmadığına bakıyoruz. Veri setinde 3 adet kategorik değişken bulunuyor.
data.nunique()
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
categorical_columns
numeric_columns = [c for c in data.columns if c not in categorical_columns]
numeric_columns
#Kategorik değişkenlerin sınıflarından kaçar adet olduğunu inceliyoruz.
print(data["League"].value_counts())
print(data["NewLeague"].value_counts())
print(data["Division"].value_counts())
data.NewLeague.value_counts().plot.barh();
data.Division.value_counts().plot.barh();
data.NewLeague.value_counts().plot.barh();
# Bağımlı değişken olan salary değişkeninin maksimum ve minimum değerlerini buluyoruz.
print("Salary maksimum değer:", data["Salary"].max())
print("Salary minimum değer:", data["Salary"].min())
# Salary değişkeninin histogram ve yoğunluk grafiğini çizdiriyoruz.
sns.distplot(data.Salary);
# Veri setinin korelasyonunu inceleyerek değişkenler arasında ne tür bir ilişki olduğunu anlayabiliriz. 
# Korelasyon değeri > 0 ise pozitif korelasyon bulunmaktadır. Bir değişken değeri artarken diğer değişkenin de değeri artmaktadır.
# Korelasyon = 0 ise değişkenler arasında ilişki yoktur anlamına gelir.
# Korelasyon < 0 ise negatif korelasyon bulunmaktadır. Bir değişken artarken diğer değişken azalmaktadır. 
# Korelasyonlar incelendiğinde salary bağımlı değişkene pozitif korelasyon olarak etkimekte olan 2 değişken bulunmaktadır. 
# Bu değişkenler CRBI ile CRuns değişkenleridir. Bunlar arttıkça Salary (Maaş) değişkeni de artmaktadır.
data.corr()
# Veri setinin korelasyon matrisi grafiğini oluşturuyoruz.
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
#League değişkenine göre groupby yapılıp hangi ligde ortalama ne kadar maaş alındığını buluyoruz.
data.groupby("League").agg({"Salary": "mean"})
#Oyuncunun oynadığı pozisyona göre ortalama ne kadar maaş aldığını buluyoruz.
data.groupby("Division").agg({"Salary": "mean"})
#Veri setinin korelasyonu incelendiğinde en yüksek pozitif yönlü korelasyona CRBI değişkeni sahipti. Bu değişken arttıkça salary değişkenin de artmasını bekleriz.
#CRBI değişkeni kariyeri boyunca yapmış olduğu vuruş sayısıdır. CRBI değişkenine göre gruplayıp salary değişkeninin ortalamasını inceliyoruz.
data.groupby("CRBI").agg({"Salary": "mean"})
#Veri setinde CRBI değişkeninden sonra en yüksek korelasyona CRuns değişkeni sahip. Aynı işlemi onun için de yapıyoruz.
data.groupby("CRuns").agg({"Salary": "mean"})
#Yeni lig ve oyuncunun ligde oynadığı süresine göre gruplanarak oyuncunun aldıkları maaşların ortalamasını buluyoruz.
data.groupby(["NewLeague","Years"]).agg({"Salary": "mean"})
#Kariyeri boyunca isabet sayısı en yüksek olan gözlem birimlerine göre veri setini azalan şekilde sıralayalım. 
data.sort_values("CHits", ascending = False)
#Liglere göre gruplayarak kariyeri boyunca yapılan isabet sayılarının maksimum değerine erişelim.
data.groupby("League").agg({"CHits": "max"})
#Oyuncuların oynadıkları pozisyonlara göre gruplama yaparak kariyeri boyunca isabet sayısının maksimum değerine erişelim.
data.groupby("Division").agg({"CHits": "max"})
#Lig değişkenine göre gruplayarak beyzbol sopası ile yapılan vuruş sayısının maksimum değerlerine erişelim.
data.groupby("League").agg({"AtBat": "max"})
#Lig değişkenine göre gruplayarak liglerde yapılan hataların ortalama değerlerine erişelim.
data.groupby("League").agg({"Errors": "mean"})
#Lig değişkenine göre gruplayarak liglerde yapılan hataların maksimum değerlerine erişelim.
data.groupby("League").agg({"Errors": "max"})
#Oyuncunun ligde oynadığı süresine göre gruplayıp oyuncunun kariyeri boyunca beyzbol sopası ile yapılan vuruş sayısının maksimum değerlerine erişelim.
data.groupby("Years").agg({"CAtBat": "max"})
#Lig değişkenine göre gruplayıp kariyeri boyunca acaba liglerde beyzbol sopası ile ortalama kaç atış gerçekleşmiştir buna erişelim.
data.groupby("League").agg({"CAtBat": "mean"})
data["OrtCAtBat"] = data["CAtBat"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez topa vurduğu
data["OrtCHits"] = data["CHits"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez isabetli vuruş yaptığı
data["OrtCHmRun"] = data["CHmRun"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez en değerli vuruşu yaptığı
data["OrtCruns"] = data["CRuns"] / data["Years"] #Oyuncunun kariyeri boyunca takımına ortalama kaç tane sayı kazandırdığı
data["OrtCRBI"] = data["CRBI"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç tane oyuncuya koşu yaptırdığı
data["OrtCWalks"] = data["CWalks"] / data["Years"] #Oyuncun kariyeri boyunca karşı oyuncuya ortalama kaç kez hata yaptırdığı
data.head()
#Amacımız maaş tahmini yapmak olduğu için maaş değişkeniyle yüksek korelasyon içerisinde olan değişkenlere bakarak tahminleme yapacağız.
data = data.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists','Errors','PutOuts'], axis=1)
data.head()
#Kategorik değişkenlerin sayısal değerlere dönüştürülmesi için Label Encoding ve One Hot Encoding yöntemleri kullanılmaktadır.
#One Hot Encoding yaparak kategorik değişkenleri sayısal değerlere dönüştürüp dummy değişken tuzağından korunalım.
data = pd.get_dummies(data, columns =  ["Division"], drop_first = True)
data = pd.get_dummies(data, columns =  ["League"], drop_first = True)
data = pd.get_dummies(data, columns =  ["NewLeague"], drop_first = True)
data.head()
#Veri setinde kaç adet eksik değer var?
data.isnull().sum()
#Veri setinde kaç adet dolu değer var?
data.notnull().sum()
msno.bar(data);
msno.matrix(data);
#Eksik gözlem olan değerlerin kaç yıllık kariyere sahip olduklarını ve hangi ligde oynadıklarını inceleyelim.
data_eksik = data[data["Salary"].isnull()].head()
data_eksik
#Eksik değerleri veri setinden çıkarıyoruz.
data=data.dropna()
data.shape
sns.boxplot(x = data["Years"]);
Q1 = data["Years"].quantile(0.25)
Q3 = data["Years"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["Years"] > upper) | (data["Years"] < lower)
data["Years"] = data["Years"][~(outliers)]
sns.boxplot(x = data["Years"]);
sns.boxplot(x = data["CAtBat"]);
Q1 = data["CAtBat"].quantile(0.30)
Q3 = data["CAtBat"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CAtBat"] > upper) | (data["CAtBat"] < lower)
data["CAtBat"] = data["CAtBat"][~(outliers)]
sns.boxplot(x = data["CAtBat"]);
sns.boxplot(x = data["CHits"]);
Q1 = data["CHits"].quantile(0.30)
Q3 = data["CHits"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CHits"] > upper) | (data["CHits"] < lower)
data["CHits"] = data["CHits"][~(outliers)]
sns.boxplot(x = data["CHits"]);
sns.boxplot(x = data["CHmRun"]);
Q1 = data["CHmRun"].quantile(0.35)
Q3 = data["CHmRun"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CHmRun"] > upper) | (data["CHmRun"] < lower)
data["CHmRun"] = data["CHmRun"][~(outliers)]
sns.boxplot(x = data["CHmRun"]);
sns.boxplot(x = data["CRuns"]);
Q1 = data["CRuns"].quantile(0.35)
Q3 = data["CRuns"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CRuns"] > upper) | (data["CRuns"] < lower)
data["CRuns"] = data["CRuns"][~(outliers)]
sns.boxplot(x = data["CRuns"]);
sns.boxplot(x = data["CRBI"]);
Q1 = data["CRBI"].quantile(0.30)
Q3 = data["CRBI"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CRBI"] > upper) | (data["CRBI"] < lower)
data["CRBI"] = data["CRBI"][~(outliers)]
sns.boxplot(x = data["CRBI"]);
sns.boxplot(x = data["CWalks"]);
Q1 = data["CWalks"].quantile(0.35)
Q3 = data["CWalks"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CWalks"] > upper) | (data["CWalks"] < lower)
data["CWalks"] = data["CWalks"][~(outliers)]
sns.boxplot(x = data["CWalks"]);
sns.boxplot(x = data["Salary"]);
Q1 = data["Salary"].quantile(0.25)
Q3 = data["Salary"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["Salary"] > upper) | (data["Salary"] < lower)
data["Salary"] = data["Salary"][~(outliers)]
sns.boxplot(x = data["Salary"]);
data=data.dropna()
data.shape
scaler = StandardScaler()
normalizer = preprocessing.Normalizer()
y = data["Salary"]
X = data.drop('Salary', axis=1)
cols = X.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = cols)
X.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Bağımlı ve Bağımsız değişkenler seçilerek X ve y değişkenlerine atanmıştır. X bağımsız değişkeni standardize edilmiştir.
# Veri setinde yer alan değişkenleri standardizasyon işlemi yapılarak model kurulumunda alınan hataların düşürülmesinde etkili bir faktördür. 
# Birden çok standardizasyon yöntemi bulunmaktadır. Bunlar "Normalize", "MinMax" ve "Scale" gibi yöntemlerdir.
# Test ve Train ayırma işlemi gerçekleştirilmiştir. Train setinin % 20'si test setini oluşturmaktadır. Yani train = % 80 ve test = % 20 olarak ayrılmıştır.
X = data[["OrtCHits"]]
y = data[["Salary"]]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
# Train setleri ile basit doğrusal regresyon model kurulum işlemi gerçekleştirilmiştir.
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# Basit Doğrusal Regresyonun model denklemi (formülü) yazdırılmıştır.
# İntercept: Model denkleminde yer alan b0 değeridir. Yükseklikde denilebilir.
# Coef: Model denkleminde yer alan b1, b2 ... bp değerleridir.
print("Formül: "+str(reg_model.intercept_)+" "+str(reg_model.coef_)+"x1")
# Model eğitim hatası hesaplanmıştır.
y_pred = reg_model.predict(X_train)
reg_model_basit_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
reg_model_basit_egitim_hata
# Modelin daha önce görmediği veriler üzerinden test işlemi gerçekleştirilip RMSE Hata Değeri hesaplanmıştır.
y_pred = reg_model.predict(X_test)
reg_model_basit_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
reg_model_basit_test_hata
# Model doğrulama yönetmlerinden K-Fold CV yöntemi kullanılarak model doğrulama işlemi gerçekleştirilmiştir.
from sklearn.model_selection import cross_val_score
print("Model Doğrulama RMSE Hata Değeri:" + str(np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))))
# Bağımlı ve Bağımsız değişkenler belirlenip X ve y değişkenlerine atanmıştır.
# Daha sonra bağımsız değişkenler standardize edilmiştir ancak one hot encoding kısmında oluşan yeni dummy değişkenleri standardize etmemek gerekmektedir.
# Bu nedenle ilk önce bütün bağımsız değişkenler standardize edilip sonrasında dummy değişkenler 1-0 olarak dönüştürülmüştür.
# Standardize işlemi sonucunda dummy değişkenlerde yer alan 0 değerleri negatfi değerlere 1 değerleri pozitif değerlere dönüştürülmüştür.
# Bu nedenle 0'dan küçük olan değerlere 0, büyük olan değerlere 1 değeri bir for döngüsü yazılarak atanmıştır.
y = data["Salary"]
X = data.drop("Salary",axis=1)
X = scaler.fit_transform(X)
for i in range(len(X)):
    if X[i][13]<0:
        X[i][13]=0
    else:
        X[i][13]=1  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
# Ayrılmış olan veri setinin boyutlarına erişim sağlanmıştır.
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Train setleri ile çoklu doğrusal regresyon model kurulum işlemi gerçekleştirilmiştir.
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# Model denkleminde yer alan b0 değeridir. Yükseklikde denilebilir.
reg_model.intercept_
# Model denkleminde yer alan b1, b2 ... bp değerleridir.
reg_model.coef_
# Model eğitim hatası hesaplanmıştır.
y_pred = reg_model.predict(X_train)
reg_model_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
reg_model_egitim_hata
# Modelin daha önce görmediği veriler üzerinden test işlemi gerçekleştirilip RMSE Hata Değeri hesaplanmıştır.
y_pred = reg_model.predict(X_test)
reg_model_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
reg_model_test_hata
# Model doğrulama yönetmlerinden K-Fold CV yöntemi kullanılarak model doğrulama işlemi gerçekleştirilmiştir.
from sklearn.model_selection import cross_val_score
print("Model Doğrulama RMSE Hata Değeri:" + str(np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))))
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=46)
ridge_model = Ridge().fit(X_train, y_train)
ridge_model.coef_
ridge_model.intercept_
# Model eğitim hatası hesaplanmıştır.
y_pred = ridge_model.predict(X_train)
ridge_model_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
ridge_model_egitim_hata
# Modelin daha önce görmediği veriler üzerinden test işlemi gerçekleştirilip RMSE Hata Değeri hesaplanmıştır.
y_pred = ridge_model.predict(X_test)
ridge_model_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
ridge_model_test_hata
# Model doğrulama yönetmlerinden K-Fold CV yöntemi kullanılarak model doğrulama işlemi gerçekleştirilmiştir.
from sklearn.model_selection import cross_val_score
print("Model Doğrulama RMSE Hata Değeri:" + str(np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))))
# Kurulmuş olunan modelin hiperparametre değerlerine erişim sağlanmıştır. 
# Hiperparametre optimizasyonunda alpha değerinin en uygun değeri bulunarak final modeli kurulacaktır.
ridge_model
# Bazı alpha setleri hazırlanılmıştır. Model için denenip en uygunu seçilecektir.
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
alphas4 = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1]
alphas5 = np.arange(100,10000,10)
# Belli alpha setleri denenerek final modeli için en uygun alpha değeri belirlenecektir. Cross Validation için RidgeCV fonk. Kullanılmaktadır.
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas = alphas3, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train, y_train)
# Final modeli için en uygun alpha değeri seçilmiştir.
ridge_cv.alpha_
# Alpha değeri değişkene atanmıştır. Karşılaştırma dataframene eklemek için. 
ridge_alpha = ridge_cv.alpha_
ax = plt.gca()
ax.plot(alphas, coefs) 
ax.set_xscale("log")
# Belirlenen alpha değerine göre final modeli kururlmuştur.
ridge_tuned = Ridge(alpha = ridge_cv.alpha_)
ridge_tuned.fit(X_train, y_train)
#Eğitim Hatası
y_pred = ridge_tuned.predict(X_train)
ridge_final_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
ridge_final_egitim_hata
#Test Hatası
y_pred = ridge_tuned.predict(X_test)
ridge_final__test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
ridge_final__test_hata
# Train setleri ile lasso regresyon model kurulum işlemi gerçekleştirilmiştir.
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
# Model denkleminde yer alan b0 değeridir. Yükseklikde denilebilir.
lasso_model.intercept_
# Model denkleminde yer alan b1, b2 ... bp değerleridir.
lasso_model.coef_
# Model eğitim hatası hesaplanmıştır.
y_pred = lasso_model.predict(X_train)
lasso_model_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
lasso_model_egitim_hata
# Modelin daha önce görmediği veriler üzerinden test işlemi gerçekleştirilip RMSE Hata Değeri hesaplanmıştır.
y_pred = lasso_model.predict(X_test)
lasso_model_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_model_test_hata
# Model doğrulama yönetmlerinden K-Fold CV yöntemi kullanılarak model doğrulama işlemi gerçekleştirilmiştir.
print("Model Doğrulama Hatası:" + str(np.sqrt(np.mean(-cross_val_score(lasso_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))))
# Kurulmuş olunan modelin hiperparametre değerlerine erişim sağlanmıştır. 
# Hiperparametre optimizasyonunda alpha değerinin en uygun değeri bulunarak final modeli kurulacaktır.
from sklearn.linear_model import LassoCV
lasso_model
# Bazı alpha setleri belirlenilmiştir.
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
alphas4 = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
alphas5 = np.arange(100,10000,10) 
# Belli alpha setleri denenerek final modeli için en uygun alpha değeri belirlenecektir. Cross Validation için LassoCV fonk. Kullanılmaktadır.
lasso_cv = LassoCV(alphas = alphas3, cv = 10)
lasso_cv.fit(X_train, y_train)
# Final modeli için gerekli olan alpha belirlenilmiştir.
lasso_cv.alpha_
# Alpha değeri değişkene atanmıştır. Karşılaştırma dataframene eklemek için. 
lasso_alpha = lasso_cv.alpha_
# Belirlenen alpha değeri ile final modeli kurulum işlemi gerçekleştirilmiştir.
lasso_tuned = Lasso(alpha = lasso_cv.alpha_)
lasso_tuned.fit(X_train, y_train)
#Eğitim Hatası
y_pred = lasso_tuned.predict(X_train)
lasso_final_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
lasso_final_egitim_hata
#Test Hatası
y_pred = lasso_tuned.predict(X_test)
lasso_final_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_final_test_hata
# ElasticNet Model kurulum işlemleri gerçekleştirilmiştir.
enet_model = ElasticNet()
enet_model.fit(X_train, y_train)
# Model denkleminde yer alan b0 değeridir. Yükseklikde denilebilir.
enet_model.intercept_
# Model denkleminde yer alan b1, b2 ... bp değerleridir.
enet_model.coef_
y_pred = enet_model.predict(X_train)
enet_model_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
enet_model_egitim_hata
y_pred = enet_model.predict(X_test)
enet_model_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
enet_model_test_hata
# Model doğrulama işlemi gerçekleştirilmiştir.
print("Model Doğrulama RMSE Hata Değeri:" + str(np.sqrt(np.mean(-cross_val_score(enet_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))))
enet_model
# Parametre değerleri belirlenmiştir.
enet_params = {"l1_ratio": [0.1,0.4,0.5,0.6,0.8,1],
              "alpha": np.linspace(0,1,1000)}
# Belirlenen parametrelere göre final modeli kurulacaktır.
enet_model = ElasticNet()
enet_model.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 5).fit(X_train, y_train)
# En iyi parametreler belirlenilmiştir.
gs_cv_enet.best_params_
# Alpha değeri değişkene atanmıştır. Karşılaştırma dataframene eklemek için. 
enet_alpha = 0.6686686686686687
# Belirlenen parametrelere göre final modeli kurulumu yapılmıştır.
enet_tuned = ElasticNet(**gs_cv_enet.best_params_)
enet_tuned.fit(X_train, y_train)
#Eğitim Hatası
y_pred = enet_tuned.predict(X_train)
enet_final_egitim_hata = np.sqrt(mean_squared_error(y_train, y_pred))
enet_final_egitim_hata
#Test Hatası
y_pred = enet_tuned.predict(X_test)
enet_final_test_hata = np.sqrt(mean_squared_error(y_test, y_pred))
enet_final_test_hata
modeller = [
    reg_model,
    ridge_tuned,
    lasso_tuned,
    enet_tuned,]


for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    hata = np.sqrt(mean_squared_error(y_test, y_pred))
    print("-"*28)
    print(isimler + ":" )
    print("Hata:" + str(hata))
sonuc = []

sonuclar = pd.DataFrame(columns= ["Modeller","Hata"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    hata = np.sqrt(mean_squared_error(y_test, y_pred))    
    sonuc = pd.DataFrame([[isimler, hata]], columns= ["Modeller","Hata"])
    sonuclar = sonuclar.append(sonuc)
    
    
sns.barplot(x= 'Hata', y = 'Modeller', data=sonuclar, color="b")
plt.xlabel('Hata')
plt.title('Modellerin Hata Oranları');
sonuclar_data= pd.DataFrame({"Eğitim Hatası":[reg_model_basit_egitim_hata, reg_model_egitim_hata, ridge_model_egitim_hata, lasso_model_egitim_hata, enet_model_egitim_hata],
                              "Test Hatası":[reg_model_basit_test_hata, reg_model_test_hata, ridge_model_test_hata, lasso_model_test_hata, enet_model_test_hata],                              
                              "Test Tuned Hatası":["Yoktur","Yoktur",ridge_final_hata, lasso_final_test_hata,enet_final_test_hata],
                               "Alpha Değerleri": ["Yoktur", "Yoktur", ridge_alpha,lasso_alpha, enet_alpha]})
sonuclar_data.index= ["BASİT_LR_DETAY", "COKLU_LR_DETAY","RIDGE_DETAY","LASSO_DETAY","ENET_DETAY"]
sonuclar_data

