#İhtiyacımız olan kütüphaneleri aktif ediyoruz

#We are including packages

import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from scipy.stats import shapiro



from warnings import filterwarnings

filterwarnings('ignore')
#Verimizi df değişkenine atadık 

#we will denote with df to our dataset 

df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv").copy()

df.head()

df=df.drop(["id",'Unnamed: 32'],axis=1)#We removed unnecessary variables

#Gereksiz değişkenlerimiz çıkardık
df.info()#Değişkenlerimizin biçimlerinin doğru olduğunu görüyoruz, diagnosis i belki sonradan kategorik yapabiliriz fakat object ten sıkıntı olmaz

df.describe().T#Veri setimizin betimsel istatistikleri 

df.isnull().values.sum()#Veri setimizde eksik veri olmadığını öğreniyoruz

#We haven't any missing value
sns.boxplot(df.radius_mean,df.diagnosis)

#We are seeing outliers
sns.boxplot(df.smoothness_mean,df.diagnosis)#Veri setimizi incelediğimizde aykırı gözlemlerin olduğunu görüyoruz

##Baskılama Yöntemiyle Aykırı değerlerin çözümü

for i in range(1,len(df.columns)):

    Q1=df.iloc[:,i].quantile(0.25)

    Q3=df.iloc[:,i].quantile(0.75)

    IQR=Q3-Q1

    alt_sinir=Q1-1.5*IQR

    ust_sinir=Q3+1.5*IQR

    df.iloc[:,i][(df.iloc[:,i]<alt_sinir)]=alt_sinir

    df.iloc[:,i][(df.iloc[:,i]>ust_sinir)]=ust_sinir

##Yukarıda baskılama yöntemi yaptık yani veri setimizdeki aykırı gözlemleri en yakın olduğu sınır noktasına eşitledik

##Outliers is assigned  to upper and lower limits 
sns.boxplot(df.radius_mean)#Now we can see clean boxplot 
sns.boxplot(df.smoothness_mean)#Gördüğümüz gibi veri setimizde uç noktalar baskılama yöntemiyle temizlendi
df.describe().T #Temizlenen verimizin betimsel istatistiklerine tekrardan bakalım

#Verinin ilk haliyle incelediğimiz standart sapmalarda azalma olduğunu görüyoruz

#Biraz daha normal dağılıma doğru yöneldiği farkediliyor
sns.countplot(df.diagnosis)#Kanser türlerinin veri setindeki sayıları

sns.pairplot(df,hue="diagnosis",kind="scatter")

#  Verimizdeki her değişkenin birbiriyle olan ilişkilerini kategorik değişkenlerimizin kırılımında incelediğimiz zaman

#Melignant grubunun değişkenlerin çoğunda incelendiğinde Bening grubuna göre daha büyük  değerler aldığı görülüyor

#  Değişkenlerin dağılımını incelediğimizde normal dağılım gibi duruyor fakat dağılım grafiğine bakıldığında basıklığının 

#çok düşük olduğu yani sivrilikler gözümüze çarpıyor Shapiro_Wilk testi yapılmasında fayda var.
sns.distplot(df.radius_mean,label="radius")

sns.distplot(df.texture_mean,label="texture")

plt.legend()
shapiro(df.texture_mean)#Ne kadar texture değişkeni normal dağılmış gibi de dursa shapiro testimizin p-value değeri 0.05 den

#küçük olduğunda H0:Veriler normal dağılım gösterir hipotezi red edilir
df.corr()#Veri setimizin koralesyon grafiği

#Correlation graphic
plt.figure(figsize=(20, 20))

sns.heatmap(df.corr(), annot=True)

#we can see better like this
sns.jointplot(x=df.symmetry_mean,y=df.symmetry_worst,kind="reg")
X=df.drop(["diagnosis"],axis=1).copy()

y=pd.DataFrame(df.diagnosis,dtype="category")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

lgbm=RandomForestClassifier().fit(X_train,y_train)

y_pred=lgbm.predict(X_test)

accuracy_score(y_test, y_pred)#Modelimizin doğruluk oranı
cross_val_score(lgbm, X_test, y_test, cv = 10).mean()#Cross valid skorumuz

#Modelimiz için en iyi parametreleri buluyoruz

#We will choose best parameters for our model

rf_params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}

rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2).fit(X_train,y_train)

print("En iyi parametreler: " + str(rf_cv_model.best_params_))
#En iyi parametrelerle oluşan yeni modelimiz

lgbm=RandomForestClassifier(max_depth= 8, max_features= 8, min_samples_split= 5, n_estimators= 1000).fit(X_train,y_train)

y_pred=lgbm.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(lgbm, X_test, y_test, cv = 10).mean()

#Temel Bileşen Analiziyle Modelleme Yapıyoruz
pca=df.drop(["diagnosis"],axis=1).copy()

from sklearn.preprocessing import StandardScaler

pca1=StandardScaler().fit_transform(pca)
from sklearn.decomposition import PCA

pca2=PCA(n_components=2)#Kaç temel bileşene ayrılacağını gösteriyor

pca2_fit=pca2.fit_transform(pca1)

#We choose two components because we can explain easier than more components and we will see that we will take best scores

bilesendf=pd.DataFrame(data=pca2_fit)

pca2.explained_variance_ratio_.cumsum()#Varyansı açıklama oranına bakıldığında 2 bileşenle açıklayabildiğini görüyoruz

#We have sufficient explained variance ratio and this value must be least %66
plt.plot(pca2.explained_variance_ratio_.cumsum())

bilesenpca=pd.concat([bilesendf,df.diagnosis],axis=1).copy()

bilesenpca.columns=["birinci_bilesen","ikinci_bilesen","diagnosis"]

bilesenpca.head()

#Temel bileşen analizi sonucunda oluşan veri setimize bağımlı değişkenimizi ekleyerek üzerinde modelleme yapacaz.

#We added our dependent variable to pca dataset
#LGBM algoritmasını deniyoruz burada

k=[]

l=[]

for i in  range(0,30):

    pca2=PCA(n_components=i)

    pca2_fit=pca2.fit_transform(pca)

    bilesendf=pd.DataFrame(data=pca2_fit)

    X=bilesendf

    y=df.diagnosis

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)

    lgbm_model=LGBMClassifier().fit(X_train,y_train)

    k.append(accuracy_score(y_test,lgbm_model.predict(X_test)))

    l.append(cross_val_score(lgbm_model, X_test, y_test, cv = 10).mean())

k=pd.DataFrame(k)

l=pd.DataFrame(l)

kl=pd.concat([k,l],axis=1)

kl.columns=["accurary","cross"]

print(kl[kl.accurary==max(kl.accurary)].index[0],". components for best accurary score:",kl[kl.accurary==max(kl.accurary)])

print("------------")

print(kl[kl.cross==max(kl.cross)].index[0],". components  for best cross_val_score değeri:",kl[kl.cross==max(kl.cross)])
#RandomForest Algoritmasını deniyoruz

X=bilesendf

y=df.diagnosis

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)

lgbm_model=RandomForestClassifier().fit(X_train,y_train)

accuracy_score(y_test,lgbm_model.predict(X_test))
cross_val_score(lgbm_model, X_test, y_test, cv = 10).mean()
#Accurary score ve cross_val skorlarına baktığımız zaman en iyi sonuçları randomforest algoritması verdi bizde bu algoritmayı seçiyoruz

print(classification_report(y_test, lgbm_model.predict(X_test)))

#Classification report
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

#Kurduğumuz modeldeki tahminlerin kaç tanesinin doğru ve yanlış olduğu gösteriliyor
proba=pd.DataFrame(lgbm_model.predict_proba(X))

proba.columns=["B","M"]

proba.head()#Malignant hastalığı olma olasılığı verilmiştir Yani Tahmin olasılıkları
#Modelimizdeki bilesenlerin önem düzeyleri

Importance = pd.DataFrame({"Importance": lgbm_model.feature_importances_*100},

                         index = bilesenpca.columns[:2])

Importance.sort_values(by="Importance",axis=0,ascending=True).plot(kind="barh",color="red")

plt.xlabel("Değişken Önem Düzeyleri")
sns.pairplot(data=bilesenpca,kind="reg")#Temel bileşen analizi sonucunda ilişkisiz iki matris oluşmuş
sns.distplot(df.texture_mean)