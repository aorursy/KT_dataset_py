# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn  as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#BÖLÜM 1: Veri Keşfi ve Görselleştirme
#Veri seti okunup yazdırılır.
df = pd.read_csv('../input/NBA_player_of_the_week.csv')
df
#Describe ile veri setinin istatistik özetleri bulunur.
df.describe()
#info ile veri setinin bilgileri elde edilir.
df.info()
#head ile parantez içine girilen sayı adedi kadar veri setinin ilk satırları getirilir.
#Parantez içi boş ise ilk 5 satır getirilir.
df.head(10)
#tail ile parantez içine girilen sayı adedi kadar veri setinin son satırları getirilir.
#Parantez içi boş ise son 5 satır getirilir.
df.tail()
#shape ile veri setinin satır ve sütun sayıları elde edilir.
df.shape
#Veri setinin histogram grafiği incelenir. Verilerin yoğunluğuna göre oluşur.
num_bins = 20
df.hist(bins = num_bins, figsize = (20, 15)) #histogramlar mavi
#'Age' özniteliğinin ortalaması bulunur.
df["Age"].mean()
#Tüm sayısal özniteliklerin ortalaması bulunur.
df.mean(axis=0,skipna=True)
#'Age' özniteliğinin medyanı bulunur.
df['Age'].median()
#'Age' özniteliğinin modu bulunur.
df['Age'].mode()
#'Age' özniteliğinin standart sapması bulunur.
df['Age'].std()
#Kovaryans matrisi hesaplanır.
cov = df.cov()
cov
#Korelasyon matrisi hesaplanır.
corr = df.corr()
corr
#Korelasyon Seaborn ısı haritası ile gösterilir.
plt.matshow(corr)
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)
#Korelasyonu yüksek olan 'Seasons in league' ve 'Age' özniteliklerinin plotting(Çizim işlemi) gerçekleştirilmiştir.
df["Seasons in league"].plot(kind='line', color='grey', label='Draft Year', linewidth=2,alpha=0.5, grid=True,linestyle=':')
df["Age"].plot(color = 'red', label = 'Season short', linewidth = 5, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc = 'upper right')
plt.xlabel('Age')
plt.ylabel('Seasons in league')
plt.title('Age - Seasons In League')
plt.show()
#Korelasyonu yüksek olan 'Draft Year' ve 'Season short' özniteliklerinin plotting(Çizim işlemi) gerçekleştirilmiştir.
df.plot(x  = 'Draft Year', y = 'Season short', style = 'o')
plt.title('Draft Year - Season short')
plt.xlabel('Draft Year')
plt.ylabel('Season short')
plt.show()
#BÖLÜM 2: Veri Ön İşleme
#Veri setinde Null olan öznitelikler ve sayısı bulunur.
df.isnull().sum()
#Veri setinde Null olan özniteliklerin toplam sayısı bulunur.
df.isnull().sum().sum()
#Veri setindeki eksik değerler ve bunların yüzdelik değerleri hesaplanarak yazdırılır.
def eksik_deger_tablosu(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat ([mis_val, mis_val_percent], axis = 1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(df)
#Değeri boş olan 'Conference' özniteliği 'Nan' değeri ile doldurulur.
df['Conference'] = df['Conference'].fillna('Nan')
df
#'Age' özniteliğinin uç değerleri bulunur.
#Mavi alanın başladığı değer alt uç değer, bittiği değer üst uç değerdir.
#Alanı ikiye bölen değer ise medyandır.
sns.boxplot(x = df ['Age'])
P = np.percentile(df.Age, [15,40])
P
new_df = df[(df.Age > P[0]) & (df.Age < P[1])]
new_df
#Real_value değeri 1'den küçük olan sporcular 2, 1 olan sporcular 1 ödül almıştır.
#Bu fonksiyonda Real_value'nin aldığı değere göre 2 veya 1 döndürülür.
def number_of_awards(Real_value):
    if(Real_value < 1):
        return 2
    else:
        return 1
#Sporcuların aldıkları ödül sayısını göstermek için 'Number of Awards' özniteliği oluşturulur.
df['Number of Awards'] = df['Real_value'].apply(number_of_awards)
df
#Veri Normalleştirme işlemi gerçekleştirilir.
from sklearn import preprocessing

#'Age' özniteliği normalleştirilir.
x = df[['Age']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanılır.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
#'Age' özniteliğinin normalleştirilmiş değerlerini tutması için 'Normalized Age' özniteliği oluşturulur.
df['Normalized Age'] = pd.DataFrame(x_scaled)

df
#BÖLÜM 3: Model Eğitimi
#'Seasons in league' ve 'Age' öznitelikleri listelenir.
data = {'Seasons in league' : df['Seasons in league'],
        'Age' : df['Age']}

ds = pd.DataFrame(data)
ds
#Eğitim için 'Age' ve 'Seasons in league' özniteliklerinin plotting(Çizim işlemi) gerçekleştirilir.
ds.plot(x = 'Age', y = 'Seasons in league', style = 'o')
plt.title('Seasons in League - Age')
plt.xlabel('Age')
plt.ylabel('Seasons in League')
plt.show()
#iloc integer kullanarak değerlere erişilmesini sağlar.
Y = ds.iloc[:,0].values
X = ds.iloc[:,1:].values
#Y 'Seasons in league' özniteliğinin değerlerini yazdırır.
Y
#X 'Age' özniteliğinin değerlerini yazdırır.
X
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)  
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#LinearRegression ve Naive Bayes modelleri oluşturulur.
model = LinearRegression()
model.fit(X_train, y_train)
model2 = GaussianNB()
model2.fit(X_train, y_train)
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
#intercept ile kesim noktası hesaplanır.
print("Kesim noktası:", model.intercept_) 
print("Eğim:", model.coef_)
X_test
#LinearRegression ile test seti sonuçlarını tahmin eder.
y_pred = model.predict(X_test)
#Naive Bayes ile test seti sonuçlarını tahmin eder.
y2_pred = model2.predict(X_test)
#BÖLÜM 4: Model Sonuçlarının Karşılaştırılması ve Yorumlanması
#LinearRegression ile Gerçek ve Tahmin Edilen değerler yazdırılır.
dm1 = pd.DataFrame({'Gerçek': y_test, 'Tahmin Edilen': y_pred})  
dm1 
#Naive Bayes ile Gerçek ve Tahmin Edilen değerler yazdırılır.
dm2 = pd.DataFrame({'Gerçek': y_test, 'Tahmin Edilen': y2_pred})  
dm2 
#Tahmin edilen değer ile gerçek değerin uyumluluğunun grafiği çizilir.
plt.scatter(X_train, y_train, color = 'darkgrey')
modelin_tahmin_ettigi_y = model.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'red')
plt.title('Seasons in Leauge - Age')
plt.xlabel('Age')
plt.ylabel('Seasons in league')
plt.show()
#Linear Regresyon ile eğitilen modelin tahmin grafiği çizilir
#Burada Gerçek değerler ile tahmin edilen değer arasında mükemmel bir ilişki vardır.
from sklearn.datasets import make_blobs
X1, y1 = make_blobs(100, 2, centers = 2, random_state = 2, cluster_std = 1.5)
plt.scatter(X1[:,0], X1[:,1], c = y1, s = 50, cmap = 'RdBu');
#Naive Bayes ile eğitilen modelin sınıflandırma grafiği çizilir
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE
#Modellerin Değerlendirme Ölçütleri Hesaplanır
for name, model in models:
    model = model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    
    #Accuracy değeri hesaplanır.
    print("%s -> ACC : %%%.2f" % (name, metrics.accuracy_score(y_test, Y_pred) * 100))
    
    #Confusion matrisi hesaplanır.
    print(classification_report(y_test, Y_pred))
    print("Confusion Matrix :\n", confusion_matrix(y_test, Y_pred))
