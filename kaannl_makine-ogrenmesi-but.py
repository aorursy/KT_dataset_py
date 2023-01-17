import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model
import warnings
#Verilerin Eklenmesi
data = pd.read_csv('../input/master.csv')
#Boş Hücrelerin Bulunması
data.isna().sum()
#Boş Hücrelerin Doldurulması
data['HDI for year'] = data['HDI for year'].fillna(0)
#Sütunların Yeniden Adlandırılması
data.rename(columns={'suicides/100k pop':'suicides_K','HDI for year':'HDI','country-year':'country_year',' gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capita'}, inplace=True)
#Verilerin Şekillendirilmesi
print(data.shape)
print(data.describe())
#Ülkelere göre intihar sayıları
sns.set(context='notebook', style='whitegrid')
pl.figure(figsize =(20,20))
data.groupby(['country']).suicides_no.count().plot('barh')
plt.xlabel('Toplam intihar sayısı', fontsize=12)
plt.ylabel('Ülke', fontsize=12)
plt.title('Ülkelere göre intihar sayıları', fontsize=15)
plt.show()
#Toplam intihar sayısının cinsiyete göre dağılımı
pl.figure(figsize =(15,3))
data.groupby(['sex']).suicides_no.sum().plot('barh')
plt.xlabel('Toplam intihar sayısı', fontsize=12)
plt.ylabel('Cinsiyet', fontsize=12)
plt.title('Cinsiyete göre intihar sayıları', fontsize=15)
plt.show()
#Yaşlara göre inrihar sayıları
pl.figure(figsize =(15,3))
data.groupby(['age']).suicides_no.sum().plot('barh')
plt.xlabel('Toplam intihar sayısı', fontsize=12)
plt.ylabel('Yaş', fontsize=12)
plt.title('Yaşa göre intihar sayıları', fontsize=15)
plt.show()
#Kuşaklara göre intihar sayılarının  dağılımı
pl.figure(figsize =(15,3))
data.groupby(['generation']).suicides_no.count().plot('barh')
plt.xlabel('Toplam intihar sayısı', fontsize=12)
plt.ylabel('Kuşak', fontsize=12)
plt.title('Kuşaklara göre intihar sayıları', fontsize=15)
plt.show()
#Yıllara göre intihar sayıları
pl.figure(figsize =(20,12))
data.groupby(['year']).suicides_no.count().plot('barh')
plt.xlabel('Toplam intihar sayısı', fontsize=12)
plt.ylabel('Yıl', fontsize=12)
plt.title('Yıllara göre intihar sayıları', fontsize=15)
plt.show()
#Verilerin döüştürülmesi
data['generation']=data['generation'].str.replace('Boomers','0')
data['generation']=data['generation'].str.replace('G.I. Generation','3')
data['generation']=data['generation'].str.replace('Generation X','1')
data['generation']=data['generation'].str.replace('Generation Z','2')
data['generation']=data['generation'].str.replace('Millenials','4')
data['generation']=data['generation'].str.replace('Silent','5')
data['gdp_for_year']=data['gdp_for_year'].str.replace(',','')
data['sex']=data['sex'].str.replace('female', '1')
data['sex']=data['sex'].str.replace('male', '0')
pd.to_numeric(data['generation'])
pd.to_numeric(data['sex'])
pd.to_numeric(data['gdp_for_year'])
print(data['generation'][:5])
print(data['sex'][:5])
print(data['gdp_for_year'][:5])
#Sütunların silinmesi
data=data.drop(columns=['country', 'age', 'country_year'])
#Her bir giriş için ölüm oranını sınıflandırma
#Eğer İntihar/100bin kişi değerinin ortalaması
#1 ise yüksek,0 ise düşüktür.
data['fatality_rate']=np.where(data['suicides_K']>data['suicides_K'].mean(), 1, 0)
#Etiketleri ve özellik kümesi sütunlarını ayırma
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['fatality_rate']]
target = 'fatality_rate'

X = data[columns]
y = data[target]
#Verileri eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print("Training FeatureSet:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing FeatureSet:", X_test.shape)
print("Testing Labels:", y_test.shape)
#Modeli bazı parametrelerle başlatıyoruz
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
#Modelin verilere uyumu
model.fit(X_train, y_train)
#Test seti için tahminlerin üretilmesi
predictions = model.predict(X_test)
#Model Doğruluğunun Hesaplanması.
print("Doğruluk oranı:",round(metrics.accuracy_score(y_test, predictions), 2)*100)
#Hataların hesaplanması
print("Ortlama mutlak hata:", round(mean_absolute_error(predictions, y_test), 2)*100)
#Sınıflandırma raporunun hesaplanması
print("Sınıflandırma Raporu:\n", classification_report(y_test, predictions))
#Confusion matrix çizimi
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0','1']])
print(df)