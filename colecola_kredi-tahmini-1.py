import pandas as pd                   # Verilerin manipülasyonu(ekleme-çıkarma-silme vb.)
import numpy as np                     # Matematiksel Hesaplamalar için
import seaborn as sns                  # Verileri Görselleştirmek için
import matplotlib.pyplot as plt        # Verileri Grafiğe dökmek için kullandığımız python kütüphaneleri
%matplotlib inline
import warnings                        # Uyarıları da yoksayalım.
warnings.filterwarnings("ignore")

#Daha Sonra Eğitim ve Test verilerimizi çalışma ortamımıza dahil ediyoruz.
egitim =pd.read_csv('../input/train_kredi_tahmini.csv')
test = pd.read_csv('../input/test_kredi_tahmini.csv')
#Train ve test verilerinin bir kopyasını yapalım, böylece bu veri setlerinde herhangi bir değişiklik yapsak bile, 
#orjinal veri setlerini kaybetmeyiz.
train_original=egitim.copy()
test_original=test.copy()
egitim.columns
test.columns
# Tüm Değişkenlerin veri Tiplerine bakalım
egitim.dtypes
egitim.shape, test.shape
egitim['Loan_Status'].value_counts()
#Normalize, sayı yerine baskı oranlarını yazdırmak için True olarak ayarlanabilir
egitim['Loan_Status'].value_counts(normalize = True)
egitim['Loan_Status'].value_counts().plot.bar()
plt.figure(1)
plt.subplot(221)
egitim['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Cinsiyet')

plt.subplot(222)
egitim['Married'].value_counts(normalize=True).plot.bar(title = 'Evli')

plt.subplot(223)
egitim['Self_Employed'].value_counts(normalize=True).plot.bar(title='Serbest Çalışan')

plt.subplot(224)
egitim['Credit_History'].value_counts(normalize=True).plot.bar(title = 'Kredi Geçmişi')

plt.show()
plt.figure(1)
plt.subplot(131)
egitim['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Bağımlı Kişiler')

plt.subplot(132)
egitim['Education'].value_counts(normalize=True).plot.bar(title='Eğitim')

plt.subplot(133)
egitim['Property_Area'].value_counts(normalize=True).plot.bar(title='Yaşam Alanı-Bölge')

plt.show()
plt.figure(1)
plt.subplot(121)
sns.distplot(egitim['ApplicantIncome']);

plt.subplot(122)
egitim['ApplicantIncome'].plot.box(figsize=(16,5))

plt.show()
egitim.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")
plt.figure(1)
plt.subplot(121)
sns.distplot(egitim['CoapplicantIncome']);

plt.subplot(122)
egitim['CoapplicantIncome'].plot.box(figsize=(14,5))

plt.show()

