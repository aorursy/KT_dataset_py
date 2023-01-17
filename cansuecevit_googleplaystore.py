import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
#Kullanacağımız data setinin okunması.
google_play = pd.read_csv('../input/googleplaystore.csv') 
#Tablonun istatistik değerlerini veren 'describe' kullanımı.
google_play.describe()
#Tablonun genel bilgilerini veren 'info' kullanımı.
google_play.info()
#Tablonun ilk beş satırını veren 'head' kullanımı.
google_play.head()
#Tablonun son beş satırını veren 'tail' kullanımı.
google_play.tail()
#Tablonun satır ve sütun sayısını veren 'shape' kullanımı.
google_play.shape
#Histogram için rating ve reviews atamaları.
google_play['Rating'] = google_play['Rating'].apply(pd.to_numeric, errors='coerce')
google_play['Reviews'] = google_play['Reviews'].apply(pd.to_numeric, errors='coerce')
#Histogram oluşturma.
google_play["Reviews"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Reviews",figsize=(10,10))
plt.legend()
plt.xlabel("Reviews")
plt.title("Reviews Distribution")
plt.show()
google_play['Type'].value_counts()
#Korelasyon matrisi oluşturma.
plt.figure(figsize=(12,10))
sns.heatmap(google_play.corr(), annot = True, cmap = 'Reds')
plt.show()
#Installs değerlerini int değeri yapma.
def get_installs(installs):
    installs = re.sub(',', '', installs)
    if installs.endswith('+'):
        installs = installs[:-1]
        return int(installs)
    else:
        return 0
google_play['Installs'] = google_play['Installs'].apply(get_installs)
# Ücretsiz ve ücretli Apps sayısı veren Plotting
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
sns.countplot(x='Type',data=google_play)
plt.title("Mevcut Uygulama Sayısı: Ücretsiz v / s Ücretli")
# Kategoriye göre en çok yüklenmiş uygulamaları veren Plotting
plt.subplot(1,2,2)
sns.barplot(x='Type',y='Installs',data=google_play,ci=None)
plt.title("Yüklenen Uygulama Sayısı: Ücretsiz v / s Ücretli")
plt.tight_layout()

# Eksik değerler olup olmadığını kontrol etme
sns.heatmap(google_play.isna())
#Eksik verileri düzeltme.
google_play=google_play.dropna()
google_play.shape
#Mevcut özniteliklerden yeni bir öznitelik oluşturma.
paid=google_play[google_play.Type=='Paid']
paid1=paid.copy()
paid1["Toplam Tutar"]=paid1.Installs*paid1.Price
paid1.head()