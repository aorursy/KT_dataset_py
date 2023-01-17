# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#MCBU YAZILIM MÜHENDİSLİĞİ
#YZM 3226 Makine Öğrenmesi Dersi
#Orçun ÖZDİL   172803065,  Ergen Altıparmak    162803058
#datasetimizi yüklüyoruz
df=pd.read_csv('../input/camera_dataset.csv')
df
#dataframemimizin tüm kolonları için count,mean,std,min,25%,50%,75%,max bilgileri
df.describe()
#dataframemimizin tüm kolon isimleri, kayıt sayısı, boş kayıt sayısı ve kayıt değişken tipleri
df.info()
#Kolon adlarıyla beraber ilk 5 kayıt
df.head()
#Kolon adlarıyla beraber son 5 kayıt
df.tail()
#dataframe boyutumuz. 1038 satır 13 kolon
df.shape
#'Max resolution' ve 'Low resolution' kolonları için histogram çiziyoruz
df_hist=df.filter(['Max resolution','Low resolution'])
df_hist.hist(figsize=(150, 150))
#Df içindeki her kolonda yer alan null kayıt sayıları
df.isnull().sum()
#Df içindeki toplam null kayıt sayıları
df.isnull().sum().sum()
#Df içindeki her kolonda yer alan null kayıtların hangi satırlarda olduğu
df[pd.isnull(df).any(axis=1)]
#Tüm null değerler 2 satirda toplandığı için bu değerleri doldurmak yerine satirlari siliyoruz
df=df.dropna()
#Artık null değer yok
df.isnull().sum().sum()
#Korelasyonu metin olarak görüntülüyoruz
df.corr()
#Korelasyonu ısı haritası olarak görüntülüyoruz
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,linewidths=.5, fmt="d")
#Korelasyonu yüksek 2 plot çiziyoruz
import matplotlib.pyplot as plt
plt.scatter(df['Max resolution'],df['Effective pixels'])
plt.xlabel('Max resolution')
plt.ylabel('Effective pixels')
plt.show()
plt.scatter(df['Max resolution'],df['Low resolution'])
plt.xlabel('Max resolution')
plt.ylabel('Low resolution')
plt.show()
#Max resolution kolonu için outlier görüntülüyoruz
sns.boxplot(x=df['Max resolution'])
#ResolutionRange adında yeni bir kolon oluşturuyoruz
df['ResolutionRange']=df['Max resolution']-df['Low resolution']
#Max resolution kolonu için 0-1 arası normalizasyon yapıyoruz
df['Max resolution']=(df['Max resolution']-df['Max resolution'].min())/(df['Max resolution'].max()-df['Max resolution'].min())
df['Max resolution']=(df['Max resolution']*10).round()
df['Max resolution']
#Dimensions kolonu için 0-1 arası normalizasyon yapıyoruz
normalized_df=(df['Dimensions']-df['Dimensions'].min())/(df['Dimensions'].max()-df['Dimensions'].min())
normalized_df
#Df mizi split ediyoruz
from sklearn.model_selection import train_test_split
data_inputs = df[["Low resolution","Effective pixels"]]
expected_output = df[["Max resolution"]]

inputs_train, inputs_test, expected_output_train, expected_output_test   = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)
#RandomForest ile fit ediyoruz
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier ()
rf.fit(inputs_train, expected_output_train)
rf.score(inputs_test, expected_output_test)
#RandomForest için accuracy
accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))
#LinearRegression ile fit ediyoruz
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(inputs_train, expected_output_train)
model.score(inputs_test, expected_output_test)

#LinearRegression için accuracy
accuracy = model.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))

#LinearRegression için Karmaşıklık matrisi, ACC, Precision, Recall değerlerinin görülmesi
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
print ('Report : ')
y_pred = np.ravel(model.predict(inputs_test).astype(int)).tolist()
y_true = np.ravel(expected_output_test.values.astype(int)).tolist()
print (classification_report(y_true, y_pred) )