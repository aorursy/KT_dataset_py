import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

data = pd.read_csv('../input/championsdata.csv')
data.describe() #veri seti hakkında genel bilgiler.
data.info() #veri setinin içeriği,türü,miktarı
data.head() #ilk 5 elemanı
data.tail() #son 5 elemanı
data.shape #satır-sütün sayısı
data.hist(bins=10,figsize=(40,20)) # verilerin grafiğini çizer
data.corr() # veri türlerinin birbiriyle olan ilişkinin büyüklüğü ve yönünü belirler
plt.matshow(data.corr()) #korelasyonun renk grafiğiyle gösterimi
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax) #korelasyonun ısı grafiğiyle gösterimi
plt.show()
# iki veri arasındaki ilişkinin çızgi grafiği ile gösterimi
data.AST.plot(kind='line',color='r',label='Yardım',linewidth=1,alpha=1,grid=True,linestyle='-')
data.TOV.plot(color='b',label='Devir',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x ekseni')
plt.ylabel('y ekseni')
plt.title('Çizgi Gösterimi')
plt.show()
data.isnull().sum() #veri türlerinede bulunan toplam boş veri sayısını gösterir
data.isnull().sum().sum() # tüm verilerde ki toplam boş eleman sayısını verir
data['TPP'] = data['TPP'].fillna('0') #boş verilerin doldurulması
data
sns.boxplot(x=data['PTS']) # verinin türünün kutu gösterimi
P = np.percentile(data.PTS, [10, 100]) #üçdegerlerinin bulunması
P
new_data=data[(data.PTS>P[0]) & (data.PTS<P[1])] # uç değerlerinin çıkarılması
print(data.shape,new_data.shape)
#var olan kolandan yeni kolon oluşturma

def sayi_yüzdesi(df1,df2):
    new_df=df1/df2
    return new_df

data['sayi_yüzdesi']=sayi_yüzdesi(data[['FG','TP','FT']].apply(np.sum,axis=1),data[['FGA','TPA','FTA']].apply(np.sum,axis=1))
data.head()
data=data.drop('Game', axis=1)#"Game" kolonunu çıkarıyoruz
data=data.drop('Year', axis=1)#"Year" kolonunu çıkarıyoruz
data.describe()
data.groupby("Team").size()# veri türlerindeki verilerin dağılımı
data.hist(figsize=(20,10))
data.plot(kind='box', subplots=True,figsize=(30,5),sharex=False,sharey=False)#tüm veri türlerini kutu gösterimi
data.corr()
y=data['Team'].values#sınıflandırma öznitelikleri
x=data.drop('Team',axis=1).values#eğitim öznitelikleri
y
#eğitim ve doğrulama verilerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
from sklearn.naive_bayes import GaussianNB#model oluşturma
model_1 = GaussianNB()
#modelin k-katlamalı çapraz doğrulamsıyla ACC hesaplama
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model_1, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model_1.fit(X_train, y_train)

y_pred = model_1.predict(X_test)
#Sınıflandırıcı tarafından yapılan tahminlerin özeti
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#ACC değrri
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
#verinin normalleştirilmesi
x = data[['PTS']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['PTS2'] = pd.DataFrame(x_scaled)
data.head()
y2=data['Team'].values#sınıflandırma öznitelikleri
x2=data.drop('Team',axis=1).values#eğitim öznitelikleri
#eğitim ve doğrulama verilerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x2, y2, test_size=validation_size, random_state=seed)
from sklearn.naive_bayes import GaussianNB#model oluşturma
model_2 = GaussianNB()
#modelin k-katlamalı çapraz doğrulamsıyla ACC hesaplama
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model_2, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.2, random_state = 0)

model_2.fit(X_train, y_train)

y_pred = model_2.predict(X_test)
#Sınıflandırıcı tarafından yapılan tahminlerin özeti
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#ACC değrri
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))