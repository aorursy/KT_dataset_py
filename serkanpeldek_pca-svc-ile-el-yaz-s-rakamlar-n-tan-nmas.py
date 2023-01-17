#Gerekli Kütüphaneler Yükleniyor



import numpy as np # linear algebra

import pandas as pd # Veri işleme



import matplotlib.pyplot as plt



#Görüntü işleme

import cv2



#Makine Öğrenmesi

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

from sklearn.feature_selection import SelectKBest



from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline



#Sistem Kütüphaneleri

import os

import warnings

print(os.listdir("../input"))

#Uyarıları kapatılıyor

warnings.filterwarnings('ignore')

#Eğitim veri seti yükleniyor

train=pd.read_csv("../input/train.csv",dtype="uint8")

train.head()
#Çıktı sütün vektörü elde ediliyor

y=train['label']



#Veri matrisi elde ediliyor

data=train.drop('label',axis=1).values



#Veri matrisinden sadece değerler alınıyor

X=data[:,0:]
print("çıktı y.shape:",y.shape)

print("veri X.shape :",X.shape)
#Eğitim için veri setinin hepsini almak yerine istediğimiz kısmını alıyoruz

#Son olarak veri setinin hepsini alıyoruz.

#Son durum itibariyle veri setinin hepsi alınmıştır.

#Kendiniz çalışırken n_samples değerini değitirip sonuçları gözlemleyebilirsiniz.



n_samples=y.shape[0]#Veri setinin hepsi alınıyor

y=y[:n_samples]

X=X[:n_samples]

print("çıktı y.shape:{}".format(y.shape))

print("veri X.shape:{}".format(X.shape))
def show_digit_matrix(digit, n=10):

    v_images=[]

    n=n

    count=0

    for i in range(0,n):

        h_images=list()

        for j in range(0,n):

            h_images.append(digit[count].reshape(28,28))

            count+=1

        h=np.hstack((h_images))

        v_images.append(h)

    image_matrix=np.vstack((v_images))

    

    fig, axarr = plt.subplots(1, 1, figsize=(12, 12))

    plt.imshow(image_matrix,cmap='gray')

show_digit_matrix(digit=X, n=20)
pca=PCA(n_components=2, whiten=True)

pca.fit(X)

X_pca=pca.transform(X)


plt.figure(1, figsize=(12,8))

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=10,cmap=plt.get_cmap('jet',10))

plt.colorbar()
pca=PCA()

pca.fit(X)
plt.figure(1,figsize=(12,8))

plt.xticks(np.arange(0, 800, 30.0))

plt.plot(pca.explained_variance_,linewidth=2)

X_train, X_test, y_train, y_test=train_test_split(X, #Veri matrisi

                                                  y, #Çıktı vektörü

                                                  stratify=y, #Her sınıftan eşit oranda ayrıştırma için

                                                  test_size=0.3, #%30 test ve %70 eğitim için ayrıştır

                                                  random_state=42 #rasgele sayı çekirdeği

                                                 )

print("Veri seti eğitim ve test olarak ayrıştırıldı")
#Kullanılacak temel bileşen sayısı

n_components=35
pca=PCA(n_components=n_components, whiten=True)

pca.fit(X_train)

X_train_pca=pd.DataFrame(pca.transform(X_train))

X_test_pca=pd.DataFrame(pca.transform(X_test))

print("Eğitim veri seti için PCA dönüşümü gerçekleştirildi")
params=[{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1.0, 10.0], 'kernel': ['rbf']}

       ]

#Çok uzun sürdüğü için GridSearchCV kısmını pasif hale getirdim.

#Siz isterseniz çalışmayı çatallayıp(Fork) deneyebilirsiniz. 

#En iyi sonuç veren parametreler: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

#clf=GridSearchCV(SVC(), params, cv=5)



clf=SVC(C=10, gamma=0.01, kernel="rbf")

clf.fit(X_train_pca, y_train)

y_pred1=clf.predict(X_test_pca)

print("Sınıflandırıcı:{}".format(clf.__class__.__name__))



print("Başarı oranı:{}".format(accuracy_score(y_pred1, y_test)))

print("Karışıklık Matrisi:\n{}".format(confusion_matrix(y_pred1, y_test)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_pred1, y_test)))
competion_dataset=pd.read_csv("../input/test.csv",dtype="uint8")

competion_dataset=competion_dataset.values

competion_dataset=competion_dataset[:,0:]

print("Yarışma competion_dataset.shape :",competion_dataset.shape)
pca=PCA(n_components=n_components, whiten=True)

pca.fit(X)

print("PCA eğitimi için 'eğitim'  veri setinin hepsi kullanıldı")
X_pca=pca.transform(X)

print("Eğitim veri setinin tümünün PCA dönüşümü gerçekleştirildi")
competion_dataset_pca=pca.transform(competion_dataset)

print("Yarışma veri seti için PCA dönüşümü gerçekleştirildi")
clf.fit(X_pca,y)

print("Tüm eğitim veri seti kullanılarak sınıflandırıcı eğitildi")
print("Yarışma gönderisi hazırlanıyor...")

y_pred2=clf.predict(competion_dataset_pca)

print("Yarışma gönderisi hazır.")
file_name="x_pca_{}_svc_mnist.csv".format(n_components)

print(file_name)
results = pd.Series(y_pred2,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv(file_name,index=False)

print("Yarışma gönderisi kaydedildi.")
work_flows=list()

work_flows.append(('pca', PCA(n_components=n_components, whiten=True)))

work_flows.append(('svm',SVC(C=10, gamma=0.01, kernel="rbf")))

clf=Pipeline(work_flows)

print("İş akışı oluşturuldu")
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("İş akışı eğitildi")
print("Sınıflandırıcı:{}".format(clf.__class__.__name__))



print("Başarı oranı:{}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık Matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))
from sklearn.linear_model import LogisticRegression
features = []

features.append(('pca', PCA(n_components=n_components)))

features.append(('select_best', SelectKBest(k=300)))

feature_union = FeatureUnion(features)

#FeatureUnion iyi sonuç vermedi; daha sonra tekrar denenecek 

# create pipeline

work_flows = []

work_flows.append(('pca', PCA(n_components=n_components, whiten=True)))

work_flows.append(('LogReg', LogisticRegression()))

clf = Pipeline(work_flows)

print("İş akışı oluşturuldu")
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

print("İş akışı eğitildi")
print("Sınıflandırıcı:{}".format(clf.__class__.__name__))



print("Başarı oranı:{}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık Matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))
clf.fit(X, y)

print("model eğitildi")
y_pred2=clf.predict(competion_dataset)

print("Yarışma veri setinde tahmin gerçekleştirildi.")
file_name="pipeline_pca_{}_lr_mnist.csv".format(n_components)

print(file_name)
results = pd.Series(y_pred2,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv(file_name,index=False)

print("İkinci yarışma gönderisi kaydedildi.")