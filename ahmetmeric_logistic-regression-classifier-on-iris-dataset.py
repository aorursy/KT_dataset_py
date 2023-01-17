#kütüphanerleri yüklüyoruz 
#import library
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np 
import os
#Datasetimizi yüklüyoruz
#Load Dataset
data = sns.load_dataset("iris")
data.head() #ilk 5 unsuru göster #show top 5 elements
data['species'].value_counts() #Türlerdeki eleman sayısı  #count of species
sns.countplot(x='species',data=data) #Türlerdeki eleman sayısını grafiklendir # Graph number of elements in species
plt.savefig('tursayisi.png') #ve kaydet  #and save
data.info() # Data hakkında bilgi 
#information about dataset
data.describe() #Datanın istatiksel bilgileri
#Statistical information of data
x = data.iloc[:, :-1] # Son sütun hariç geri kalan sütunlar = x  
# The remaining columns are excluding the last column = x
y = data.iloc[:, -1] # Son sütun yani species(tür) = y
# The last column is the type = y
plt.xlabel('Tür')
plt.ylabel('Özellikler')

pltX = data.loc[:,'species']
pltY = data.loc[:, 'sepal_length']
plt.scatter(pltX, pltY, color='yellow', label='sepal_length',marker='1') #çanak yaprağı uzunluğu

pltX = data.loc[:,'species']
pltY = data.loc[:, 'sepal_length']
plt.scatter(pltX, pltY, color='green', label='sepal_width',marker='.')#çanak yaprağı genişliği

pltX = data.loc[:,'species']
pltY = data.loc[:, 'petal_length']
plt.scatter(pltX, pltY, color='red', label='petal_length',marker='*')#tac yaprağı uzunluğu 

pltX = data.loc[:,'species']
pltY = data.loc[:, 'petal_width']
plt.scatter(pltX, pltY, color='blue', label='petal_width',marker='x')#tac yaprağı genişliği

plt.legend(loc='upper left', prop={'size':8}) 
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40) 
#Datasetimizin yüzde 80'lik kısımı eğitim için ve yüzde 20'lik kısmı ise test amacıyla kullanılmaktadır.
#80 percent of our dataset is used for Train and 20 percent for testing

#random_state komutuda her çalıştırıldığında farklı(randomize) test verileri elde edilmek için kullanılır.
#It is used to obtain different test data each time the random_state command is run.
#Model Eğitimi logistic regrasyon ile sağlanmaktadır.
#Model training is provided with logistic regression.
model = LogisticRegression()
model.fit(x_train, y_train)
#Modelin test edildiği aşama 
#The stage at which the model was tested
predictions = model.predict(x_test)

#Presion,recall, f1-score ve support değerlerinin kontrol edildiği aşama 
#Check precision, recall, f1-score
print( classification_report(y_test, predictions) ) #Sınıflandırmanın doğruluk oranı # accuracy rate of classification
print("accuracy") 
print(accuracy_score(y_test, predictions)) # test verisi üzerindeki doğruluk ağırlıkları #accuracy on test data
#Bir algoritmanın performansının, tipik olarak denetimli bir öğrenme olanının görselleştirilmesine izin veren özel bir tablo düzenidir.
#It is a special table layout that allows the visualization of an algorithm's performance, typically a controlled learning one.
confused = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(4,4))
sns.heatmap(confused, annot=True, linewidths=2, square = True, cmap = 'Reds_r');
plt.ylabel('Doğru olan');
plt.xlabel('Tahmin edilen');
