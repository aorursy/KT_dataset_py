import numpy as np 

import pandas as pd



from sklearn import tree 

from sklearn import metrics

from sklearn import datasets

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split



from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



import graphviz #Karar ağacını çizdirme



import warnings

warnings.filterwarnings("ignore")
iris = datasets.load_iris()  #Örnek datasetlerden birini yüklüyoruz.

# alınan dataset bölünüyor.

X= iris.data

y= iris.target



print("Sınıf Etiketleri:", np.unique(y))
ss = StandardScaler()

ss.fit(X)

X= ss.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
dtree = tree.DecisionTreeClassifier(criterion="entropy",random_state=0)

dtree.fit(X_train,y_train)

# Alınan sonuçları metrics yardımı ile ekranda gösteriyoruz.

# Eğitim verileri için

print("Eğitim - Doğruluk Oranı\n", metrics.accuracy_score(y_train,dtree.predict(X_train)))

print("Eğitim - Karmaşıklık Matrisi\n", metrics.confusion_matrix(y_train,dtree.predict(X_train)))

print("Eğitim - Sınıflandırma Raporu\n", metrics.classification_report(y_train,dtree.predict(X_train)))

print("\n")

# Test verileri için

print("Test - Doğruluk Oranı\n", metrics.accuracy_score(y_test,dtree.predict(X_test)))

print("Test - Karmaşıklık Matrisi\n", metrics.confusion_matrix(y_test,dtree.predict(X_test)))

print("Test - Sınıflandırma Raporu\n", metrics.classification_report(y_test,dtree.predict(X_test)))
#sklearn kütüphanesi yardımıyla MSE MAE RMSE değerleri hesaplandı. 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dtree.predict(X_test)))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, dtree.predict(X_test)))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dtree.predict(X_test))))
dot_data = tree.export_graphviz(dtree,out_file=None,

                                feature_names=iris.feature_names,

                                class_names=iris.target_names,

                                filled= True,

                                rounded=True,

                                special_characters=True)



graph = graphviz.Source(dot_data)

graph.render("iris") # dosya oluşturma 

graph #ekrana grafiği yazdırma
