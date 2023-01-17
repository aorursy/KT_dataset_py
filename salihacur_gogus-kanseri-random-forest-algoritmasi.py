# Kütüphaneleri Ekleyelim

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

# Uyarıları Kaldıralım

warnings.filterwarnings("ignore")
# uyarıları filtrele

warnings.filterwarnings("ignore")

# verisetini aldık

data = pd.read_csv('../input/breastcancerwisconsin/breast-cancer-wisconsin.csv',delimiter=',')

data.head()
# ID sütunu gereksiz kaldıralım

data.drop(['id'], inplace=True, axis=1)

data.replace('?', -99999, inplace=True) # ? verisini sayı ile değiştirdik
# Verisetini kopyaladık ve boş veri olan satırları kaldırdık

dataframe = data.copy()

dataframe = dataframe.dropna()

dataframe.head()
# Çıktı verilerinin dağılımını gösterelim

count = data.benormal.value_counts()

count.plot(kind='bar')

plt.legend()
# Son sütundaki 2,4 çıktısını 0,1 yaptık

data['benormal'] = data['benormal'].map(lambda x: 1 if x == 4 else 0)
# Girdi parametreleri

X = data.iloc[:,0:9]

# Çıktı parametresi

y = data.iloc[:,-1]
# Min max normalleştirme

scaler = preprocessing.MinMaxScaler()

X = scaler.fit_transform(X)
# train test verisi ayırma

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Random Forest Modelimizi oluşturalım

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)
# Modeli Eğitelim

history = model.fit(X_train,y_train)
# Ortaya çıkan ağacı görselleştirelim

estimator = model.estimators_[5]
girdi_isimleri = data.columns[0:-1]

cikti_verileri = np.array(['benign','malignant'])

print(girdi_isimleri)

print(cikti_verileri)
data.head()
from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = girdi_isimleri,

                class_names = cikti_verileri,

                rounded = True, proportion = False, 

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
# Tahminler

Y_pred = model.predict(X_test)

Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred]
# Konfüsyon Matris

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, Y_pred)

print(cm)
# Konfüsyon Matris

sns.heatmap(cm,annot=True)
# Model Performans Sonuçları

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

print('Confusion Matrix :')

print(cm) 

print('Accuracy Score :',accuracy_score(y_test, Y_pred))

print('Report : ')

print(classification_report(y_test, Y_pred))