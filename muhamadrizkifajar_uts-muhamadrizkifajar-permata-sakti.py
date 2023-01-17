#Nama : Muhamad Rizki Fajar (PERMATA-SAKTI)
#NIM : 2018310043 / 20.01.53.9018
#ASAL PTS : UNIVERSITAS BINA INSANI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import tree

dataset = pd.read_csv('../input/caesarian-section/caesarian.csv2.csv')
# print(dataset)

X = dataset.drop(columns=['Caesarian'])
y = dataset['Caesarian']
#---------------------------------------------------------KNN---------------------------------------------------------
# knn punya keakuratan yang sangat kecil
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)
    
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
    
model = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
# model = joblib.dump(model, './datatrainknn.joblib')
# model = joblib.load('./datatrainknn.joblib')


# rp = model.predict([[31, 3, 0, 1, 0], [26, 2, 0, 1, 0], [25, 2, 0, 1, 0],
#                     [29, 3, 0, 1, 1], [31, 3, 1, 1, 1], [32, 2, 0, 0, 0],
#                     [21, 1, 0, 2, 0], [27, 2, 1, 0, 0], [28, 2, 0, 0, 1]])
# print(rp)

#mengukur keakuratan dengan menggunakan accuracy_score
score = accuracy_score(y_test, predictions)
print(score) #hasil 50 - 100% dikarenakan masih kurang banyak nya data

#mengukur keakuratan dengan menggunakan confusion matrix
CM = confusion_matrix(y_test, predictions)
print(CM)
#-------------------------------------------------DECISION TREE-----------------------------------------------------

#performa keakuratan mesin jauh lebih baik menggunakan KNN karena dengan KNN dapat mengatur berapa banyak
# tentangga yang dinginkan sehingga cakupan pencarian data menjadi lebih luas, sedangkan DT hanya mengandalkan
# pembagian data training dan testing yang dimana 80% data untuk training dan 20%data untuk Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20) #jika ditambah dengan random_state keakuratan menjadi 75%
#pertanyaan? Apa fungsi sebenar dari random_state(Random Number Generator(RNG))?

# SC = StandardScaler()
# X_train = SC.fit_transform(X_train)
# X_test = SC.transform(X_test)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
# model = joblib.load('../data.joblib')

# rp = model.predict([[31, 3, 0, 1, 0], [26, 2, 0, 1, 0], [25, 2, 0, 1, 0],
#                     [29, 3, 0, 1, 1], [31, 3, 1, 1, 1], [32, 2, 0, 0, 0],
#                     [21, 1, 0, 2, 0],[27, 2, 1, 0, 0]])
# print(rp)

score = accuracy_score(y_test, predictions)
print(score) #sama seperti KNN
CM = confusion_matrix(y_test, predictions)
print(CM)

# tree.export_graphviz(model, out_file='./tree.dot',
#                     feature_names=["'Age'", "'Delivery Number", "'Delivery Time", "'Blood", "'Heart"],
#                     class_names=str(sorted(y.unique())),
#                     label='all',
#                     rounded=True,
#                     filled=True)
print(dataset)