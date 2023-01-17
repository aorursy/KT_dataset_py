import pandas as pd
data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
data.tail()
data.info()
data
from seaborn import pairplot
pairplot(data, hue="species")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop("species", axis=1),

                                                    data["species"],

                                                    test_size=0.3)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier   # MACHINE YANG AKAN LEARNING

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(x_train, y_train)   # SI MACHINE LEARNING'NYA DI SINI

clf
hasil_prediksi = clf.predict(x_test)

hasil_prediksi
y_test.ravel()
from sklearn.metrics import accuracy_score
accuracy_score(y_test, hasil_prediksi)