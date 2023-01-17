# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

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
from sklearn.neighbors import KNeighborsClassifier  # MACHINE AKAN LEARNING

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(x_train, y_train)  #MACHINE AKAN LEARNING DISINI

clf
hasil_prediksi = clf.predict(x_test)

hasil_prediksi
y_test.ravel()
from sklearn.metrics import accuracy_score
accuracy_score(y_test, hasil_prediksi)