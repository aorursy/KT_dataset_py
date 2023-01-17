# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

import statsmodels.api as sm



from warnings import filterwarnings

filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.head()
df.info()
df.describe().T
abnormal = df[df["class"] == "Abnormal"]

normal = df[df["class"] == "Normal"]



plt.scatter(abnormal.pelvic_radius, abnormal.sacral_slope,color = "red",label = "abnormal")

plt.scatter(normal.pelvic_radius, normal.sacral_slope,color = "green",label = "normal")

plt.legend()

plt.xlabel("pelvic_radius")

plt.ylabel("sacral_slope")

plt.show()
df["class"] = [1 if i=="Abnormal" else 0 for i in df["class"]]
y = df["class"].values

X_ = df.drop("class",axis=1)
X = (X_ - np.min(X_)) / (np.max(X_) - np.min(X_)).values #bagimsiz degiskenleri donusturduk
#test-train

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state = 50)
#knn model

knn = KNeighborsClassifier()

knn_model = knn.fit(X_train,y_train)
#prediction(tahmin)

y_pred = knn_model.predict(X_test)
#accuaccuracy_score(n = 5 iken)

accuracy_score(y_test,y_pred)
#1 den 50 ye kadar "n_neighbors" degerlerini deneyecegiz...

knn_params = {"n_neighbors": np.arange(1,50)}
#En iyi "n_neighbors" degerini bularak model yeniden kuruluyor

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,knn_params, cv=10)

knn_cv.fit(X_train,y_train)
print("En iyi skor:" + str(knn_cv.best_score_))

print("En iyi parametre:" + str(knn_cv.best_params_))