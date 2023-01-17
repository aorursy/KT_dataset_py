import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/winewithclassification/Wine.csv')
df.head()
df.shape
X = df.iloc[:, 0:13]

y = df.iloc[:, 13]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
X_train.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # kac oznitelik alinacagi

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)
X_train_pca[5]
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)# 13 oznitelik ile tahmin.
classifier_pca = LogisticRegression(random_state=0)

classifier_pca.fit(X_train_pca,y_train)

y_pred_pca = classifier_pca.predict(X_test_pca)# 2 oznitelik ile tahmin
from sklearn.metrics import accuracy_score
PCAOncesi = accuracy_score(y_test,y_pred)

PCASonrasi = accuracy_score(y_test,y_pred_pca)
print("Onceki skor :  " + str(PCAOncesi))

print("Sonraki skor :  " + str(PCASonrasi))