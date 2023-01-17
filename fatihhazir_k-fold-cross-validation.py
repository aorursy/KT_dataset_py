import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
df.sample(10)
X = df.iloc[:, [2, 3]]

y = df.iloc[:, 4]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
X_train.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
crossValidation = cross_val_score(estimator = classifier,#Egitmek icin kullandigimiz model.

                                  X=X_train, 

                                  y=y_train, 

                                  cv = 4)# Katlama sayisi.
crossValidation