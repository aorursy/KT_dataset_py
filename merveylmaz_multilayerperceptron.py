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



# Datasetimizi yüklüyoruz

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

# X değişkenine data sütunlarını atıyoruz

X = data.data



# Y değişkeninine son sütunu atıyoruz

y = data.target

# Dataset eğitim ve test verisi olarak ikiye ayrılıyoruz

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
# Eğitim ve test verileri için StandardScaler ile dönüşümleri yapıyoruz

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# MLP ile sınıflandırma işlemi gerçekleştiriyoruz

# Giriş katmanı: 7, gizli katman: 8, çıkış katmanı: 3

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(7, 8, 3), max_iter=1000)

mlp.fit(X_train, y_train)
# Test verileriyle tahmin gerçekleştiriyoruz

predictions = mlp.predict(X_test)
# Confusion matrix ve report değerleri hesaplıyoruz

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test,predictions)

print(cm)

print(classification_report(y_test,predictions))
# Sensitivity ve specificity hesaplıyoruz

TP = cm[1][1]

TN = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]



sensitivity  = TP / (TP+FN)

specificity  = TN / (TN+FP)



print(" Sensitivity : ",sensitivity)

print(" Specificity : ",specificity)