# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
haberman=pd.read_csv('../input/habermans-survival-data-set/haberman.csv')
col_names = ['age', 'year', 'nodes', 'survival_status']
haberman.columns = col_names
col_names
haberman.head()

#Menentukan variabel independen
x = haberman.drop(["survival_status"], axis = 1)
x.head()
#Menentukan variabel dependen
y = haberman["survival_status"]
y.head()
#Import package model selection dari SKLearn
from sklearn.model_selection import train_test_split
#Membagi data training dan data testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
#Mengaktifkan package dan syntax untuk mengubah skala data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Mengaktifkan packages untuk klasifikasi dengan mengimport packages KNeighbors Classifierdari skLearn
from sklearn.neighbors import KNeighborsClassifier
#Mengaktifkan fungsi classifikasi untuk KNN
knn = KNeighborsClassifier (n_neighbors=4)
#Memasukan data training pada fungsi classifikasi untuk KNN
knn.fit(x_train, y_train)
#Menentukan prediksi
y_pred = knn.predict(x_test)
y_pred
#Menentukan probabilitas prediksi
knn.predict_proba(x_test)
from sklearn.metrics import classification_report, confusion_matrix
#Menampilkan matriks hasil prediksi
print(confusion_matrix(y_test, y_pred))
#Ketepatan hasil prediksi
print(classification_report(y_test, y_pred))
