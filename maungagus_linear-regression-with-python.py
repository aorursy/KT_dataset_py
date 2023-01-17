# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/USA_Housing.csv') #Import data
df.head() #Melihat 5 data pertama
df.describe() #Melihat statistik setiap feature (variabel independen)
df.columns #Untuk melihat nama-nama kolom feature atau variabel independen
sns.pairplot(df) #Untuk melihat visualisasi hubungan (korelasi) antar feature atau variabel independen
sns.distplot(df.Price) #Melihat distribusi Price
X = df.drop(['Price','Address'], axis=1) #Untuk mendapatkan variabel independen atau feature. Karena Address tidak mempengaruhi harga rumah maka di-drop dan Price adalah variabel dependen atau target.
X.head()
y = df.Price
y.head()
from sklearn.model_selection import train_test_split #Untuk split data menjadi data pembuatan model dan data untuk test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #test size 0.3 atau 30% berarti 30% dari data yg ada digunakan untuk test, dan sisanya digunakan untuk membuat model.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train) #Memasukan data untuk membentuk model
print('b =', model.intercept_)
coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df
pred = model.predict(X_test) #Memprediksi sebagian data (X Test) menggunakan model
plt.scatter(y_test,pred)
sns.distplot((y_test-pred),bins=50)
from sklearn import metrics
metrics.mean_absolute_error(y_test,pred)
metrics.mean_squared_error(y_test,pred)
np.sqrt(metrics.mean_squared_error(y_test, pred))
abs(y_test-pred).mean()
((y_test-pred)**2).mean()
np.sqrt(((y_test-pred)**2).mean())
metrics.r2_score(y_test,pred)