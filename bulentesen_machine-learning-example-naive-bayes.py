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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

veri = pd.read_csv('../input/data.csv')
veri.head()
veri.drop(["Unnamed: 32", "id"], axis=1, inplace= True)
M = veri[veri.diagnosis == "M"]

B = veri[veri.diagnosis == "B"]



plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha=0.3)

plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.3)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()
veri.head()

veri.diagnosis = [1 if each == "M" else 0 for each in veri.diagnosis]
y = veri.diagnosis.values

x_data = veri.drop(["diagnosis"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train, x_validate, y_train, y_validate = train_test_split(x_train,y_train,test_size=0.25, random_state=42)
x_train.head()
x_test.head()
print("x_train: ",x_train.shape)

print("y_train: ",y_train.shape)

print("x_test: ",x_test.shape)

print("y_test: ",y_test.shape)

print("x_validate: ",x_test.shape)

print("y_validate: ",y_test.shape)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

print("Tahmin Edilen Deger: ",y_pred)
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

print("Confussion Matrisi: ",confusion_matrix(y_test, y_pred))

print("Dogruluk oranı: ",metrics.accuracy_score(y_test,y_pred))

print("f1 Score(macro): ",f1_score(y_test, y_pred, average='macro'))

print("f1 Score(micro): ",f1_score(y_test, y_pred, average='micro'))

print("f1 Score(Agirliklandirilmis): ",f1_score(y_test, y_pred, average='weighted'))

print("Kesinlik Değeri(macro): ",precision_score(y_test, y_pred, average='macro'))

print("Kesinlik Değeri(micro): ",precision_score(y_test, y_pred, average='micro'))

print("Kesinlik Değeri(Agirliklandirilmis): ",precision_score(y_test, y_pred, average='weighted'))
