# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data.head()
data.info()
data['class']
#verimi x , y diye ayırdım

x,y=data.loc[:,data.columns !='class'],data.loc[:,data.columns =='class']
#x verilerimi normalize ettim

x =(x-np.min(x))/(np.max(x)-np.min(x))
#verimi egitim ve test diye böldüm

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=42)
x_test
#knn algoritmam

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction =knn.predict(x_test)

print('{} knn score {}'.format(3,knn.score(x_test,y_test)))
#belli bir aralıktaki k degerlerine göre score degerlerimi buluyorum

score_list =[]

for i in range(1,20):

    knn_2=KNeighborsClassifier(n_neighbors=i)

    knn_2.fit(x_train,y_train)

    score_list.append(knn_2.score(x_test,y_test))
#scorlarımı attıgım dizi

score_list
#grafik çizdiriyorum

plt.plot(range(1,20),score_list)

plt.xlabel('k_values')

plt.ylabel('accuracy')

plt.show()