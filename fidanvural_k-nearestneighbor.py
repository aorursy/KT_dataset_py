# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# csv dosyamızı alıyoruz.

data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.head()
data.tail()
abnormal=data[data["class"]=="Abnormal"]

normal=data[data["class"]=="Normal"]
abnormal.info()
normal.info()
# Rastgele ikili alıp görselleştirelim



plt.scatter(abnormal["pelvic_incidence"],abnormal["pelvic_radius"],color="red",label="kötü")

plt.scatter(normal["pelvic_incidence"],normal["pelvic_radius"],color="green",label="iyi")

plt.legend()

plt.show()
data["class"]=[1 if each=="Abnormal" else 0 for each in data["class"]]

y=data["class"].values

x_data=data.drop(["class"],axis=1)

y
# Normalization



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

x
# Train,Test Split



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
# KNN model



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3) # k_neihgbors = k

knn.fit(x_train,y_train)
prediction=knn.predict(x_test)

prediction
# Doğruluk oranımıza bir bakalım

print("Accuracy: {}".format(knn.score(x_test,y_test)))
# Farklı k değerleri deneyerek en doğru oranı bulmaya çalışıyoruz

score_list_test=[]

score_list_train=[]



for each in range(1,40):

    knn_=KNeighborsClassifier(n_neighbors=each)

    knn_.fit(x_train,y_train)

    print("{} Test: {}".format(each,knn_.score(x_test,y_test)))

    score_list_test.append(knn_.score(x_test,y_test))

    score_list_train.append(knn_.score(x_train,y_train))



plt.plot(range(1,40),score_list_train,color="black",label="train")

plt.plot(range(1,40),score_list_test,color="purple",label="test")

plt.legend()

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()