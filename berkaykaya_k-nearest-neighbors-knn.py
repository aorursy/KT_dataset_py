# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.rename(columns={"pelvic_tilt numeric":"pelvic_tilt_numeric"},inplace = True)
data
A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]

plt.scatter(A.pelvic_tilt_numeric,A.lumbar_lordosis_angle,color = "r",label = "Abnormal",alpha = 0.3)

plt.scatter(N.pelvic_tilt_numeric,N.lumbar_lordosis_angle,color = "b",label = "Normal",alpha = 0.3)

plt.xlabel("pelvic_tilt_numeric")

plt.ylabel("lumbar_lordosis_angle")

plt.legend()

plt.show()
data["class"] = [1 if each == "Abnormal" else 0  for each in data["class"]]  # classtaki isimleri integer hale getiryoruz.

y=data["class"].values                      

x_data = data.drop(["class"],axis = 1) 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3 , random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(x_train,y_train)

prediction=(knn.predict(x_test))

print("{} nn Score : {}".format(3,knn.score(x_test,y_test)))

score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("K Values")       

plt.ylabel("Accuracy")

plt.show()