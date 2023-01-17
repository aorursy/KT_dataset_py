# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #interactive plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data1 = pd.read_csv("../input/breastcancer-dataset/data.csv")



data1.info()
data1.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data1.diagnosis
M = data1[data1.diagnosis == "M"]

B = data1[data1.diagnosis == "B"]

# scatter plot

plt.subplots(figsize=(12,8))

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Malignant",alpha= 0.6)

plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Benign",alpha= 0.6)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()
data1.diagnosis = [1 if each == "M" else 0 for each in data1.diagnosis]

y = data1.diagnosis.values

x_data = data1.drop(["diagnosis"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

 
print("Print Accuracy of SVM Algoritm: ",svm.score(x_test,y_test))