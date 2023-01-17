# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/knn-algorithm-dataset/KNNAlgorithmDataset.csv")
data.info()
data.describe()
data.columns
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data.tail()

# malignant = M  kotu huylu tumor

# benign = B     iyi huylu tumor



# %%

M = data[data.diagnosis == "M"]

B = data[data.diagnosis == "B"]

# scatter plot

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)

plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()



# %%

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)



# %%

# normalization 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#%%

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# %%

# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))



# %%

# find k value

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()