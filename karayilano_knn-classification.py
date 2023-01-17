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
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df.info()
df.head()
#Making abnormal ones 1 and normal ones 0

df["class"] = [1 if i == "Abnormal" else 0 for i in df["class"]]
x = df.drop(["class"], axis=1)

y = df["class"]
# normalization process

x = (x - x.min()) / (x.max() - x.min())
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)
# knn for neighbor value 3

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

knn.score(x_test, y_test)
#lets find what is best neighbor value



import matplotlib.pyplot as plt

score_list = []

for i in range(1,100):

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test, y_test))

    

plt.plot(range(1,100), score_list)

plt.xlabel("neighbor value")

plt.ylabel("score")

plt.show()
import numpy as np

np.argmax(score_list)
#lets check

knn = KNeighborsClassifier(n_neighbors=38)

knn.fit(x_train, y_train)

knn.score(x_test, y_test)
score_list[37]