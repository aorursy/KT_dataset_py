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
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/heart.csv")
df.head()
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")
plt.show()
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("percentage of persons haven't heart disease: {:.2f}%".format((countNoDisease/(len(df.target))*100)))
print("percentage have disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
df.target.unique()
y = df.target.values
x_data = df.drop(['target'],axis=1)

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

#x_train = x_train.T
#x_test = x_test.T
#y_train = y_train.T
#y_test = y_test.T

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=18)

knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("{}-NN Score : {}".format(18,knn.score(x_test,y_test)))
#best score
score_list=[]
for each in range(5,20):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(5,20),score_list)
plt.show()