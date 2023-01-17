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
data=pd.read_csv("../input/Reviews.csv")

data.tail()

textdata=data["Text"]
scoreData=data["Score"]

scoreData2= data['Score'].map({0: 0, 1:0,2:0,3:1,4:1,5:1}) 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()

textVectors=vectorizer.fit_transform(textdata)

import math
k=math.sqrt(568454)*0.5 #Rule of thumb for k-nearast-neighbour algorithm
k=round(k)
k

print(k)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(textVectors,scoreData2,test_size=0.33, random_state=0)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=y_test,y_pred=y_pred)
cm

