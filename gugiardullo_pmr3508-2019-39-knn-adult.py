# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', na_values = "?")
test_data =  pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv')
sample = pd.read_csv('/kaggle/input/adult-pmr3508/sample_submission.csv')
train_data.head()
train_data.describe()
train_data.info()
train_data["income"].value_counts().plot(kind="bar")
train_data["relationship"].value_counts().plot(kind="bar")
train_data["education.num"].value_counts().plot(kind="bar")
train_data["workclass"].value_counts().plot(kind="bar")
from sklearn import preprocessing
nadult = train_data.dropna()
nadult.shape
attributes = ["age", "workclass", "education.num","relationship", "capital.gain", "hours.per.week"]
train_adult_x = nadult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
train_adult_x.head()
train_adult_y = nadult.income
test_adult_x = test_data[attributes].apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
#best_n = 0

#best_acc = 0

#for n in range (20,40):

#    neighbors = n

#    cross = 10

#    knn = KNeighborsClassifier(n_neighbors = neighbors)

#    scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = cross)

#    score = np.mean(scores)

#    if (best_acc<score):

#        best_acc = score

#        best_n = n



    
#best_acc
#best_n
knn = KNeighborsClassifier(n_neighbors = 27)

scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = 10)

scores
accuracy = np.mean(scores)

accuracy
knn.fit(train_adult_x, train_adult_y)

test_pred_y = knn.predict(test_adult_x)

test_pred_y
id_index = pd.DataFrame({'Id' : list(range(len(test_pred_y)))})

income = pd.DataFrame({'income' : test_pred_y})

result = id_index.join(income)

result
result.to_csv("submission.csv", index = False)