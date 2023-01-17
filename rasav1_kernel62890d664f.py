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
df0 = pd.read_csv('/kaggle/input/emg-4/0.csv', header = None)

df0['gesture'] = 0

df0.head()
df1 = pd.read_csv('/kaggle/input/emg-4/1.csv', header = None)

df1['gesture'] = 1
df2 = pd.read_csv('/kaggle/input/emg-4/2.csv', header = None)

df2['gesture'] = 2
df3 = pd.read_csv('/kaggle/input/emg-4/3.csv', header = None)

df3['gesture'] = 3
from functools import reduce 
df = df0.merge(df1, how = 'outer')

df = df.merge(df2, how = 'outer')

df = df.merge(df3, how = 'outer')

from sklearn.utils import shuffle

df = shuffle(df)

df
#  разделяю предикоторы и отклики

X = df.iloc[:, 1:64] 

y = df.iloc[:, 65]
#  разделяю на обучающую и тестовую выборку



from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  

#  Стандартизация предикторов



from sklearn.preprocessing import StandardScaler  



scaler = StandardScaler()  

scaler.fit(X_train)



X_train_2 = scaler.transform(X_train)  

X_test_2 = scaler.transform(X_test)  
#  Построение классификатора

#  По умолчанию Евклидово расстояние



from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors = 4, weights =  'distance')  

classifier.fit(X_train_2, y_train)  
y_pred_train = classifier.predict(X_train_2)  

y_pred_test = classifier.predict(X_test_2)  
from sklearn.metrics import classification_report, confusion_matrix  
classifier.score(X_test_2, y_test)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy', class_weight=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn import metrics



conf_mat = metrics.confusion_matrix(y_test, y_pred)

conf_mat = pd.DataFrame(conf_mat, index=model.classes_, columns=model.classes_)

conf_mat
print(metrics.classification_report(y_pred, y_test))