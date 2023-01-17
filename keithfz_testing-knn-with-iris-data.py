# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris_data = pd.read_csv("../input/Iris.csv")
iris_data.head()
sns.set(style="ticks")
sns.set_palette("husl")
sns.pairplot(iris_data.iloc[:,1:6],hue="Species")
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
species_encoded=le.fit_transform(iris_data.Species)
def run_knn(data, labels, neighbors):
    X_train, X_test, y_train, y_test = train_test_split(data,labels,random_state=7,train_size=0.7)

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
accuracy_list = []
for i in range(1, 51):
    accuracy = run_knn(iris_data.iloc[:, [1,2,3,4]].values, species_encoded, i)
    accuracy_list.append(accuracy)

df = pd.DataFrame({'Accuracy':accuracy_list})
df['Neighbors'] = df.index + 1
ax = sns.lineplot(x="Neighbors", y="Accuracy", data=df)

