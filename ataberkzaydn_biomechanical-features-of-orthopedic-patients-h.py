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

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.head()

a_list = df["class"].unique()

print(a_list)
df['class'] = df['class'].replace("Abnormal", 0)

df['class'] = df['class'].replace("Normal", 1)

df['class'] = df['class'].astype(int)

df.info()
df["class"].unique()
x_data = df.drop(["class"],axis=1)

y = df["class"].values
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.45, random_state=42)
neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[16,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

knn = KNeighborsClassifier(n_neighbors=test_accuracy.index(np.max(test_accuracy))) 
#from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.45,random_state = 1)

#knn = KNeighborsClassifier(n_neighbors = 12)

x,y = df.loc[:,df.columns != 'class'], df.loc[:,'class']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy



y_true = y_test

y_pred = knn.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

print(cm)
import seaborn as sns



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="red",fmt =".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()