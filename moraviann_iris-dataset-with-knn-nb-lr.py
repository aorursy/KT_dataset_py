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
df = pd.read_csv("/kaggle/input/iris/Iris.csv")

df.rename(columns={"SepalLengthCm": "sepal_length", "SepalWidthCm": "sepal_width"},inplace=True ,errors="raise")



df.head()
df.info()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(df["Species"])

integer_encoded
df["class"]= integer_encoded

df.head()
import matplotlib.pyplot as plt



colors = {0:'r', 1:'g', 2:'b'}

# create a figure and axis

fig, ax = plt.subplots()

# plot each data-point

for i in range(len(df['sepal_length'])):

    ax.scatter(df['sepal_length'][i], df['sepal_width'][i],color=colors[df["class"][i]])

# set a title and labels

ax.set_title('Iris Dataset')

ax.set_xlabel('sepal_length')

ax.set_ylabel('sepal_width')



from sklearn.model_selection import train_test_split



X = df[['sepal_length','sepal_width','PetalLengthCm','PetalWidthCm']]

y = df.iloc[:,6].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=24)



print(X)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import model_selection



error_rate = []

accuracylist = []

crossV = []



for i in range(1,15):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    accuracy= metrics.accuracy_score(y_pred, y_test)*100

    error_rate.append(np.mean(y_pred != y_test))

    kfold = model_selection.KFold(n_splits=10, random_state=0)

    scores = cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')*100

    accuracylist.append(accuracy)

    crossV.append(scores.mean())

    

plt.figure(figsize=(10,6))

plt.plot(range(1,15),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')    



for i, (value1,value2) in enumerate(zip(accuracylist, crossV),1):

    print("For k =",i,"Accuracy:{0:.2f} CrossValScore:{1:.2f} ".format(value1,value2))

    

    



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression





clfs = {

    'KNN': KNeighborsClassifier(n_neighbors = 8),

    'Naive_bayes': GaussianNB(),

    'Logistic_Regression': LogisticRegression()

}



for clf_name in clfs.keys():

    print("Training",clf_name,"classifier")

    clf = clfs[clf_name]

    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    print(classification_report(y_test, y_predict))

    print()