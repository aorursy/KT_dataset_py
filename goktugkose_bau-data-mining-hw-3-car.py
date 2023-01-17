import numpy as np 

import pandas as pd 

import os

from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

print(os.listdir("../input/car-evaluation"))
df_car = pd.read_csv("../input/car-evaluation/car.data.txt",header=None)

df_car.head(10)
print('There are', df_car.shape[0], 'rows and', df_car.shape[1], 'columns in the dataset.')
print(df_car[0].value_counts(),"\n")

print(df_car[1].value_counts(),"\n")

print(df_car[2].value_counts(),"\n")

print(df_car[3].value_counts(),"\n")

print(df_car[4].value_counts(),"\n")

print(df_car[5].value_counts(),"\n")

print(df_car[6].value_counts())
le = LabelEncoder()

df_car = df_car.apply(le.fit_transform)
df_car.head(10)
X = df_car.iloc[:,:6]

y = df_car[6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
list_kernel = ['linear', 'poly', 'rbf', 'sigmoid']

list_error = [0.5, 1.0, 1.5, 2.0]

for kernel in list_kernel:

    print("Kernel", ': ',kernel)

    for error in list_error:

        #print('Error (C) = ',error)

        svm = SVC(kernel=kernel, gamma='auto', C = error)

        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)

        scores = cross_val_score(svm, X, y, cv=5)

        print("Accuracy of SVM Algorithm with the",kernel,"kernel",'with an error constant (C) of',error,'= %.3f'%svm.score(X_test, y_test))

        print('Cross Validation Scores: ',scores,'\n')

        print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred),'\n')

    print('\n')