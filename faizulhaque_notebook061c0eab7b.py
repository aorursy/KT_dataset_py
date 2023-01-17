#RFE svc based classifier

#Faiz ul haque Zeya

#Data scientist



import numpy as np # linear algebra AND PROBABILITY

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import LinearSVC

from sklearn.feature_selection import RFE 

from sklearn import datasets



dataset = datasets.load_iris()



#select Linear SVC with recursive feature extraction with 4 frearues

svm = LinearSVC()

rfe = RFE(svm, 4)

rfe = rfe.fit(dataset.data, dataset.target)

#predict the data set

a=rfe.predict(dataset.data[:])

#print the predcited

print(a)

#print the target

b=dataset.target[:]

print(b)

#Calculate error

error=np.array(a-b)

notzero = (error != 0).sum()

print("Number of misclassified:",notzero)
