# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import this cool stuff
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#read the file into a numpy array
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/MnistTrain.csv')

ugly = (data.iloc[:,0]).tolist()
labels = np.array(data.iloc[:,1])

def convert(path,length):
    data = pd.read_csv(path)
    ugly = (data.iloc[:,0]).tolist()
    labels = np.array(data.iloc[:,1])
    ugly = np.array([(np.array([int(float(x)) for x in image.split()])).reshape(28,28) for image in ugly])
    ugly = ugly.reshape(length,784)
    return (ugly,labels)

ugly,labels = convert("MnistTrain.csv",37800)
X_train,X_test,y_train,y_test = train_test_split(ugly,labels,test_size = 0.1,random_state = 49)
print(X_test.shape)
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X_train,y_train)

def make_csv(path_name, model):
    X_predict,y_predict = convert("MnistAnswers.csv",4200)
    answers = model.predict(X_predict)
    data = []
    data.append(["image","label"])
    for i in range(len(answers)):
        wowstring = " ".join(str(float(x)) for x in list(X_predict[i]))
        data.append([wowstring,answers[i]])
    with open(path_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

    csvFile.close()
#make_csv("testsample",neigh)


