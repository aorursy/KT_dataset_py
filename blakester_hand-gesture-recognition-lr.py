import numpy as np 

import pandas as pd 

import os

import warnings

warnings.filterwarnings("ignore")

print(os.listdir("../input/leapgestrecog/leapGestRecog"))
from PIL import Image

import matplotlib.image as mpimg 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import IPython.display

path='../input/leapgestrecog/leapGestRecog'

folders=os.listdir(path)

folders=set(folders)



different_classes=os.listdir(path+'/'+'00')

different_classes=set(different_classes)



print("The different classes that exist in this dataset are:")

print(different_classes)
x=[]

z=[]

y=[]

threshold=200

import cv2





for i in folders:

    print('***',i,'***')

    subject=path+'/'+i

    subdir=os.listdir(subject)

    subdir=set(subdir)

    for j in subdir:

        images=os.listdir(subject+'/'+j)

        for k in images:

            results=dict()

            results['y']=j.split('_')[0]

            img = cv2.imread(subject+'/'+j+'/'+k,0)

            img=cv2.resize(img,(int(160),int(60)))

            

            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            imgD=np.asarray(img,dtype=np.float64)

            z.append(imgD)

            imgf=np.asarray(imgf,dtype=np.float64)

            x.append(imgf)

            y.append(int(j.split('_')[0]))

            results['x']=imgf
l = []

list_names = []

for i in range(10):

    l.append(0)

for i in range(len(x)):

    if(l[y[i] - 1] == 0):

        l[y[i] - 1] = i

        if(len(np.unique(l)) == 10):

            break

for i in range(len(l)):

    %matplotlib inline

    print("Class Label: " + str(i + 1))

    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)

    plt.show()

    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     

    plt.show()
x=np.array(x)

y=np.array(y)

y = y.reshape(len(x), 1)

print(x.shape)

print(y.shape)

print(max(y),min(y))
x_data = x.reshape((len(x), 60, 160, 1))



x_data/=255

x_data=list(x_data)

for i in range(len(x_data)):

    x_data[i]=x_data[i].flatten()
len(x_data)
from sklearn.decomposition import PCA

pca = PCA(n_components=20)

x_data=np.array(x_data)

print("Before PCA",x_data.shape)
x_data=pca.fit_transform(x_data)

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  



print('___________________')

print("After PCA",x_data.shape)
from sklearn.model_selection import train_test_split

x_train,x_further,y_train,y_further = train_test_split(x_data,y,test_size = 0.2)

x_train,x_valid,y_train, y_valid = train_test_split(x_train,y_train,test_size = 0.5)
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

#The idea behind StandardScaler is that it will transform your data

#such that its distribution will have a mean value 0 and standard deviation of 1.

scaler.fit(x_train)



X_train = scaler.transform(x_train)  

X_valid = scaler.transform(x_valid)

X_test = scaler.transform(x_further)  
from sklearn.linear_model import LogisticRegression



solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

solver_error = []

for solverInstance in solvers:

    logistic = LogisticRegression(solver=solverInstance, multi_class='auto')

    logistic.fit(X_train, y_train)

    y_pred_logistic=logistic.predict(X_test)

    error = 1 - logistic.score(X_valid, y_valid)

    solver_error.append(error)

plt.plot(solvers, solver_error)

plt.title('Solver vs. Model Error')

plt.xlabel('Solver')

plt.ylabel('Error')

plt.show()



minError = solver_error.index(min(solver_error))

bestSolver = solvers[minError]



print("Optimal Solver: {}".format(bestSolver))
c_values = [.001, .01, .1, 1, 10, 100, 1000]

c_error = []

for c_value in c_values:

    logistic = LogisticRegression(solver='newton-cg', C = c_value, multi_class='auto')

    logistic.fit(X_train, y_train)

    error = 1 - logistic.score(X_valid, y_valid)

    c_error.append(error)

    

plt.loglog(c_values, c_error)

plt.title('C Value vs. Model Error')

plt.xlabel('C Value')

plt.ylabel('Error')

plt.show()



minError = c_error.index(min(c_error))

bestC = c_values[minError]



print("Optimal C value: {}".format(bestC))
iter_values = [100, 250, 500, 750, 1000]

iter_error = []

for num_iter in iter_values:

    logistic = LogisticRegression(solver='newton-cg', C = 10, max_iter=num_iter, multi_class='auto')

    logistic.fit(X_train, y_train)

    error = 1 - logistic.score(X_valid, y_valid)

    iter_error.append(error)

    

plt.plot(iter_values, iter_error)

plt.title('Iterations vs. Model Error')

plt.xlabel('Iterations')

plt.ylabel('Error')

plt.show()



minError = iter_error.index(min(iter_error))

bestIter = iter_values[minError]



print("Optimal Iterations: {}".format(bestIter))
logistic = LogisticRegression(solver=bestSolver, C=bestC, max_iter=bestIter, multi_class='auto')

logistic.fit(X_train, y_train)

y_pred_logistic=logistic.predict(X_test)

y_train_score_logistic=logistic.predict(X_train)
from sklearn.metrics import accuracy_score

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_logistic, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_logistic, normalize=True, sample_weight=None))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



matrix = confusion_matrix(y_further, y_pred_logistic, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

precision = precision_score(y_further, y_pred_logistic, average = None)

accuracy = accuracy_score(y_further, y_pred_logistic, normalize=True, sample_weight=None)

recall = recall_score(y_further, y_pred_logistic, average=None)



print("Confusion Matrix:\n", matrix, "\n")

print("Accuracy:", accuracy, "\n")

print("Recall:", recall, "\n")

print("Precision:", precision)