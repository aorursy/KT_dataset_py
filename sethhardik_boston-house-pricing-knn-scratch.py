#IMPORT FILE NECESSARY TO RUN THE CODE

#dataset is imported in order to get the boston house dataset from sklearn 

from sklearn import datasets

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import mean_squared_error
boston=datasets.load_boston()

x=boston.data[:,:]

y=boston.target

print(x.shape,y.shape)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)
def dis(v,w):

    return np.sqrt(np.sum((v-w)**2))
def knn_r(tr, tr_lab, te , k):

    distances = []

    

    for i in range(tr.shape[0]):

        distances.append(dis(tr[i], te))

    

    distances = np.array(distances)

    inds = np.argsort(distances)

    

    distances = distances[inds]

    tr_lab_2 = tr_lab[inds]

    value = np.average(tr_lab_2[:k])

    

    return value
def knn_reg(tr , tr_lab, te , te_lab , k):

    preds = []

    for i in range(te.shape[0]):

        value = knn_r(tr, tr_lab, te[i] , k)

        preds.append(value)

    

    preds  = np.array(preds)

    err = mean_squared_error(te_lab , preds)

    return err
acc = knn_reg(xtrain , ytrain , xtest , ytest ,5)

print ("MEAN SQUARED ERROR:",acc)