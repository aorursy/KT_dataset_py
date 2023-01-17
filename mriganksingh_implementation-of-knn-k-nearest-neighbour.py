import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
cancer=datasets.load_breast_cancer()
x_train,x_test,y_train,y_test=model_selection.train_test_split(cancer.data,cancer.target)
def fit(x_train,y_train):
    return
def predict_one(x_train,y_train,x_test,k):
    distances=[]
    for i in range(0,len(x_train),1):
        distance=((x_train[i,:]-x_test)**2).sum()
        distances.append([distance,i])
    distances=sorted(distances)
    targets=[]
    for i in range(0,k,1):
        targets.append(y_train[distances[i][1]])
    return Counter(targets).most_common(1)[0][0]    
def predict(x_train,y_train,x_test,k):
    predictions=[]
    for x in x_test:
        predictions.append(predict_one(x_train,y_train,x,k))
    return predictions    
clf=fit(x_train,y_train)
y_pred=predict(x_train,y_train,x_test,7)
accuracy_score(y_test,y_pred)
