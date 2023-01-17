import pandas as pd
import numpy as np
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
data = pd.read_csv('../input/irisdata.txt', header=None)
y = data[4]
x = data.drop(4, axis=1).astype(np.float64)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.5)
x_train.insert(4,"4",1)
y_train = lb.fit_transform(y_train)
x_test.insert(4,"4",1)
lmbda = 1000
w1 = x_train.T.dot(y_train).T
w2 = np.linalg.inv(x_train.T.dot(x_train) + lmbda)
weights = w1.dot(w2)
y_test = lb.fit_transform(y_test)
y1 = []
maximum = -2
for index, x in x_test.iterrows():
    for idx,w in weights.iterrows():
        a = np.array(x).flatten()
        b = np.array(w).flatten()
        s = a.dot(b)
        if(s > maximum):
            maximum = s
            ind = idx
    maximum = -2
    y1.append(ind)
    ind = 0
y1 = lb.fit_transform(y1)
from sklearn.metrics import accuracy_score
print(1-accuracy_score(y1,y_test))
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y1 ]
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_non_category, y_predict_non_category)
   
