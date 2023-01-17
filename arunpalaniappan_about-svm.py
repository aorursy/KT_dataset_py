import numpy as np # linear algebra
import matplotlib.pyplot as plt
X = np.array([[3,5],[2,4],[1,4],[4,8],[1,5],[8,3],[3,1],[7,3],[5,2],[2,0]])
Y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
for d,sample in enumerate(X):
    if d < 5:     #Class 1
        plt.scatter(sample[0],sample[1],marker = "+",s = 120)
    else:
        plt.scatter(sample[0],sample[1],marker = "_",s = 120)
plt.plot([0,8],[0,8])
plt.show()
w = np.zeros(len(X[0]))
epochs = 10000
eta = 1
for epoch in range(1,epochs):
    for  i,x in enumerate(X):
        if (Y[i]*np.dot(X[i],w) < 1):
            w = w + eta*(X[i]*Y[i])+(-2*w*(1/epoch))
        else:
            w = w + eta*(-2*w*(1/epoch))
print (w)            
for i in X:
    if(sum(i*w) > 1):
        print (1)
    else:
        print (-1)
from sklearn import svm
clf = svm.SVC(gamma='auto')  #SVC is support vector machine for classification
clf.fit(X,Y)
clf.predict(X)