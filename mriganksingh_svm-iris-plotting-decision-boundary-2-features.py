from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
iris=datasets.load_iris()
x=iris.data[:,0:2]
y=iris.target
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
clf=svm.SVC()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
def makegrid(x1,x2,h=0.02):
    x1_min,x1_max=min(x1)-1,max(x1)+1
    x2_min,x2_max=min(x2)-1,max(x2)+1
    a=np.arange(x1_min,x1_max,h)
    b=np.arange(x2_min,x2_max,h)
    xx,yy=np.meshgrid(a,b)
    return xx,yy
xx,yy=makegrid(x[:,0],x[:,1])
predictions=clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.scatter(xx.ravel(),yy.ravel(),c=predictions)
plt.show()
clf1=svm.SVC(kernel='linear')
clf1.fit(x_train,y_train)
clf1.score(x_test,y_test)
xx1,yy1=makegrid(x[:,0],x[:,1])
predictions=clf1.predict(np.c_[xx1.ravel(),yy1.ravel()])
plt.scatter(xx1.ravel(),yy1.ravel(),c=predictions)
plt.show()