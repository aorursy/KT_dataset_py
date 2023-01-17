import  numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X1, y1 = make_blobs(n_samples=300, centers=2,random_state=0, cluster_std=0.99)


plt.scatter(X1[:,0],X1[:,1],c=y1,cmap='cool')
plt.show()
X1.shape
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X1,y1)
clf.coef_
clf.intercept_
w = clf.coef_[0]
a = -w[0] / w[1]

x1 = np.linspace(-1,4.5,50)
x2 =  a * x1 + (-clf.intercept_[0]) / w[1]
plt.plot(x1,x2)
plt.scatter(X1[:,0],X1[:,1],c=y1,cmap='winter')
plt.title('when C={}'.format(1.0))
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()
def plot_line(clf):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    x1 = np.linspace(-1,4.5,50)
    x2 =  a * x1 + (-clf.intercept_[0]) / w[1]
    return x1,x2

    

#plt.figure(figsize=(8,6))
fig,subplt = plt.subplots(1,2,figsize=(12,4))
p=[.01,10]
for i,sub in zip(p,subplt):
    clf = LinearSVC(C=i)
    clf.fit(X1,y1)
    
    x1,x2=plot_line(clf)
    
    title='When C={}'.format(i)
    sub.plot(x1,x2)
    sub.set_title(title)
    sub.scatter(X1[:,0],X1[:,1],c=y1,cmap='winter')
    sub.set_xlabel('feature_1')
    sub.set_ylabel('feature_2')
   
    
    