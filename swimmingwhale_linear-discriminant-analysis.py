from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target

def calcMC(X,Classes):
    mean = []
    covariance = []
    for i in range(Classes):
        u = np.mean(X[y == i],axis = 0)
        mean.append(u)
        X_u = X-u
        sig = np.dot(X_u.T,X_u)
        covariance.append(sig)
    return mean,covariance

mean,covariance = calcMC(X,2)
print(mean)
print(covariance)
def calcS(covariance):
    return np.sum(covariance,axis=0)

S = calcS(covariance)
print(S)
def calcw(S,mean):
    # 为保证数值解的稳定性,通过奇异值分解求伪逆
    S_inv = np.linalg.pinv(S)
    return np.dot(S_inv,mean[0]-mean[1])

w = calcw(S,mean)
print(w)
def predect(X,w,mean):
    shadow_c1 = np.dot(w.T,mean[0])
    shadow_c2 = np.dot(w.T,mean[1])
    
    y_pred = []
    shadows = []
    for i in range(len(X)):
        shadow = np.dot(w.T,X[i])
        shadows.append(shadow)
        if (shadow - shadow_c1)**2 < (shadow - shadow_c2)**2:
            y = 0
        else:
            y = 1
        y_pred.append(y)
    return np.array(y_pred),np.array(shadows)

indexs = (y == 0) | (y ==1)
y_pred,shadows = predect(X[indexs],w,mean)
import matplotlib.pyplot as plt
plt.scatter(shadows[y_pred==0], np.zeros(len(shadows[y_pred==0])), alpha=.8, color='navy')
plt.scatter(shadows[y_pred==1], np.zeros(len(shadows[y_pred==1])),alpha=.8, color='turquoise')
from sklearn.metrics import accuracy_score

print(accuracy_score(y[indexs],y_pred))
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names


lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()