import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import optimize
import math
from scipy.stats import logistic
df = pd.read_csv("../input/train.csv").dropna() # Drop NaN, NULL etc
le = preprocessing.LabelEncoder()  # Encode categorical data as numbers
le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])

X_T = df[['Sex','Age']].values.T # Scaling
max_x = X_T[1].max()
mean_x = X_T[1].mean()
X_T[1] = list(map(lambda el: (el-mean_x)/max_x,X_T[1]))
X = X_T.T
y = df['Survived'].values

# Cost function with logistic error
def J(th, *args): # J(theta) = -1/m * sum( y dot log(sigmoid(theta dot x)) + (1-y) dot log(1-sigmoid(theta dot x) )
    X,y=args
    m = len(y)
    err = []
    for i in range(0,m):
        h_th=logistic.cdf(np.dot(th[:-1],X[i])+th[-1])
        err.append(y[i]*math.log(h_th+0.0001)+(1-y[i])*(math.log(1.0001-h_th)))
    return -1/m  * sum(err)
args1=(X,y)
th0 = np.asarray((1,1,1))  # Initial guess.
res1 = optimize.fmin_cg(J, th0, args=args1) # gradient descent, uses numerical approximation in lieu of gradient
print("Optimal theta: "+str(res1))
y_best = list(map(lambda el: 1 if np.dot(res1[:-1],el)>= -res1[-1] else 0,list(X))) # estimated output from training set

X_t=X.transpose()
plt.subplot(1,2,1)
ax_1 = list(X_t[0])
ax_2 = list(X_t[1])
colors = ['r','g','c']
colors_2 = ['b','y','m']
labels = list(map(lambda label: colors[label],list(y)))
labels_est=list(map(lambda label: colors_2[label],list(y_best)))
plt.scatter(ax_1, ax_2, c=labels, alpha=0.6)
plt.subplot(1,2,2)

theta_1 = res1[0]
theta_2 = res1[1]
theta_k = res1[2]

# th_1*x_1 + th_2*x_2 +th_k = 0 | x_2 = (-th_k - th_1 x_1)/th_2
def y_from_x(x):
    return (-theta_k-theta_1*x)/theta_2
plt.title("Training Set")
x_s = np.arange(0,1.1,0.1)
y_s = list(map(lambda el:y_from_x(el),x_s))

plt.plot( x_s,y_s)

plt.title("Prediction")
x_s = np.arange(-2.5,2.6,0.1)
y_s = list(map(lambda el:logistic.cdf(y_from_x(el)),x_s))

plt.plot( x_s,y_s)
plt.scatter(ax_1, ax_2, c=labels_est, alpha=0.6)
df_2 = pd.read_csv("../input/test.csv")[['PassengerId','Sex','Age']]

df_2['Sex'] = le.transform(df_2['Sex'])
X_test_T = df_2[['Sex','Age']].values.T # Scaling
X_test_T[1] = list(map(lambda el: (el-mean_x)/max_x,X_test_T[1]))
X_test= X_test_T.T
y_est = list(map(lambda el: 1 if np.dot(res1[:-1],el)>= -res1[-1] else 0,list(X_test)))
df_2['Survived'] = pd.Series(y_est)
df_2[['PassengerId','Survived']].to_csv("output.csv",index=False)