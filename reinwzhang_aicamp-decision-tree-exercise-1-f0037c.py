%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
def plot_decision_boundary(X, model):
    h = .02
    
    x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contour(xx, yy, z, cmap=plt.cm.Paired)
    

#np.random.seed(10)
#
#N = 300
#D = 2
#X = np.random.randn(N, D)

#delta = 1.5
## this following means each element of each row and column add delta
#X[:N//3] += np.array([delta, delta])
#X[N//3+1:2*N//3] += np.array([-delta, delta])
#X[2*N//3+1:] += np.array([-delta, -delta])
#Y = np.array([0] * (N//4) + [1] * (N//4) + [0] * (N//4) + [1] * (N//4) )

#important, this generates a N size array with half 0 and half 1
# for list "+" means concatenate 
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.5
# this following means each element of each row and column add delta
X[:N//2] += np.array([delta, delta])
X[N//2:] += np.array([-delta, -delta])
Y = np.array([0] * (N//2) + [1] * (N//2) )
plt.scatter(X[:,0], X[:,1], s=200, c=Y, alpha=0.5)
plt.title('scatter plot 1')
plt.show()

model = DecisionTreeClassifier()
model.fit(X, Y)
print(model.tree_)
print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()
h = 0.1
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
# if Sparse = True len(xx) = 1
xxv = np.c_[xx.ravel(), yy.ravel()]
print(len(xxv))

z = model.predict(xxv)
z = z.reshape(xx.shape)

#z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
plt.contourf(xx, yy, z)
plt.contour(xx, yy, z)
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
model_depth_3.fit(X,Y)
print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()
model_depth_5 = DecisionTreeClassifier(criterion='entropy', max_depth = 5)
model_depth_5.fit(X,Y)
print("score for basic tree:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.75
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [1] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X,Y)

print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()


print("score for basic tree:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()


np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.75
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [0] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X,Y)


print("score for basic tree:", model.score(X, Y))
plt.subplot(221)
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
print(model.splitter)
plt.title('Auto depth for XOR data')
plt.show()


print("score for depth_3 tree:", model_depth_3.score(X, Y))
plt.subplot(222)
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
print(model_depth_3.splitter)
plt.title('Depth 3 for XOR data')
plt.show()


print("score for depth_5 tree:", model_depth_5.score(X, Y))
plt.subplot(222)
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
print(model_depth_5.splitter)
plt.title('Depth 5 for XOR data')
plt.show()

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller = 5
R_larger = 10

R1 = np.random.randn(N//2) + R_smaller
theta = 2 * np.pi * np.random.random(N//2)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T


R2 = np.random.randn(N//2) + R_larger
theta = 2 * np.pi * np.random.random(N//2)
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

Y = np.array([0] * (N//2) + [1] * (N//2))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()
model.fit(X, Y)
model_depth_3.fit(X, Y)
model_depth_5.fit(X,Y)


print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()


print("score for basic tree:", model_depth_5.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_5)
plt.show()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, Y)
clf.score(X, Y)