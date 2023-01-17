from sklearn import datasets

import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]

y = iris.target



print('Class labels:', np.unique(y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y)
print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train', np.bincount(y_train))

print('Labels counts in y_test', np.bincount(y_test))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0 = 0.1, random_state=1)

ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
print('Miscalssified samples: %d'%(y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: {0:.2f}'.format(accuracy_score(y_test,y_pred)))
#alternative

print('Accuracy: {0:.2f}'.format(ppn.score(X_test_std,y_test)))
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifiers, test_idx = None, resolution = 0.02):

    

    markers = ('s','x','o','^','v')

    colors = ('red','blue','lightgreen','gray','cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1

    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1

    

    xx1, xx2 = np.meshgrid(

        np.arange(x1_min, x1_max, resolution),

        np.arange(x2_min, x2_max, resolution)

    )

    

    Z = classifiers.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    plt.contourf(xx1, xx2, Z, cmap = cmap, alpha = 0.3)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())

    

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x = X[y==cl, 0], y = X[y==cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor='black')

        

    if test_idx:

        X_test, y_test = X[test_idx, :], y[test_idx]

        

        plt.scatter(X_test[:,0], X_test[:,1], c = '', edgecolor='black', alpha = 1.0, linewidth = 1, marker = 'o', s = 100, label = 'test set')
X_combined_std = np.vstack([X_train_std,X_test_std])

y_combined = np.hstack([y_train,y_test])
plot_decision_regions(X = X_combined_std, y = y_combined, classifiers= ppn, test_idx= range(105,150))

plt.xlabel('petal length [stanarized]')

plt.ylabel('petal width [standarized]')

plt.legend(loc = 'upper left')

plt.show()
def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7,7,0.1)

phi_z = sigmoid(z)

plt.plot(z,phi_z)

plt.axvline(0.0,c = 'k')

plt.ylim(-0.1,1.1)

plt.xlabel('z')

plt.ylabel('$\phi (z)$')

plt.yticks([0.0,0.5,1.0])

ax = plt.gca()

ax.yaxis.grid(True)
def cost_1(z):

    return -np.log(sigmoid(z))



def cost_0(z):

    return -np.log(1 - sigmoid(z))
z = np.arange(-10,10,0.1)

phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z] 

c2 = [cost_0(x) for x in z]
plt.plot(phi_z, c1, label = 'J(w) if y = 1')

plt.plot(phi_z, c2, label = 'J(w) if y = 0', color = 'red')

plt.legend(loc = "upper left")

plt.show()
class LogisticRegressionGD(object):

    

    def __init__(self, epoch = 50, learning_rate = 0.01, random_state = 1):

        self.epoch = epoch

        self.learning_rate = learning_rate

        self.random_state = random_state

        

    def net_input(self, X):

        return np.dot(X,self.w_[1:]) + self.w_[0]

    

    def predict(self, X):

        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    

    def activation(self,z):

        return 1. / (1. + np.exp(-z))

        

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])

        self.costs_ = []

        

        for i in range(self.epoch):

            net_input = self.net_input(X)

            output = self.activation(net_input)

            errors = (y - output)

            self.w_[1:] += self.learning_rate * X.T.dot(errors)

            self.w_[0] += self.learning_rate * errors.sum()

            

            cost = (-y.dot(np.log(output))) - ((1 - y).dot(np.log(1-output)))

            self.costs_.append(cost)

            

        return self

            
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)] #because our model is binary model, so we are taking only two classes

y_train_01_subset = y_train[(y_train)==0 | (y_train == 1)]
print('The shape of our X and y is', X_train_01_subset.shape, y_train_01_subset.shape)
model = LogisticRegressionGD(learning_rate=0.05, epoch = 1000)
model.fit(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X_train_01_subset,y_train_01_subset, model)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=100.0, random_state=1) #X_train_std is standarize data, rememver standarization help in gradient descent optimization, it help in converging earlier

model.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, model, range(105,150))

plt.xlabel('Petal length [standarized]')

plt.ylabel('Petal length [standarized]')

plt.legend(loc = 'upper left')

plt.show()
model.predict_proba(X_test_std[:3,:])

model.predict_proba(X_test_std[:3,:]).sum(axis=1) #total probability sum will be 1
model.predict_proba(X_test_std[0,:].reshape(1,-1))
model.coef_
#Plotting the L2-regularization path for the two weight coefficients:

#for visualising how decreasing C will increase the strength of regularization

weights, params = [], []



for c in np.arange(-5,5):

    model = LogisticRegression(C=10. ** c, random_state=1)

    model.fit(X_train_std, y_train)

    weights.append(model.coef_[1]) # we are considering weight of our class 1 in iris datasets 

    params.append(10.**c)
weights = np.array(weights)

plt.plot(params, weights[:,0], label = 'petal length')

plt.plot(params, weights[:,1], label = 'petal width', linestyle = '--')

plt.ylabel('weights coefficient')

plt.xlabel('C')

plt.legend(loc = 'upper left')

plt.xscale('log')

plt.show()
from sklearn.svm import SVC
#initializing the model, here C is the inverse of regularization parameter

model = SVC(kernel='linear', C = 1.0, random_state=1)



#fitting the model

model.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, model, range(105,150))

plt.xlabel('petal length [standarized]')

plt.ylabel('petal width [standarized]')

plt.legend(loc = 'upper left')

plt.show()
#creating nonlinear classification data



np.random.seed(1)



X_xor = np.random.randn(200,2)

y_xor = np.logical_xor(X_xor[:,0] > 0,

                       X_xor[:, 1] > 0

                      )





y_xor = np.where(y_xor, 1 ,-1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor==1, 1], c = 'b', marker = 'x', label = '1')

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c = 'r', marker = 's', label = '1')

plt.xlim([-3,3])

plt.ylim([-3,3])

plt.legend(loc = 'best')

plt.show()
model = SVC(kernel='rbf', random_state=1, gamma=0.1, C = 10.0)

model.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, model)

plt.legend(loc = 'upper left')

plt.show()
model = SVC(kernel='rbf', random_state=1, gamma=10, C = 10.0)

model.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, model)

plt.legend(loc = 'best')

plt.show()
#visualizing different impurity measures for suppose class 1 in binary class classification p is the probaility of class 1, 1-p is the probaility of other class.



def gini(p):

    return 2 * ( p - p ** 2)



def entropy(p):

    return -p * np.log2(p) - (1 - p) * np.log2(1-p)



def error(p):

    return 1 - np.max([p,1-p])

x =  np.arange(0.0, 1.0, 0.01) #probability have range 0 to 1 inclusive.



ent = [entropy(p) if p!=0 else None for p in x]

sc_ent = [e * 0.5 if e else None for e in ent] #scaled entropy

err = [error(i) for i in x]

fig = plt.figure()

ax = plt.subplot(111)



for i, lab, ls, c in zip([ent,sc_ent,gini(x),err],['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],['-','-','--','-.'],['black','lightgray','red','green','cyan']):

    line = ax.plot(x,i,label = lab, linestyle = ls, lw = 2, color = c)

ax.legend(loc = 'upper center', ncol= 5, bbox_to_anchor = [0.5,1.15])

ax.axhline(y = 0.5, linewidth = 1, color = 'k', linestyle = '--')

ax.axhline(y = 1, linewidth = 1, color = 'k', linestyle = '--')

plt.ylim([0,1.1])

plt.xlabel('p(i=1)')

plt.ylabel('Impurity Index')

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
model = DecisionTreeClassifier(

    criterion='gini',

    max_depth=4,

    random_state=1

)
model.fit(X_train,y_train)
X_combined = np.vstack([X_train,X_test])

y_combined = np.hstack([y_train,y_test])
plot_decision_regions(X_combined,y_combined,model, range(105,150))

plt.xlabel('petal length [cm]')

plt.ylabel('petal width [cm]')

plt.legend(loc = 'upper left')

plt.show()
fig = plt.figure(figsize=  (10,15))

tree.plot_tree(

    model,

    rounded=True,

    filled=True,

    feature_names=['petal length [cm]', 'petal width [cm]'],

    class_names= ['Setosa', 'Versicolor', 'Virginica'],

    fontsize=12,

    impurity=True



)



plt.savefig('decision_tree_1.png')
#Let's change our depth to 3

model = DecisionTreeClassifier(

        criterion='gini',

        max_depth=3,

        random_state=1

        

)
model.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined, model, range(105,150))
fig = plt.figure(figsize=(10,15))

tree.plot_tree(

    model,

    feature_names=['petal length [cm]', 'petal width [cm]'],

    class_names=['Setosa','versicolor','Virginica'],

    rounded = True,

    filled=True,

    fontsize=12

)

plt.show()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(

        criterion='gini',

        n_estimators=25,

        random_state=1,

        n_jobs=-1

)
forest.fit(X_train,y_train)
plot_decision_regions(X_combined, y_combined, forest, range(105,150))

plt.xlabel('petal length')

plt.ylabel('petal width')

plt.legend(loc = 'best')

plt.show()
fig = plt.figure(figsize=(10,10))



for i in range(len(forest.estimators_)):

    plt.subplot(5,5,i+1)

    tree.plot_tree(

            forest.estimators_[i],

            rounded = True,

            filled = True

    )

plt.savefig('RandomForest.png')
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(

        n_neighbors=5,

        p = 2,

        metric= 'minkowski'

)
model.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, model, range(105,150))

plt.xlabel('petal length standarized')

plt.ylabel('petal width standarized')

plt.legend(loc = 'upper left')

plt.show()