import numpy as np
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
print(os.listdir("../input"))
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target
tree = DecisionTreeClassifier(max_depth = 2, random_state = 0)
tree.fit(X,y)
dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = wine.feature_names,
                class_names=wine.target_names,
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph
wine = load_wine()
X = wine.data[:,[6,12]] # flavanoids and proline
y = wine.target

# random_state is set to guarantee consistent result. You should remove it when running your own code.
tree1 = DecisionTreeClassifier(random_state=5) 
tree1.fit(X,y)
# preparing to plot the decision boundaries
x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))
Z = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
plt.subplots(figsize=(12,10))
plt.contourf(xx0, xx1, Z, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
plt.legend(fontsize=18)
plt.xlabel('flavanoids', fontsize = 18)
plt.ylabel('proline', fontsize = 18)
plt.show()
# limit maximum tree depth
tree1 = DecisionTreeClassifier(max_depth=3,random_state=5) 
tree1.fit(X,y)

# limit maximum number of leaf nodes
tree2 = DecisionTreeClassifier(max_leaf_nodes=4,random_state=5) 
tree2.fit(X,y)

x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))

Z1 = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z1 = Z1.reshape(xx0.shape)
Z2 = tree2.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z2 = Z2.reshape(xx0.shape)

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
ax[0].contourf(xx0, xx1, Z1, cmap=plt.cm.RdYlBu)
ax[1].contourf(xx0, xx1, Z2, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    ax[0].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
    ax[1].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
ax[0].legend(fontsize=14)
ax[0].set_xlabel('flavanoids', fontsize = 18)
ax[0].set_ylabel('proline', fontsize = 18)
ax[0].set_ylim(260,1690)
ax[0].set_title('max_depth = 3', fontsize = 14)
ax[1].legend(fontsize=14)
ax[1].set_xlabel('flavanoids', fontsize = 18)
ax[1].set_ylabel('proline', fontsize = 18)
ax[1].set_ylim(260,1690)
ax[1].set_title('max_leaf_nodes = 4', fontsize = 14)
plt.show()
wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()
wineData['category'] = wineData['quality'] >= 7
wineData.head()
X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7],'min_samples_leaf':[1,10,100],'random_state':[14]} 
# random_state is only to ensure repeatable results. You can remove it when running your own code.

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline
plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 2, random_state = 5) # max tree depth is limited to 2
tree.fit(x,y)
dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = ['flavanoids'],
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph
xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 3, random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10],'random_state':[14]} 
# random_state is only to ensure repeatable results. You can remove it when running your own code.

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline

plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
reg = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_depth=2, random_state = 5)
x = x.reshape(-1,1)
reg.fit(x,y)
xx = np.arange(0,5,0.02).reshape(-1,1)
yhat = reg.predict(xx)
plt.plot(xx,yhat,color='red')
plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()