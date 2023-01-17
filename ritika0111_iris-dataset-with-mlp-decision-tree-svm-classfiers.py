%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#iris = pd.read_csv("C:\\Users\\user\\Downloads\\Iris.csv")


from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)
x = iris.data
y = iris.target
features = iris.feature_names
target = iris.target_names

print("Feature Names:",features)
print("-"*100)
print("Target Names:", target)
print("-"*100)
print("data:", x[:10])
print("-"*100)
df = pd.DataFrame(x, columns=iris.feature_names)
df['target'] = iris.target
df
df_norm = df[iris.feature_names].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)
df_norm.describe()
df.sample(n=5)
df = pd.concat([df_norm, df['target']], axis=1)
df.sample(n=5)
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics

train, test = train_test_split(df, test_size = 0.3)
trainX = train[features]# taking the training data features
trainY=train.target# output of our training data
testX= test[features] # taking test data features
testY =test.target   #output value of test data
trainX.head(5)
trainY.head(5)
testX.head(5)
testY.head(5)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
clf.fit(trainX, trainY)
prediction = clf.predict(testX)
print(prediction)
print(testY.values)
print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size=0.4, random_state=17)
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import graphviz

clf = tree.DecisionTreeClassifier(random_state=17)
clf = clf.fit(X_train, y_train)
clf = tree.DecisionTreeClassifier(random_state=17)
clf = clf.fit(X_train, y_train)
 
y_pred = clf.predict(X_test)

data = df.drop(columns="target")
print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
from sklearn import tree
from graphviz import Source
import pandas as pd

import graphviz
feat = data.columns 
Source(tree.export_graphviz(clf, out_file = None, feature_names = feat,max_depth=4))
X_train, X_test, y_train, y_test = train_test_split( df[iris.feature_names], df['target'],  test_size=0.2, random_state=20)
from sklearn.svm import LinearSVC

clf = LinearSVC(penalty='l2', loss='squared_hinge',
                dual=True, tol=0.0001, C=100, multi_class='ovr',
                fit_intercept=True, intercept_scaling=1, class_weight=None,verbose=0
                , random_state=0, max_iter=1000)
clf.fit(X_train,y_train)

print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))

print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) *100))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
    
c = np.logspace(start = -15, stop = 1000, base = 1.02)
param_grid = {'C': c}


grid = GridSearchCV(clf, param_grid =param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)
  
print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100 ))
print( "Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100 ) )
from sklearn.svm import SVC

clf_SVC = SVC(C=100.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
          probability=False, tol=0.001, cache_size=200, class_weight=None, 
          verbose=0, max_iter=-1, decision_function_shape="ovr", random_state = 0)
clf_SVC.fit(X_train,y_train)

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))

print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

from sklearn.metrics import classification_report
import numpy as np
    
c_SVC = np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64')
print( 'the generated array of c values')
print ( c_SVC )
param_grid_S = {'C': c_SVC}



print("\n Array of means \n")
clf = GridSearchCV(clf_SVC, param_grid =param_grid_S, cv=20 , scoring='accuracy')
clf.fit(X_train, y_train)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
print(means)

y_true, y_pred = y_test, clf.predict(X_test)
print( '\nClassification report\n' )
print(classification_report(y_true, y_pred))

from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["SVM","Train Score","Test Score"]
x.add_row(["LinearSVC",97.5,90.00])
x.add_row(["SVC",97.50,96.67])

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
printmd('****Final Conclusion:****')
print(x)