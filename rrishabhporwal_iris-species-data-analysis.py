import numpy
import scipy
import matplotlib
import pandas
import sklearn

print('numpy: {}'.format(numpy.__version__))
print('scipy: {}'.format(scipy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


import os
print(os.listdir("../input"))
# Importing Packages
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Load Datasets
url = '../input/Iris.csv'
names = ['sepal-length','sapel-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names=names)
#Shape
dataset.shape
#Head
dataset.head(20)

#Distrubution (class)
print(dataset.groupby('class').size())
#histograms
dataset.hist()
plt.show()
#scatter plot matrix
scatter_matrix(dataset)
plt.show()
#Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=0.2, random_state=7)
#Testing option and evalution matrix
scoring = 'accuracy'
#Initiallizing models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
print(models)
# Evaluating each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = 7)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Make some prediction on validation dataset
for name, model in models:
    model.fit(X_train,Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
