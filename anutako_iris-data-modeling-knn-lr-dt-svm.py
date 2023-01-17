# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Iris.csv')
print(dataset.head(5))
dataset = dataset.drop(columns = 'Id')

feature_names = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'class']
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

dataset.columns = feature_names
#Get to know the data set
print(dataset.shape)
dataset.describe()
dataset.groupby('class').size()
#Learn about different classes by looking at mean values of data across the classes:
dataset.groupby('class').mean()
dataset.groupby('class').std()
#Draw a colored scatter plot matrix to visualize the data
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 9,6

import seaborn as sns
sns.set(style = 'ticks')

sns.pairplot(dataset, hue = 'class')
plt.show()
from sklearn import model_selection
X = dataset.values[:,:4]
Y = dataset.values[:,4]
validation_size = 0.30
seed = 7
np.random.seed(seed)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# The code in this cell intends to implement scaling of the features. However,
# scaling is not necessary for some methods (like Decision Tree, for example), and 
# I find that scaling does not help improve the accuracy of the models where scaling matters.
# There is no benefit to scaling because the mean values and stdandard deviations 
# for the factors are in a similar range (all within 10).

#from sklearn.preprocessing import StandardScaler  
#scaler = StandardScaler()  

#X_train = scaler.fit_transform(X_train)  
#X_test = scaler.transform(X_test) 
# box and whisker plots
plt.rcParams['figure.figsize'] = 9,6
plt.style.use('fivethirtyeight')
dataset.plot(kind='box', sharex=False, sharey=False)
plt.show()
from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier(n_neighbors = 15, weights = 'distance', p = 3)

from sklearn.linear_model import LogisticRegression
model_LR =  LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier(min_samples_leaf = 3, random_state = seed)

from sklearn import svm
model_SVC = svm.SVC(gamma = 'auto')

from sklearn.metrics import classification_report, confusion_matrix  

models = ['KNN','LR','DT', 'SVC']
num_models = len(models)

#parameters for cross-valiations:
scoring = 'accuracy'
n_splits = 10
kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
cv_results = np.zeros((num_models,n_splits))

for i in range(num_models):
    model = eval('model_'+models[i])
    cv_results[i,:] = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    
print('\n Accuracy: Mean (std): \n')
for i in range(num_models):
    print_results = "%s: %.4f (%.4f)" % (models[i], cv_results[i,:].mean(), cv_results[i,:].std())
    print(print_results)
#KNN model

model_KNN.fit(X_train,Y_train)
Y_pred = model_KNN.predict(X_test)

print('1) Confusion Matrix for KNN Model:')
confusion_mat = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
from IPython.display import display
display(confusion_mat)

from sklearn.metrics import classification_report
cr = classification_report(Y_test, Y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
print('2) Classification report:\n',cr)

accuracy_train = model_KNN.score(X_train,Y_train)
accuracy_test = model_KNN.score(X_test,Y_test)
print('3) Accuracy on train and test datasets:', "%.4f and %.4f" % (accuracy_train, accuracy_test))

#SVC model

model_SVC.fit(X_train,Y_train)
Y_pred = model_SVC.predict(X_test)

print('1) Confusion Matrix for SVC Model:')
confusion_mat = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
from IPython.display import display
display(confusion_mat)

from sklearn.metrics import classification_report
cr = classification_report(Y_test, Y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
print('2) Classification report:\n',cr)

accuracy_train = model_SVC.score(X_train,Y_train)
accuracy_test = model_SVC.score(X_test,Y_test)
print('3) Accuracy on train and test datasets:', "%.4f and %.4f" % (accuracy_train, accuracy_test))

print('\n SVC Model with sepal length and width features only: \n')
features= [0,1]
cv_results = model_selection.cross_val_score(model_SVC,X_train[:,features], Y_train, cv=kfold, scoring=scoring)
print_results = "%.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print('Accuracy: Mean (std):')
print(print_results)

print('\n SVC Model with petal length and width features only: \n')
features= [2,3]
cv_results = model_selection.cross_val_score(model_SVC,X_train[:,features], Y_train, cv=kfold, scoring=scoring)
print_results = "%.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print('Accuracy: Mean (std):')
print(print_results)

print('\n SVC Model with petal length feature only: \n')
features= [2]
cv_results = model_selection.cross_val_score(model_SVC,X_train[:,features], Y_train, cv=kfold, scoring=scoring)
print_results = "%.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print('Accuracy: Mean (std):')
print(print_results)

print('\n SVC Model with petal  width feature only: \n')
features= [3]
cv_results = model_selection.cross_val_score(model_SVC,X_train[:,features], Y_train, cv=kfold, scoring=scoring)
print_results = "%.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print('Accuracy: Mean (std):')
print(print_results)
#SVC model, reduced set of features

features_select = [2,3]
X_feat = X_train[:,features_select]

model_SVC.fit(X_feat,Y_train)
Y_pred = model_SVC.predict(X_test[:,features_select])

print('1) Confusion Matrix for SVC Model with petal length and width features only:')
confusion_mat = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_mat)

cr = classification_report(Y_test, Y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
print('2) Classification report:\n',cr)

accuracy_train = model_SVC.score(X_feat,Y_train)
accuracy_test = model_SVC.score(X_test[:,features_select],Y_test)
print('3) Accuracy on train and test datasets:', "%.4f and %.4f" % (accuracy_train, accuracy_test))
# Plot decision regions for the model with reduced number of features
x_min, x_max = X_feat[:, 0].min()*0.1, X_feat[:, 0].max()*1.2
y_min, y_max = X_feat[:, 1].min()*0.1, X_feat[:, 1].max()*1.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z_SVC = model_SVC.predict(np.c_[xx.ravel(), yy.ravel()])
Z_SVC = pd.get_dummies(Z_SVC)
Z_SVC = Z_SVC.values.argmax(1)
Z_SVC = Z_SVC.reshape(xx.shape)

class_setosa = dataset[dataset['class'] == 'Iris-setosa'].values[:,features_select]
class_versic = dataset[dataset['class'] == 'Iris-versicolor'].values[:,features_select]
class_virgin = dataset[dataset['class'] == 'Iris-virginica'].values[:,features_select]

data_by_class = (class_setosa, class_versic, class_virgin)
colors = ('green','red', 'black')

fig = plt.figure()
ax = plt.gca()
plt.contourf(xx, yy, Z_SVC, alpha=0.2,cmap=plt.cm.Dark2)
for data, color, group in zip(data_by_class, colors, class_labels):
    x,y = data[:,0],data[:,1]
    ax.scatter(x, y, alpha=0.8, c=color,  s=30, label=group)
plt.legend(loc = 4, fontsize = 'small')
plt.title('SVM with 2 features')
plt.xlabel(feature_names[features_select[0]])
plt.ylabel(feature_names[features_select[1]])
plt.show()
model = model_KNN
model.fit(X_train[:,features_select],Y_train)

Z_KNN = model_KNN.predict(np.c_[xx.ravel(), yy.ravel()])
Z_KNN = pd.get_dummies(Z_KNN)
Z_KNN = Z_KNN.values.argmax(1)
Z_KNN = Z_KNN.reshape(xx.shape)

class_setosa = dataset[dataset['class'] == 'Iris-setosa'].values[:,features_select]
class_versic = dataset[dataset['class'] == 'Iris-versicolor'].values[:,features_select]
class_virgin = dataset[dataset['class'] == 'Iris-virginica'].values[:,features_select]

data_by_class = (class_setosa, class_versic, class_virgin)
colors = ('green','red', 'black')

fig = plt.figure()
ax = plt.gca()
plt.contourf(xx, yy, Z_KNN, alpha=0.2,cmap=plt.cm.Dark2)
for data, color, group in zip(data_by_class, colors, class_labels):
    x,y = data[:,0],data[:,1]
    ax.scatter(x, y, alpha=0.8, c=color,  s=30, label=group)
plt.legend(loc = 4, fontsize = 'small')
plt.title('KNN with 2 features')
plt.xlabel(feature_names[features_select[0]])
plt.ylabel(feature_names[features_select[1]])
plt.show()
# 1) Demonstrate that default choice for parameter C (Penalty of the error term, C = 1) 
# provides the best model outcome. Note that linear kernel is also chosen as best-performing.

ind = [1, 2, 5, 10, 50, 100]
num_models = len(ind)
cv_results = np.zeros((num_models,n_splits))

count = 0
for i in ind:
    model = svm.SVC(kernel = 'linear', C = i)
    cv_results[count,:] = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    count +=1
print('Varying parameter C\n')
print('Accuracy: Mean (std): \n')
for i in range(num_models):
    print_results = "%.0f: %.4f (%.4f)" % (ind[i], cv_results[i,:].mean(), cv_results[i,:].std())
    print(print_results)
# KNN model
ind = [ 2, 3, 5, 7, 10, 15,20,30, 50]
num_models = len(ind)
cv_results = np.zeros((num_models,n_splits))

count = 0
for i in ind:
    model = KNeighborsClassifier(n_neighbors = i, weights = 'distance', p = 3)
    cv_results[count,:] = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    count +=1
print('KNN: Varying the number of neighbors \n')

print('Accuracy: Mean (std): \n')
for i in range(num_models):
    print_results = "%.0f: %.4f (%.4f)" % (ind[i], cv_results[i,:].mean(), cv_results[i,:].std())
    print(print_results)
# Decision Tree Model
ind = [ 2, 3, 5, 7, 10, 15,20,30, 50]
num_models = len(ind)
cv_results = np.zeros((num_models,n_splits))

count = 0
for i in ind:
    model = DecisionTreeClassifier(min_samples_leaf = i,  random_state=seed)
    cv_results[count,:] = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    count +=1
print('DT: varying min_samples_leaf')
print('Accuracy: Mean (std): \n')
for i in range(num_models):
    print_results = "%.0f: %.4f (%.4f)" % (ind[i], cv_results[i,:].mean(), cv_results[i,:].std())
    print(print_results)   
# Logisic Regression - try with polynomial features, got only a small improvement when 
# polynomial terms are included
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias = False)
XX_train = poly.fit_transform(X_train)

model = LogisticRegression(random_state = seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

print('LR: adding non-linear terms')
print('Accuracy: Mean (std): \n')

print_results = " %.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print(print_results)  

model =  LogisticRegression(random_state = seed)
cv_results = model_selection.cross_val_score(model, XX_train, Y_train, cv=kfold, scoring=scoring)
print_results = " %.4f (%.4f)" % (cv_results.mean(), cv_results.std())
print(print_results)  