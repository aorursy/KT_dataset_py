import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
lbc = load_breast_cancer()
print(lbc.keys())
print(lbc['DESCR'])
dataset = pd.DataFrame(np.c_[lbc['data'], lbc['target']], columns=np.append(lbc['feature_names'], ['target']))
dataset.tail()
corr = dataset.corr()
plt.figure(figsize = (25,15))
sns.heatmap(corr, annot=True, cmap="YlGnBu")
sns.pairplot(dataset, vars=['mean radius', 'mean texture', 'mean perimeter'], hue='target')
sns.countplot(dataset['target'])
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, plot_confusion_matrix
y_pred = classifier.predict(X_test)
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=lbc.target_names,
                             cmap=plt.cm.Blues)
print(classification_report(y_test, y_pred))
# Apply StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, plot_confusion_matrix
y_pred = classifier.predict(X_test)
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=lbc.target_names,
                             cmap=plt.cm.Blues)
print(classification_report(y_test, y_pred))
# Apply GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {
    'C':[0.1, 1, 10, 100],
    'gamma':[1, 0.1, 0.01, 0.001],
    'kernel':['rbf']
}
grid = GridSearchCV(SVC(), params, refit=True, verbose=4)
grid.fit(X_train, y_train)
# Best Parameters
print(grid.best_params_)
# Applying best params
from sklearn.svm import SVC
classifier = SVC(C=10, gamma=0.01, kernel='rbf')
classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, plot_confusion_matrix
y_pred = classifier.predict(X_test)
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=lbc.target_names,
                             cmap=plt.cm.Blues)
print(classification_report(y_test, y_pred))