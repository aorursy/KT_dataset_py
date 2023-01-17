import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.naive_bayes import GaussianNB

import xgboost

from xgboost import XGBClassifier, plot_importance, plot_tree

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, chi2, RFE
address = '/kaggle/input/mushroom-classification/mushrooms.csv'

mushroom_data = pd.read_csv(address) 

mushroom_data.head()
mushroom_data.isna().sum()
X = mushroom_data.drop('class',axis=1)

y = mushroom_data['class'].values

X = pd.get_dummies(X, drop_first=True)

y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, shuffle=True)
X_train.head()
model = LogisticRegression()

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(accuracy_score(preds, np.ravel(y_test)))

print(confusion_matrix(preds, y_test))
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train,

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)],

             verbose=True)

xgb_preds=xgb_model.predict(X_test)

confusion_matrix(xgb_preds, y_test)
perm = PermutationImportance(xgb_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plot_importance(xgb_model, importance_type='weight')

fig, ax = plt.subplots(figsize=(30, 30))

xgboost.plot_tree(xgb_model, ax=ax)

plt.show()
nb_model = GaussianNB()

nb_model.fit(X_train, np.ravel(y_train))

preds = nb_model.predict(X_test)

confusion_matrix(preds, y_test)
perm = PermutationImportance(nb_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)

preds = tree_model.predict(X_test)

confusion_matrix(preds, y_test)
perm = PermutationImportance(tree_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(5,5))

plot_tree(tree_model, feature_names=X_test.columns.tolist())

plt.show()
svm_model = SVC()

svm_model.fit(X_train, np.ravel(y_train))

preds = svm_model.predict(X_test)

confusion_matrix(preds, y_test)
#perm = PermutationImportance(svm_model, random_state=1).fit(X_test, y_test)

#eli5.show_weights(perm, feature_names = X_test.columns.tolist())
per_model = Perceptron()

per_model.fit(X_train, np.ravel(y_train))

preds = per_model.predict(X_test)

confusion_matrix(preds, y_test)
perm = PermutationImportance(per_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
ridge_model = RidgeClassifier()

ridge_model.fit(X_train, np.ravel(y_train))

preds = ridge_model.predict(X_test)

confusion_matrix(preds, y_test)
sgd_model = SGDClassifier()

sgd_model.fit(X_train, np.ravel(y_train))

preds = sgd_model.predict(X_test)

confusion_matrix(preds, y_test)
cvs = cross_val_score(sgd_model, X_test, np.ravel(y_test), cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca_fitted = pca.fit_transform(X_train)

pca_fitted
plt.scatter(pca_fitted[:,0],pca_fitted[:,1], c=np.ravel(y_train))
tree_model = DecisionTreeClassifier()

selector = RFE(tree_model, n_features_to_select=10, step=1)

selector = selector.fit(X_train, np.ravel(y_train))

selector.support_

print('Tree model columns: ',X_train.columns[[i for i in selector.support_==True]])

sgd_model = SGDClassifier()

selector = RFE(sgd_model, n_features_to_select=10, step=1)

selector = selector.fit(X_train, np.ravel(y_train))

selector.support_

print('SGD model columns: ',X_train.columns[[i for i in selector.support_==True]])

per_model = Perceptron()

selector = RFE(per_model, n_features_to_select=10, step=1)

selector = selector.fit(X_train, np.ravel(y_train))

print('Perceptron model columns: ',X_train.columns[[i for i in selector.support_==True]])

reduced_X_train = X_train.drop(X_train.columns[[i for i in selector.support_==False]], axis=1)

reduced_X_test = X_test.drop(X_train.columns[[i for i in selector.support_==False]], axis=1)

reduced_X_train.head()
sgd_model = SGDClassifier()

sgd_model.fit(reduced_X_train, np.ravel(y_train))

preds = sgd_model.predict(reduced_X_test)

print(confusion_matrix(preds, y_test))

cvs = cross_val_score(sgd_model, reduced_X_test, np.ravel(y_test), cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
tree_model = DecisionTreeClassifier()

tree_model.fit(reduced_X_train, np.ravel(y_train))

preds = tree_model.predict(reduced_X_test)

print(confusion_matrix(preds, y_test))

cvs = cross_val_score(tree_model, reduced_X_test, np.ravel(y_test), cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
red_pca = PCA(n_components=3)

red_pca_fitted = red_pca.fit_transform(reduced_X_train)

plt.scatter(red_pca_fitted[:,0],red_pca_fitted[:,1],c=np.ravel(y_train))

plt.scatter(red_pca_fitted[:,0],red_pca_fitted[:,2],c=np.ravel(y_train))
X = mushroom_data.drop('class',axis=1)

y = mushroom_data['class'].values

X = pd.get_dummies(X, drop_first=True)

y = pd.get_dummies(y, drop_first=True)

X_new = SelectKBest(chi2, k=5).fit_transform(X, y)



X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.33, shuffle=True)
tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, np.ravel(y_train))

preds = tree_model.predict(X_test)

print(confusion_matrix(preds, y_test))

cvs = cross_val_score(tree_model, X_test, np.ravel(y_test), cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))