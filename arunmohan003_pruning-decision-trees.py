import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import tree

from sklearn.metrics import accuracy_score,confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt
data = '/kaggle/input/heart-disease-uci/heart.csv'

df = pd.read_csv(data)

df.head()
X = df.drop(columns=['target'])

y = df['target']

print(X.shape)

print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

print(x_train.shape)

print(x_test.shape)
clf = tree.DecisionTreeClassifier(random_state=0)

clf.fit(x_train,y_train)

y_train_pred = clf.predict(x_train)

y_test_pred = clf.predict(x_test)
plt.figure(figsize=(20,20))

features = df.columns

classes = ['Not heart disease','heart disease']

tree.plot_tree(clf,feature_names=features,class_names=classes,filled=True)

plt.show()
# helper function

def plot_confusionmatrix(y_train_pred,y_train,dom):

    print(f'{dom} Confusion matrix')

    cf = confusion_matrix(y_train_pred,y_train)

    sns.heatmap(cf,annot=True,yticklabels=classes

               ,xticklabels=classes,cmap='Blues', fmt='g')

    plt.tight_layout()

    plt.show()

    
print(f'Train score {accuracy_score(y_train_pred,y_train)}')

print(f'Test score {accuracy_score(y_test_pred,y_test)}')

plot_confusionmatrix(y_train_pred,y_train,dom='Train')

plot_confusionmatrix(y_test_pred,y_test,dom='Test')
params = {'max_depth': [2,4,6,8,10,12],

         'min_samples_split': [2,3,4],

         'min_samples_leaf': [1,2]}



clf = tree.DecisionTreeClassifier()

gcv = GridSearchCV(estimator=clf,param_grid=params)

gcv.fit(x_train,y_train)

model = gcv.best_estimator_

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print(f'Train score {accuracy_score(y_train_pred,y_train)}')

print(f'Test score {accuracy_score(y_test_pred,y_test)}')

plot_confusionmatrix(y_train_pred,y_train,dom='Train')

plot_confusionmatrix(y_test_pred,y_test,dom='Test')
plt.figure(figsize=(20,20))

features = df.columns

classes = ['Not heart disease','heart disease']

tree.plot_tree(model,feature_names=features,class_names=classes,filled=True)

plt.show()
path = clf.cost_complexity_pruning_path(x_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(ccp_alphas)
# For each alpha we will append our model to a list

clfs = []

for ccp_alpha in ccp_alphas:

    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

    clf.fit(x_train, y_train)

    clfs.append(clf)
clfs = clfs[:-1]

ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]

depth = [clf.tree_.max_depth for clf in clfs]

plt.scatter(ccp_alphas,node_counts)

plt.scatter(ccp_alphas,depth)

plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")

plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")

plt.legend()

plt.show()
train_acc = []

test_acc = []

for c in clfs:

    y_train_pred = c.predict(x_train)

    y_test_pred = c.predict(x_test)

    train_acc.append(accuracy_score(y_train_pred,y_train))

    test_acc.append(accuracy_score(y_test_pred,y_test))



plt.scatter(ccp_alphas,train_acc)

plt.scatter(ccp_alphas,test_acc)

plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")

plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")

plt.legend()

plt.title('Accuracy vs alpha')

plt.show()
clf_ = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.020)

clf_.fit(x_train,y_train)

y_train_pred = clf_.predict(x_train)

y_test_pred = clf_.predict(x_test)



print(f'Train score {accuracy_score(y_train_pred,y_train)}')

print(f'Test score {accuracy_score(y_test_pred,y_test)}')

plot_confusionmatrix(y_train_pred,y_train,dom='Train')

plot_confusionmatrix(y_test_pred,y_test,dom='Test')
plt.figure(figsize=(20,20))

features = df.columns

classes = ['Not heart disease','heart disease']

tree.plot_tree(clf_,feature_names=features,class_names=classes,filled=True)

plt.show()