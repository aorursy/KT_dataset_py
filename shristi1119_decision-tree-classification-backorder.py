#!conda install --yes python-graphviz
import pandas as pd

import numpy as np



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, auc



from sklearn import tree





from sklearn.model_selection import GridSearchCV



import graphviz

#!conda install --yes python-graphviz

#!conda install --yes graphviz

#import pydotplus

#import pydot

#from PIL import Image



import matplotlib.pyplot as plt
data=pd.read_csv("BackOrders.csv",header=0)
data.shape
data.head()
data.went_on_backorder.value_counts()
data.columns
data.dtypes
data.describe(include='all')
for col in ['sku', 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']:

    data[col]=data[col].astype('category')
data.dtypes
np.size(np.unique(data.sku, return_counts=True)[0])
#data.drop('sku',axis=1,inplace=True)

data.shape
data.isnull().sum()
print (data.shape)
#Since the number of missing values is about 5%. For 

#initial analysis we ignore all these records

data.dropna(axis=0,inplace=True)


print(data.shape)
print (data.columns, data.shape)
categorical_Attributes=data.select_dtypes(include=['category']).columns
data=pd.get_dummies(columns=categorical_Attributes,data=data,prefix=categorical_Attributes,prefix_sep="-",drop_first=True)

data.dtypes

print (data.columns, data.shape)
data.dtypes
data['went_on_backorder-Yes'].value_counts()
#Performing train test split on the data

X,y=data.iloc[:,data.columns!='went_on_backorder-Yes'].values,data.loc[:,'went_on_backorder-Yes'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=123)

#To get the distribution in the target in train and test

print(pd.value_counts(y_train))

print(pd.value_counts(y_test))


clf=tree.DecisionTreeClassifier(max_depth=3)

clf=clf.fit(X_train,y_train)
clf.feature_importances_
data.columns
features = data.columns

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

pd.DataFrame([data.columns[indices],np.sort(importances)[::-1]])
importances
plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='black')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
train_pred=clf.predict(X_train)

test_pred=clf.predict(X_test)

print(train_pred[:5])

print(test_pred[:5])
# Set up Train and Test Confusion Matrices

confusion_matrix_test=confusion_matrix(y_test,test_pred)

confusion_matrix_train=confusion_matrix(y_train,train_pred)





print(confusion_matrix_train)

print(confusion_matrix_test)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])

TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])



print("Train TNR: ",TNR_Train)

print("Train TPR: ",TPR_Train)

print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])

TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])



print("Test TNR: ",TNR_Test)

print("Test TPR: ",TPR_Test)

print("Test Accuracy: ",Accuracy_Test)
indices
# Rebuild Model with the top 10 important variables

train_pred = clf.predict(X_train[:,select])

test_pred = clf.predict(X_test[:,select])
print(train_pred[:5])

print(test_pred[:5])
confusion_matrix_test = confusion_matrix(y_test, test_pred)

confusion_matrix_train = confusion_matrix(y_train, train_pred)



print(confusion_matrix_train)

print(confusion_matrix_test)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])

TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])



print("Train TNR: ",TNR_Train)

print("Train TPR: ",TPR_Train)

print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])

TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])



print("Test TNR: ",TNR_Test)

print("Test TPR: ",TPR_Test)

print("Test Accuracy: ",Accuracy_Test)
max_depths = np.linspace(1, 32, 32, endpoint=True)



train_results = []

test_results = []



for max_depth in max_depths:

    dt = tree.DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train, y_train)

    

    train_pred = dt.predict(X_train)

    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

    train_results.append(Accuracy_Train)

    

    test_pred = dt.predict(X_test)

    confusion_matrix_test = confusion_matrix(y_test, test_pred)

    Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

    test_results.append(Accuracy_Test)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results,'b', label='Train Accuracy')

line2, = plt.plot(max_depths, test_results,'r', label='Test Accuracy')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy')

plt.xlabel('Tree depth')

plt.show()
# set of parameters to test

param_grid = {"criterion": ["gini", "entropy"],

              "min_samples_split": [2, 10, 20],

              "max_depth": [None, 2, 5, 10],

              "min_samples_leaf": [1, 5, 10],

              "max_leaf_nodes": [None, 5, 10, 20],

              }
# setup GridSearchCV and fit model

dt=tree.DecisionTreeClassifier()

clf2=GridSearchCV(dt,param_grid,cv=10)

clf2.fit(X_train,y_train)

ss

clf2.best_params_
clf2.best_estimator_
train_pred = clf2.predict(X_train)

test_pred = clf2.predict(X_test)
confusion_matrix_test = confusion_matrix(y_test, test_pred)

confusion_matrix_train = confusion_matrix(y_train, train_pred)



Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])

TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])



print("Train TNR: ",TNR_Train)

print("Train TPR: ",TPR_Train)

print("Train Accuracy: ",Accuracy_Train)



Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])

TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])



print("Test TNR: ",TNR_Test)

print("Test TPR: ",TPR_Test)

print("Test Accuracy: ",Accuracy_Test)
dot_data = tree.export_graphviz(clf2.best_estimator_, out_file=None, 

                                feature_names=data.drop(['went_on_backorder_Yes'], axis = 1).columns,

                                class_names=['No','Yes'], 

                                filled=True, rounded=True, special_characters=True) 

graph = graphviz.Source(dot_data) 

graph

graph.render("back_orders") 