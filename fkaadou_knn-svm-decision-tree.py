import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split,GridSearchCV, KFold

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import confusion_matrix

import graphviz

import pandas as pd
#import data into pandas dataframe

data = pd.read_csv("../input/Iris.csv")



#display first 5 lines

display(data.head())



display(data.describe(percentiles=[]))



sns.countplot(x=data['Species'])



#print data properties

print('\nData Shape: {}'.format(data.shape))



for column in data:

    if data[column].dtype == np.float64:

        print('{} range: [{}, {}]'.format(column, 

              data[column].min(), data[column].max()))

print('\n')

for spec in np.unique(data['Species']):

    print("{} : {}".format(spec, 

          data.loc[data['Species']==spec,'Species'].agg(['count'][0])))

        
#Extract data and label target

X = data.iloc[:,1:5]

target = data.iloc[:,5]

le = LabelEncoder()

le.fit(target)

y = le.transform(target)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=4)

sns.pairplot(data.iloc[:,1:6], hue = "Species",diag_kind='kde')
#Feature correlations

corr_mat=sns.heatmap(X.corr(method='spearman'),annot=True,cbar=True,

            cmap='viridis', vmax=1,vmin=-1,

            xticklabels=X.columns,yticklabels=X.columns)

corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)

acc_train=[]

acc_test=[]

for i in range(1,11):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    acc_train.append(knn.score(X_train,y_train))

    acc_test.append(knn.score(X_test,y_test))

plt.figure()

plt.plot(range(1,11), acc_train, label='training accuracy')

plt.plot(range(1,11), acc_test,label='test accuracy')

plt.legend()

plt.show()
kfold = KFold(n_splits=3, shuffle=True, random_state=4)

param_grid = {'n_neighbors': [1,2,3,4,5]}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold)

grid_knn.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid_knn.best_score_))

print("Best parameters: {}".format(grid_knn.best_params_))

print("Test set score: {:.2f}".format(grid_knn.score(X_test,y_test)))

#KNN Cofusion matrix

param = grid_knn.best_params_

knn_best = KNeighborsClassifier(n_neighbors = param["n_neighbors"])

knn_best.fit(X_train,y_train)

conf_mat_knn = confusion_matrix(y_test, knn_best.predict(X_test))

plt.figure()

sns.heatmap(conf_mat_knn, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=le.classes_, xticklabels=le.classes_)

plt.show()



#print misclassified points

mc_pnts_ind = X_test.iloc[knn_best.predict(X_test) != y_test, :].index.tolist()

print("\nMisclassified points: ")

for ind in range(len(mc_pnts_ind)):

    display(data.loc[data['Id']==(mc_pnts_ind[ind]+1)])
#Pipline: scaling + SVM parameter grid search

pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",SVC(kernel='rbf'))])

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],

              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_svm = GridSearchCV(pipe, param_grid=param_grid, cv=kfold)

grid_svm.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid_svm.best_score_))

print("Test set score: {:.2f}".format(grid_svm.score(X_test,y_test)))

print("Best parameters: {}".format(grid_svm.best_params_))





#SVM Cofusion matrix

param = grid_svm.best_params_

svm_best = SVC(gamma = param["svm__gamma"], C = param["svm__C"])

svm_best.fit(X_train,y_train)

conf_mat_svm = confusion_matrix(y_test, svm_best.predict(X_test))

sns.heatmap(conf_mat_svm, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=le.classes_, xticklabels=le.classes_)



#print misclassified points

mc_pnts_ind = X_test.iloc[svm_best.predict(X_test) != y_test, :].index.tolist()

print("\nMisclassified points: ")

for ind in range(len(mc_pnts_ind)):

    display(data.loc[data['Id']==(mc_pnts_ind[ind]+1)])
#Decision tree

tree = DecisionTreeClassifier(max_depth=3, random_state=0)

tree.fit(X_train, y_train)

print("Decison Tree test set accuracy: {:.2f}".format(tree.score(X_test,y_test)))



export_graphviz(tree, out_file="tree.dot",class_names=target.unique(),

                feature_names=X.columns,impurity=False,filled=True)



with open("tree.dot") as f:

    dot_graph=f.read()

display(graphviz.Source(dot_graph))



#print misclassified points

mc_pnts_ind = X_test.iloc[tree.predict(X_test) != y_test, :].index.tolist()

print("\nMisclassified points: ")

for ind in range(len(mc_pnts_ind)):

    display(data.loc[data['Id']==(mc_pnts_ind[ind]+1)])
