#Import packages and dependencies

import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from numpy import linalg

from sklearn import svm

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import normalize

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

import seaborn as sns

from  matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl

import matplotlib.patches as mpatches
#Import raw data

train_raw = pd.read_csv("../input/data_set_ALL_AML_train.csv").drop(['Gene Description','Gene Accession Number'],axis=1)

test_raw = pd.read_csv("../input/data_set_ALL_AML_independent.csv").drop(['Gene Description','Gene Accession Number'],axis=1)

y_train = list(pd.read_csv("../input/actual.csv")['cancer'][:38])

y_test = list(pd.read_csv("../input/actual.csv")['cancer'][38:])



#Clean raw data

train_data = pd.DataFrame(train_raw[[col for col in train_raw.columns if "call" not in col]])

test_data = pd.DataFrame(test_raw[[col for col in test_raw.columns if "call" not in col]])

y_train = [0 if x == "ALL" else 1 for x in y_train]

y_test = [0 if x == "ALL" else 1 for x in y_test]
class PCA:

    '''

    Class performs principle component analysis

    Stores explained variance and cumalative sum of variance

    '''

    def __init__(self, data,mean):

        self.mean_gene = mean

        #Initialize data matrix

        self.data = self.normalize(data)

        #Initialize dimensions of data matrix, m observations on p variables

        [self.m,self.p] = self.data.shape

        #Initialize array to store the variance and sum of variance

        self.explained_variance = []

        self.cumsum = []

        

    def normalize(self,X):

        #Column center the data matrix X

        #In this case, subtracts the mean gene expression from each assay

        i,j = X.shape

        return X - np.outer(self.mean_gene, np.ones(j))

    

    def SVD(self,X):

        #Method returns the singular value decomposition of a data matrix

        [U,S,V] = np.linalg.svd(X,full_matrices=False)

        #Compute the explained variance derived from the SVD

        var_sum = sum(S)

        self.explained_variance = S/var_sum*100

        #Compute the comulative sum derived from the SVD

        self.cumsum = np.cumsum(self.explained_variance)

        return U,S,normalize(V.T,axis=0)

    

    def PC(self,X):

        #Method returns the principle component projection of the data matrix

        [U,S,V] = self.SVD(X)

        return normalize(np.dot(X,V.T),axis=0)
#Initialize PCA object with training data and mean gene expression

trainPCA = PCA(train_data,train_data.mean(1))
#Acquire the low dimensional projection in eigenspace of the training data

PC = trainPCA.PC(trainPCA.data)

U,S,V = trainPCA.SVD(trainPCA.data)
cmap=LinearSegmentedColormap.from_list('rg',["r", "black", "g"], N=256) 

ax = sns.heatmap(V,cmap=cmap)

plt.title('Training Eigengenes')

plt.show()
#Plot of individual assays in 2-dimensional eigenspace

%matplotlib inline

colors = ['red', 'blue']

levels = [0, 1]

cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')

plt.scatter(V[:,0],V[:,1],c=y_train,cmap=cm.bwr)

plt.title("Training Data Projection in 2D")

plt.xlabel("1st Principle Axis")

plt.ylabel("2nd Principle Axis")

red_patch = mpatches.Patch(color='red', label='AML')

blue_patch = mpatches.Patch(color='blue', label='ALL')

plt.legend(handles=[red_patch,blue_patch])

plt.show()
#Plot of individual assays in 3-dimensional eigenspace

%matplotlib inline

colors = ['red', 'blue']

levels = [0, 1]

cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')

plt.clf()

fig = plt.figure(1, figsize=(10,6 ))

ax = Axes3D(fig, elev=-130, azim=30,)

ax.scatter(V[:,0], V[:,1], V[:,2], c=y_train,cmap=cm.bwr,linewidths=10)

ax.set_title("Training Data Projection in 3D")

ax.set_xlabel("1st Principle Axis")

ax.set_ylabel("2nd Principle Axis")

ax.set_zlabel("3rd Principle Axis")

ax.legend(handles=[red_patch,blue_patch])

plt.show()
#Determine the low dimension representation of the testing data

testnorm = test_data - np.outer(train_data.mean(1), np.ones(34))

#This is accomplished by projecting the testing data in the eigenspace of the training eigengenes

W = np.dot(testnorm.T,PC)

W = normalize(W,axis=0)
cmap=LinearSegmentedColormap.from_list('rg',["r", "black", "g"], N=256) 

ax = sns.heatmap(W,cmap=cmap)

plt.title('Testing Eigengenes')

plt.show()
%matplotlib inline

colors = ['red', 'blue']

levels = [0, 1]

cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')

plt.scatter(W[:,0],W[:,1],c=y_test,cmap=cm.bwr)

plt.title("Testing Data Projection in 2D")

plt.xlabel("1st Principle Axis")

plt.ylabel("2nd Principle Axis")

red_patch = mpatches.Patch(color='red', label='AML')

blue_patch = mpatches.Patch(color='blue', label='ALL')

plt.legend(handles=[red_patch,blue_patch])

plt.show()
%matplotlib inline

plt.clf()

colors = ['red', 'blue']

levels = [0, 1]

cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')

fig = plt.figure(1, figsize=(10,6 ))

ax = Axes3D(fig, elev=-130, azim=30,)

ax.scatter(W[:,0],W[:,1],W[:,2], c=y_test,cmap=cm.bwr,linewidths=10)

ax.set_title("Testing Data Projection in 3D")

ax.set_xlabel("1st Principle Axis")

ax.set_ylabel("2nd Principle Axis")

ax.set_zlabel("3rd Principle Axis")

ax.legend(handles=[red_patch,blue_patch])

plt.show()
#Neural Network

mlp = MLPClassifier(solver='lbfgs', alpha=1e-9,hidden_layer_sizes=(38, 10), random_state=4)

mlp.fit(V, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(mlp.score(W, y_test)))
confusion_matrix(y_test, mlp.predict(W))
#Decision Tree

tr = DecisionTreeClassifier(random_state=15).fit(V,y_train)

print("Accuracy on training set: {:.2f}".format(tr.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(tr.score(W, y_test)))
confusion_matrix(y_test, tr.predict(W))
#Logistic Regression

logreg = LogisticRegression().fit(V,y_train)

print("Accuracy on training set: {:.2f}".format(logreg.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(logreg.score(W, y_test)))
confusion_matrix(y_test, logreg.predict(W))
#Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=14)

rf.fit(V, y_train)

print("Accuracy on training set: {:.2f}".format(rf.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(rf.score(W, y_test)))
confusion_matrix(y_test, rf.predict(W))
#Support Vector Machine

svc = svm.SVC()

svc.fit(V, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(W, y_test)))
confusion_matrix(y_test, svc.predict(W))
#Non-linear Support Vector Machine

nsvc = svm.NuSVC().fit(V,y_train)

print("Accuracy on training set: {:.2f}".format(nsvc.score(V, y_train)))

print("Accuracy on test set: {:.2f}".format(nsvc.score(W, y_test)))
confusion_matrix(y_test, nsvc.predict(W))