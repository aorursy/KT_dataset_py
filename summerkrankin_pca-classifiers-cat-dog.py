from __future__ import print_function

import pandas as pd

import pickle

import numpy as np

import glob, os, re

from pylab import *



from sklearn.preprocessing  import  StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.decomposition import PCA, KernelPCA



from sklearn.metrics import roc_curve, auc,  classification_report, make_scorer, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import  LogisticRegression  

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import svm



import matplotlib.style as ms

ms.use('seaborn-muted')

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from mpl_toolkits.mplot3d import Axes3D





path = '/your location/pickle/'
def plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=0,title="Checking Train-Test Split"):

    x_plot = []

    x_plot_test = []

    for j in X_train:

        x_plot.append(j[feature_num])

    for j in X_test:

        x_plot_test.append(j[feature_num])



    #plt.figure(figsize=(8,6))

    plt.scatter(x_plot, y_train, c='b')

    plt.scatter(x_plot_test, y_test, c='r')

    plt.xlabel("Feature " + str(feature_num),fontsize=16)

    plt.ylabel("Cat    Dog",fontsize=16);

    plt.title(title,fontsize=16);
with open(path + "both.pkl", 'rb') as picklefile: #both_3fft

    df = pickle.load(picklefile)



#targets# . cat = 0;   dog = 1

y = df['y_val'].values



#features

X1 = df

del X1['y_val']

X = X1
fig0 = plt.figure(figsize=(20,8));



plt.subplot(1,2,1);



plt.title('histogram cat');

plt.axis([-75, 60, 0, 40]);

plt.xlabel('magnitude')



# the row represents one sample

plt.hist(X.iloc[0,:],bins=2000);

plt.hist(X.iloc[1,:],bins=2000);

plt.hist(X.iloc[15,:],bins=2000);

plt.hist(X.iloc[20,:],bins=2000);

plt.hist(X.iloc[120,:],bins=2000);

plt.hist(X.iloc[110,:],bins=2000);





plt.subplot(1,2,2);



plt.title('histogram dog');

plt.axis([-75, 60, 0, 40]);

plt.xlabel('magnitude')



# the row represents one sample

plt.hist(X.iloc[200,:],bins=2000);

plt.hist(X.iloc[210,:],bins=2000);

plt.hist(X.iloc[215,:],bins=2000);

plt.hist(X.iloc[220,:],bins=2000);

plt.hist(X.iloc[250,:],bins=2000);

plt.hist(X.iloc[230,:],bins=2000);

rs = StratifiedShuffleSplit(n_splits=1, random_state=24, test_size=0.25, train_size=None)

for train_index, test_index in rs.split(X,y):

    print("TRAIN:", train_index, "TEST:", test_index)
X_traina = X.iloc[train_index,:]

X_testa =  X.iloc[test_index,:]



y_train = y[train_index]

y_test = y[test_index]



X_train1 = np.array(X_traina)

X_test1 = np.array(X_testa)
stdScale = StandardScaler()



X_train = stdScale.fit_transform(X_train1)



X_test = stdScale.transform(X_test1)
fig0 = plt.figure(figsize=(21,13));



plt.subplot(3,3,1);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=0)



plt.subplot(3,3,2);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=1)



plt.subplot(3,3,3);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=2)



plt.subplot(3,3,4);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=3)



plt.subplot(3,3,5);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=6)



plt.subplot(3,3,6);

plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=10)

plt.tight_layout()
def plot_pca(X,y,titlestr,v1=0,v2=0):

    """ pass in the X from pca.transform ,and the corresponding y values and a string to add to the title

    plots the first three PCA directions/eigenvectors with target values as the color

    ___________________________________________________________________"""

 

    fig = plt.figure(1, figsize=(13, 10))

    ax = Axes3D(fig, elev=-150, azim=110)



    # plot transformed values (the three features that we have decomposed to) , colors correspond to target values

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,

               cmap=plt.cm.cool, edgecolor='k', s=50)

    ax.set_title("First three PCA directions " + titlestr, fontsize=16)

    ax.set_xlabel("1st eigenvector",fontsize=16)

    ax.w_xaxis.set_ticklabels([])

    ax.set_ylabel("2nd eigenvector",fontsize=16)

    ax.w_yaxis.set_ticklabels([])

    ax.set_zlabel("3rd eigenvector",fontsize=16)

    ax.w_zaxis.set_ticklabels([])

    ax.view_init(v1,v2)
# initialize model

pca = PCA(n_components=75)

# fit to training data

pca.fit(X_train)

# transform

X_pca = pca.transform(X_train)
#sns.set()

sns.set_style('whitegrid')

sns.set_context("talk")

fig0 = plt.figure(figsize=(15,8));

sns.barplot(y=pca.explained_variance_ratio_, x=np.arange(75), data=None,color="deeppink", saturation=.5)

plt.plot(np.cumsum(pca.explained_variance_ratio_),'m',alpha=.5)

plt.xlabel('Number of Components', fontsize=16)

plt.ylabel('Ratio of Variance', fontsize=16);

plt.title('Explained Variance Training', fontsize=16)

#plt.grid()  fig1.text(0.80, .8, 'p = {:1f}'.format(pdr[1]), fontsize=12);
pca = PCA(n_components=6)

# fit to training data

pca.fit(X_train)

# transform

X_train_pca = pca.transform(X_train)



plot_pca(X_train_pca,y_train,'train',20,80)
X_test_pca = pca.transform(X_test)

plot_pca(X_test_pca,y_test,'test',30,60)
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=3)

X_kpca = kpca.fit(X_train)

X_back = kpca.transform(X_train)

plot_pca(X_back,y_train,'Kernel RBF train',30,60)
def modeling(model,X_train,X_test,y_train,y_test):

    """ pass through the type of model and the variables to train and predict, yeilds accuracy score"""

    

    model_ = model



    model_.fit(X_train, y_train)

    y_pred = model_.predict(X_test)

    

    score = round(model_.score(X_test,y_test),6)

    print("Accuracy Score: \n", "%.3f"%score)

    print(classification_report(y_test, y_pred, 

                                target_names=["cat","dog"]))
modeling(DummyClassifier(strategy='constant', constant=0),X_train_pca,X_test_pca,y_train,y_test)
modeling(DummyClassifier(strategy='constant', constant=1),X_train_pca,X_test_pca,y_train,y_test)
acc = []

k_range = list(range(1, 21))

for ks in k_range:

    knn = KNeighborsClassifier(n_neighbors=ks)

    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)

    acc.append(accuracy_score(y_test, y_pred))



plt.plot(np.arange(20),acc)

plt.xlabel('Value of K for KNN')

plt.ylabel('Accuracy')

plt.title('Accuracy for each value of K')

plt.grid()
modeling(KNeighborsClassifier(n_neighbors=5),X_train_pca,X_test_pca,y_train,y_test)
modeling(GaussianNB(),X_train_pca,X_test_pca,y_train,y_test)
modeling(GaussianNB(),X_train_pca,X_test_pca,y_train,y_test)
modeling(LogisticRegression(),X_train_pca,X_test_pca,y_train,y_test)
modeling(SVC(kernel="poly"),X_train_pca,X_test_pca,y_train,y_test)
modeling(DecisionTreeClassifier(),X_train_pca,X_test_pca,y_train,y_test)
modeling(RandomForestClassifier(n_estimators=2000),X_train_pca,X_test_pca,y_train,y_test)
def roc_plot(X_train, X_test, y_train, y_test):

    

    

    def fit_roc(model, X_train, X_test, y_train, y_test):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)

        score = round(model.score(X_test,y_test),2)

        fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred_proba[:,1])

        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc,score



    sns.set_style('white')

    sns.set_context("talk")

    fig0 = plt.figure(figsize=(15,8));

    plt.plot([0, 1], [0, 1], lw=2, color = 'black' , linestyle='--')

    

    fpr1, tpr1, roc_auc1,score1 = fit_roc(GaussianNB(), X_train, X_test, y_train, y_test)

    plt.plot(fpr1, tpr1, lw=2, color = 'brown', label='Gauss NB area=%0.2f,accuracy={}'.format(score1) % roc_auc1)

    

    fpr2, tpr2, roc_auc2, score2 = fit_roc(LogisticRegression(), X_train, X_test, y_train, y_test)

    plt.plot(fpr2, tpr2, lw=2, color = 'darkviolet', label='Log Reg area=%0.2f,accuracy={}'.format(score2) % roc_auc2)

    

    fpr5, tpr5, roc_auc5,score5 = fit_roc(KNeighborsClassifier(n_neighbors=5), X_train, X_test, y_train, y_test)

    plt.plot(fpr5, tpr5, lw=2, color = 'darkgray', label='KNN area=%0.2f,accuracy={}'.format(score5)  % roc_auc5)

    

    fpr3, tpr3, roc_auc3,score3 = fit_roc(RandomForestClassifier(n_estimators=2000), X_train, X_test, y_train, y_test)

    plt.plot(fpr3, tpr3, lw=2, color = 'green', label='Rand Forest area=%0.2f,accuracy={}'.format(score3) % roc_auc3)

    

    fpr4, tpr4, roc_auc4,score4 = fit_roc(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)

    plt.plot(fpr4, tpr4, lw=2, color = 'royalblue',label='Decision Tree area=%0.2f,accuracy={}'.format(score4)% roc_auc4)

    

    fpr6, tpr6, roc_auc6,score6 = fit_roc(DummyClassifier(strategy='constant', constant=0), X_train, X_test, y_train, y_test)

    plt.plot(fpr6, tpr6, lw=2, color = 'black',label='Dummy area=%0.2f,accuracy={}'.format(score6)% roc_auc6)

    

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    #plt.title('ROC {}'.format(model), fontsize=12)

    plt.title('ROC All Models', fontsize=12)

    plt.legend(loc="lower right")

    plt.show()

    
roc_plot(X_train_pca,X_test_pca,y_train,y_test)

#save the test/train sets

with open(path + 'X_train.pkl', 'wb') as picklefile:

        pickle.dump(X_traina, picklefile)    

        

with open(path + 'X_test.pkl', 'wb') as picklefile:

        pickle.dump(X_testa, picklefile)    

        

with open(path + 'y_train.pkl', 'wb') as picklefile:

        pickle.dump(y_train, picklefile)  

        

with open(path + 'y_test.pkl', 'wb') as picklefile:

        pickle.dump(y_test, picklefile)    



with open(path + 'test_index.pkl', 'wb') as picklefile:

        pickle.dump(test_index, picklefile)    

        

with open(path + 'train_index.pkl', 'wb') as picklefile:

        pickle.dump(train_index, picklefile)    