# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#functions
def classification_reports(classifier,features,labels):
    import sklearn.metrics as metr
    size_data = len(labels)
    count_class_1 = sum(labels)
    count_class_0 = size_data - count_class_1
    print(' class 1 : ', count_class_1)
    print(' class 0 : ', count_class_0)
    y = classifier.predict(features)
    fpr, tpr, thresholds = metr.roc_curve(labels, y)
    print("Confusion Matrix: \n",metr.confusion_matrix(labels,y))
    score=metr.accuracy_score(labels,y)
    print("Score: ",score)
    auc=metr.roc_auc_score(labels,y)
    print("AUC: ",auc)
    print(metr.classification_report(labels,y))
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    Legend = "FirstStage"
    plt.plot( fpr, tpr,color='darkorange')
    plt.show()

    return fpr,tpr,score
def pre_process(features) :
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import robust_scale
    from sklearn.preprocessing import Imputer
    del features['Name']
    del features['PlayerID']
    x = np.isfinite(features)
    features[(x == False) & (features > 0 )] =  10000000
    features[(x == False) & (features < 0 )] = -10000000

    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(features)
    features= imputer.transform(features)

    features = normalize(features, axis=0,norm='max')
    # features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)
    # features = StandardScaler().fit_transform(features)
    # features = robust_scale(features)

    return features
def pre_process_pca(features,test):
    Data = np.concatenate((features,test))
    Data = PCA(n_components=.95).fit_transform(Data)
    return Data[0:900,:],Data[900:1340,:]
def pre_process_manifold(features,test):
    from sklearn import manifold
    n_neighbors = 15
    Data = np.concatenate((features, test))
    Data = manifold.Isomap(n_neighbors, n_components=15).fit_transform(Data)
    return Data[0:900,:],Data[900:1340,:]
def visualaze_data_pca(Data,labels):
    Data = PCA(n_components=2).fit_transform(Data)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0,1]
    colors = ['r', 'b']
    Data = pd.DataFrame(data=Data
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([Data,labels], axis=1)
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['TARGET_5Yrs'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    return
def visualaze_data_manifold(Data,labels):
    from sklearn import manifold
    n_neighbors = 5
    Data = manifold.Isomap(n_neighbors, n_components=2) \
        .fit_transform(Data)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_title('2 component Manifold', fontsize=20)
    targets = [0,1]
    colors = ['r', 'b']
    Data = pd.DataFrame(data=Data
                               , columns=['component 1', 'component 2'])
    finalDf = pd.concat([Data,labels], axis=1)
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['TARGET_5Yrs'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'component 1']
                   , finalDf.loc[indicesToKeep, 'component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    return
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import ensemble
from  sklearn import naive_bayes
from  sklearn import neural_network
from sklearn import  neighbors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
# main code

###read data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
###extracting labels
labels = train.TARGET_5Yrs
features = train.iloc[:,0:21]

###pre processing
features = pre_process(features)
test = pre_process(test)
###visualiaze data
visualaze_data_pca(features,labels)
visualaze_data_manifold(features,labels)
###feature extraction
# features,test = Supply.pre_process_pca(features,test)
# features,test = Supply.pre_process_manifold(features,test)

#classifiers
classifier1 = neural_network.MLPClassifier(hidden_layer_sizes=(200,10),random_state=0)
classifier2 = naive_bayes.GaussianNB()
classifier3 =neighbors.KNeighborsClassifier(n_neighbors=1)
classifier4= svm.NuSVC(kernel='rbf',random_state=0)
classifier5 =  neural_network.MLPClassifier(hidden_layer_sizes=(20,5),random_state=0)
classifier6 = svm.SVC(kernel='rbf', C=.5,random_state=0)
classifier7 = ensemble.AdaBoostClassifier(n_estimators=60,learning_rate=.3,random_state=0)
classifier8 = ensemble.GradientBoostingClassifier(n_estimators=20,max_depth=30,random_state=0)
classifier9 = ensemble.RandomForestClassifier(n_estimators=60,random_state=0)
classifier10 = ensemble.ExtraTreesClassifier(n_estimators=10,random_state=0)
classifier11 = ensemble.BaggingClassifier(n_estimators=10,random_state=0)
classifier13 = neighbors.KNeighborsClassifier(n_neighbors=3)
classifier14 = QuadraticDiscriminantAnalysis()
###finding outlier data
classifier12 = ensemble.IsolationForest(n_estimators=100,bootstrap="true",contamination=0.1,random_state=0)
classifier12.fit(features, labels)
result12 = classifier12.predict(test)
outlier  = classifier12.predict(features)
delfeatures = np.where(outlier == -1)
###downgrading outlier data
for i in np.int32(delfeatures):
    features[i,:] = np.sqrt(features[i,:])
###build final classifier
classifier = ensemble.VotingClassifier\
    (estimators=[('mlp',classifier1),('svm',classifier4),('knn3',classifier13),
                 ('knn1',classifier3),('nb',classifier2),('mlpmini', classifier5),
                 ('svm.5',classifier6), ('ada',classifier7), ('gb',classifier8),
                 ('rf', classifier9),('et',classifier10), ('bag',classifier11),
                 ('quad',classifier14)], voting='hard',flatten_transform='true',
                  weights=[2,2,1,1,1,1,1,1,1,1,1,1,1]
     )
classifier.fit(features, labels)
result = classifier.predict(test)
result[result==-1]=0
###report
scores = cross_val_score(classifier, features, labels, cv=10,scoring='f1')
classification_reports(classifier,features,labels)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': result }
submission = pd.DataFrame(cols)
submission.to_csv("subm.csv", index=False)
print(submission)
