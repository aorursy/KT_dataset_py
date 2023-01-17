# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics as st
import matplotlib.pyplot as plt 
import json


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file1 = os.path.join(dirname, filename)
        files.append(file1)
# files
# files_json = [f for f in files if f[-4:] == 'json']
files_csv = [f for f in files if f[-3:] == 'csv']
print('There are total ', len(files_csv), 'files')
files_csv.sort()
files_csv
df = pd.DataFrame()
shp = np.zeros(len(files_csv))
i=0

for f in files_csv:
    
    data = pd.read_csv(f, names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"])
    shp[i] = len(data)
    df = df.append(data, sort=False)
    i = i+1
    
 
    
print(df.shape)
df.to_csv(r'/kaggle/working/file1.csv', index = False)
def convert_data(pd_data):
    pdd = pd_data.loc[:,:].values
    return(pdd)
def export_training_data(d, lb, cn):
    d1 = pd.DataFrame(d)
    lb1 = pd.DataFrame(lb)
    exporting_data = pd.concat([d1, lb1], axis=1, sort=False) 
    return(exporting_data)
def join_array(d3, dnew):
    return(np.concatenate((d3, dnew), axis=0))
def find_vars(X_1):
    wa = np.zeros((X_1.shape[0],1))
    for i in range (1,X_1.shape[0]):
        wa[i] = st.mean(X_1[i,:])

    plt.plot(wa)
    vvr = np.zeros((wa.shape[0],1))
    for i in range (0, wa.shape[0]-15):
        vvr[i] = st.variance(wa[i:i+14,0])
    return(vvr)
def apply_pca(da):
    from sklearn.decomposition import PCA 

    pca = PCA(n_components = 1) 

    X_1 = pca.fit_transform(da) 
    # X_2 = pca.transform(X_test) 

    # explained_variance = pca.explained_variance_ratio_ 
    X_1.shape
    return(X_1)
def predict_knn(X1, X2, y1, y2, kn):
    classifier = KNeighborsClassifier(n_neighbors=kn, algorithm='brute')
    classifier.fit(X1, y1)
    predicted = classifier.predict(X2)
    print("KNN: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("KNN is ", (correct), "percent accurate")
    return predicted, correct
def predict_nb(X1, X2, y1, y2):
    classifier = GaussianNB()
    classifier.fit(X1, y1)
    predicted = classifier.predict(X2)
    print("Naive Bayes: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("Naive Bayes is ", (correct), "percent accurate")
    return predicted, correct
def predict_lda(X1, X2, y1, y2, n):
    classifier = LinearDiscriminantAnalysis(n_components=n, store_covariance=False)
    classifier.fit(X1,y1)
    lda = classifier.fit_transform(X1,y1)
    predicted = classifier.predict(X2)
    print("LDA: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("LDA is ", (correct), "percent accurate")
    return predicted, correct, lda
def apply_linear_reg(X1, X2, y1, y2):
    classifier = LinearRegression(normalize=True)
    classifier.fit(X1, y1)
    predicted = classifier.predict(X2)
    print("Linear Regression: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("Linear Regression is ", (correct), "percent accurate")
    return predicted, correct
def apply_lg(X1, X2, y1, y2):
    
    classifier = LogisticRegression()

    classifier.fit(X1, y1)
    predicted=classifier.predict(X2)
    print("Logistic Regression: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("Logistic Regression is ", (correct), "percent accurate")
    return predicted, correct
def predict_svm(X1, X2, y1, y2):
    classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    classifier.fit(X1, y1)
    predicted = classifier.predict(X2)
    print("SVM: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("SVM is ", (correct), "percent accurate")
    return predicted, correct
def predict_dt(X1, X2, y1, y2):
    classifier = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
    classifier.fit(X1, y1)
    tree.plot_tree(classifier)
    predicted = classifier.predict(X2)
    print("Decision Tree: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("Decision Tree is ", (correct), "percent accurate")
    return predicted, correct
def predict_rf(X1, X2, y1, y2):
    classifier = RandomForestClassifier(n_estimators = 200, oob_score = True, criterion = "gini", random_state = 0)
    classifier.fit(X1, y1)
    predicted = classifier.predict(X2)
    print("Random Forrest: Number of mislabeled points out of a total %d points : %d",
          (predicted.shape, (y2 != predicted).sum()))
    correct = 100-(y2 != predicted).sum()/predicted.shape*100
    print("Random Forrest is ", (correct), "percent accurate")
    return predicted, correct
def cluster_data(d):
    kmeans = KMeans(n_clusters=5, max_iter=500)
    kmeans.fit(d)
    labels = kmeans.predict(d)
    centroids = kmeans.cluster_centers_
    return(labels, centroids)
def split_for_kfold(t_data, lbs, kd):
    kfold = KFold(n_splits=kd)
    for train_index1, test_index1 in kfold.split(t_data):
        print("TRAIN:", train_index1.shape, "TEST:", test_index1.shape)
        X_train, X_test = t_data[train_index1], t_data[test_index1]
    for train_index2, test_index2 in kfold.split(lbs):
        print("TRAIN labels:", train_index2.shape, "TEST labels:", test_index2.shape)
        y_train, y_test = lbs[train_index2], lbs[test_index2]
#     t_data.shape
    return X_train, X_test, y_train, y_test
# def apply_all(Xt, yt, kn, n, kd):
#     predicted = np.zeros((0, 6))
#     correct = np.zeros((0, 6))
#     X1, X2, y1, y2 = split_for_kfold(Xt, yt, kd)
#     predicted[0], correct[0] = predict_knn(X1, X2, y1, y2, kn)
#     predicted[1], correct[1] = predict_nb(X1, X2, y1, y2)
#     predicted[2], correct[2] = predict_lda(X1, X2, y1, y2, n)
#     predicted[3], correct[3] = predict_svm(X1, X2, y1, y2)
#     predicted[4], correct[4] = predict_dt(X1, X2, y1, y2)
#     predicted[5], correct[5] = predict_rf(X1, X2, y1, y2)
    
#     predictions = pd.DataFrame()
#     predictions['knn_score'] = predicted[0]
#     predictions['nb_score'] = predicted[1]
#     predictions['svm_score'] = predicted[3]
#     predictions['dt_score'] = predicted[4]
#     predictions['rf_score'] = predicted[5]
#     predictions

#     return predict
# Training Data
traincv_data = convert_data(df)
traincv_data = traincv_data[:,1:]
df.shape
traincv_data.shape
df
features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]
plt.plot(df)
traincv_data.shape
av = np.zeros((traincv_data.shape[0],1))
for i in range (1,traincv_data.shape[0]):
    av[i] = st.mean(traincv_data[i,:])

# av
plt.plot(av[:,:])
av_pd = pd.DataFrame(av[:,:])
sns.pairplot(av_pd, palette='OrRd')
X_pca = apply_pca(traincv_data) #In 1
vvr = find_vars(X_pca)
plt.plot(vvr)
plt.plot(traincv_data, 'green')
plt.plot(X_pca, 'red')
vvr
#Clustering data
mydata = convert_data(df)
mydata = mydata[:,1:]
# labels, centroids = cluster_data(mydata)
# traincv1_data = traincv_data[700+116:,:]
# labels, centroids = cluster_data(traincv_data)
labels, centroids = cluster_data(vvr)
training_data = export_training_data(mydata, labels, centroids)
# training_data.to_csv(r'/kaggle/working/training_data.csv', index = False)
labels
plt.plot(labels)
# plt.plot(vvr)
v_l = pd.DataFrame(vvr)
v_l["Labels"] = labels
v_l
sns.pairplot(v_l, hue='Labels', palette='OrRd')
df["Pre"] = labels
df["vvr"] = vvr
sns.pairplot(df, hue='Pre', palette='OrRd')
X_train, X_test, y_train, y_test = split_for_kfold(traincv_data, labels, kd=9)
X_train.shape
y_train.shape
X_pca_testing = apply_pca(X_test)
vvr_testing = find_vars(X_pca_testing)
X_pca_training = apply_pca(X_train)
vvr_training = find_vars(X_pca_training)
# Training & Prediction of the model
knn_pred, knn_score = predict_knn(vvr_training, vvr_testing, y_train, y_test, kn=3)
knn_pred
plt.plot(knn_pred)
nb_pred, nb_score = predict_nb(vvr_training, vvr_testing, y_train, y_test)
nb_pred
plt.plot(nb_pred)
lng_pred, lng_score = apply_linear_reg(vvr_training, vvr_testing, y_train, y_test)
plt.plot(lng_pred)
lg_pred, lg_score = apply_lg(vvr_training, vvr_testing, y_train, y_test)
plt.plot(lg_pred)
lda_pred, lda_score, lda = predict_lda(X_train, X_test, y_train, y_test, n=2)
plt.plot(X_train, 'green')
plt.plot(lda, 'red')
plt.plot(lda_pred, 'red')
svm_pred, svm_score = predict_svm(vvr_training, vvr_testing, y_train, y_test)
svm_pred
plt.plot(svm_pred)
dt_pred, dt_score = predict_dt(vvr_training, vvr_testing, y_train, y_test)
plt.plot(dt_pred)
rf_pred, rf_score = predict_rf(vvr_training, vvr_testing, y_train, y_test)
rf_pred
plt.plot(rf_pred)
k1 = 9
predictions = pd.DataFrame()
predictions['Kfold'] = [k1]
predictions['knn_score'] = knn_score
predictions['nb_score'] = nb_score
predictions['svm_score'] = svm_score
predictions['dt_score'] = dt_score
predictions['rf_score'] = rf_score
predictions
for i in range (5, 9):
    print('\nKfold = ', i)
    Xt1, Xt2, yt1, yt2 = split_for_kfold(traincv_data, labels, kd=i)
    knnpred, knnscore = predict_knn(Xt1, Xt2, yt1, yt2, kn=5)
    nbpred, nbscore = predict_nb(Xt1, Xt2, yt1, yt2)
    # ldapred, ldascore = predict_lda(X1, X2, y1, y2, n)
    svmpred, svmscore = predict_svm(Xt1, Xt2, yt1, yt2)
    dtpred, dtscore = predict_dt(Xt1, Xt2, yt1, yt2)
    rfpred, rfscore = predict_rf(Xt1, Xt2, yt1, yt2)
    new_row = {'Kfold':i,'knn_score':knnscore, 'nb_score':nbscore, 'svm_score':svmscore, 'dt_score':dtscore, 'rf_score':rfscore}
    predictions = predictions.append(new_row, ignore_index=True)
predictions
