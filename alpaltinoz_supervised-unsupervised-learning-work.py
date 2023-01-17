# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
display(df.head(),df.tail(),df.describe(),df.info())
#Seperation of Data and Target
x_data,y = df.loc[:,df.columns != 'target'],df.loc[:,'target'].values
#normalize
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=42)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
print('LR accuracy = {}'.format(lr.score(x_test,y_test)))
cm_lr = confusion_matrix(y_test,y_pred_lr)
#Logistic Regression Visiualization
sns.set_palette('muted')

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_lr,annot=True,linewidths=0.5,linecolor='black',fmt='.0f',ax=ax,)
plt.show()
#K=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
print('With K = {}, Accuracy is {}'.format(3,knn.score(x_test,y_test)))
#finding best K value
score_test = []
score_train =[]
for i in range(1,30):
    knn2 = KNeighborsClassifier(i)
    knn2.fit(x_train,y_train)
    score_test.append(knn2.score(x_test,y_test))
    score_train.append(knn2.score(x_train,y_train))
f, ax = plt.subplots(figsize=(15,10))
plt.plot(range(1,30),score_test,label='Test Data Accuracy')
plt.plot(range(1,30),score_train,label='Train Data Accuracy')
plt.legend()
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.show()
print('Best Accuracy is = {} with the K Value of = {}'.format(np.max(score_test),+1+score_test.index(np.max(score_test))))
y_pred_knn = knn2.predict(x_test)
cm_knn = confusion_matrix(y_test,y_pred_knn)


f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,linewidths=0.5,ax=ax)
plt.show()
from sklearn.svm import SVC
svc = SVC(random_state=42)
svc.fit(x_train,y_train)
y_pred_svc = svc.predict(x_test)
print('Svc score is = {}'.format(svc.score(x_test,y_test)) )
cm_svc = confusion_matrix(y_test,y_pred_svc)


f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_svc,annot=True,cmap='coolwarm',linewidth=0.5,ax=ax)
plt.show()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred_nb = nb.predict(x_test)
print('Accuracy with Naive Bayes = {}'.format(nb.score(x_test,y_test)))
cm_nb = confusion_matrix(y_test,y_pred_nb)

f, ax= plt.subplots(figsize=(5,5))
sns.heatmap(cm_nb,annot=True,linewidth=0.5,ax=ax)
plt.show()
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
cm_dtc = confusion_matrix(y_test,y_pred_dtc)

print('Accuracy with Decision Tree Classifier {}'.format(dtc.score(x_test,y_test)))


f,ax= plt.subplots(figsize=(5,5))
sns.heatmap(cm_dtc,annot=True,linewidths=0.5,ax=ax)
plt.show()
#finding best Estimator value
from sklearn.ensemble import RandomForestClassifier
score_list = []
for i in range(1,101):
    rfc2= RandomForestClassifier(n_estimators=i,random_state=42)
    rfc2.fit(x_train,y_train)
    score_list.append(rfc2.score(x_test,y_test))
    print('Accuracy with Random forest is {} with {} Trees'.format(rfc2.score(x_test,y_test),i))

print('Best Accuracy with Random Forest Classifier is {} with the Decision Tree number of {} '.format(np.max(score_list),+1+score_list.index(np.max(score_list))))
plt.plot(range(1,101),score_list,c='orange',label='RF Accuracy')
plt.legend()
plt.xlabel('# of Decision Trees')
plt.ylabel('Accuracy')
plt.show()
best_estimator = score_list.index(np.max(score_list))+1
rfc = RandomForestClassifier(n_estimators=best_estimator,random_state=42)
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test,y_pred_rfc)

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_rfc, annot=True, linewidths=0.5,ax=ax)
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
linr = LinearDiscriminantAnalysis()
linr.fit(x_train,y_train)
print('Accuracy with Linear Disc Analysis is {}'.format(linr.score(x_test,y_test)))
y_pred_linr = linr.predict(x_test)

cm_linr = confusion_matrix(y_test,y_pred_linr) 
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_linr,annot=True)
plt.show()
#ML models in Automation

models = []

models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('RF',RandomForestClassifier(n_estimators=13)))
models
#Results for ML Models


for name, model in models:
    
    clf=model

    clf.fit(x_train, y_train)

    y_pred =clf.predict(x_test)
    print(10*"=","{} için Sonuçlar".format(name).upper(),10*"=")
    print("Accuracy Score:{:0.2f}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))
    print("Classification Report:\n{}".format(classification_report(y_test,y_pred)))
    print(30*"=")

print("Logistic Regression Accuracy:", 100*lr.score(x_test, y_test), "%")
print("KNN Prediction Accuracy:", 100*knn.score(x_test, y_test), "%")
print("SVM Prediction Accuracy:", 100*svc.score(x_test, y_test), "%")
print("Naive Bayes Prediction Accuracy:", 100*nb.score(x_test, y_test), "%")
print("Decision Trees Prediction Accuracy:", 100*dtc.score(x_test, y_test), "%")
print("Random Forest Prediction Accuracy:", 100*rfc.score(x_test, y_test), "%")
print('Linear Disc Analysis Accuracy:',100*linr.score(x_test,y_test),'%')

dfu_data = df.loc[:, df.columns != 'target']
dfu_data.head()
#normalize data
dfu = (dfu_data-np.min(dfu_data))/(np.max(dfu_data)-np.min(dfu_data))

#Kmeans Clustering
from sklearn.cluster import KMeans
inertia_list = np.empty(10)
for i in range(1,10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit_predict(dfu)
    inertia_list[i]= kmeans.inertia_
plt.plot(range(0,10),inertia_list,'-o',c='r')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

#graph suggests Cluster number to be 2, as we know from the labeled data there were 2 labels
kmeans2= KMeans(n_clusters=2)


labels = kmeans2.fit_predict(dfu)
checkdata = pd.DataFrame({'labels': labels,'target':df.target})
ct = pd.crosstab(checkdata.labels,checkdata.target)
ct

#Standartization and making pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
pipe = make_pipeline(scalar,kmeans2)
pipe.fit(dfu)
labels = pipe.predict(dfu)
pipedf = pd.DataFrame({'labels':labels,'target':df.target})
ct = pd.crosstab(pipedf.labels,pipedf.target)
ct

#Hierarchical Clustering
from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(dfu,method='ward')
dendrogram(merg,leaf_rotation=90)
plt.show
