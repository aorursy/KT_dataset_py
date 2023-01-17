import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color=sns.color_palette()#helps to give colors to the plots

import warnings ## importing warnings library. 

warnings.filterwarnings('ignore') ## Ignore warning
dt=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

dt.head()
dt.head()
fig,ax=plt.subplots(1,1,figsize=(14,5))#fig-specifies whole graph whereas ax represents the subplot that are in graph as:-ax

#count of people having diseases

sns.countplot(data=dt,x='target',ax=ax,palette='Set1')

ax.set_xlabel("No.of people has heart disease")

ax.set_ylabel("count")

ax.set_title("Frequency of people having heart diseases")

fig,ax=plt.subplots(1,1,figsize=(7,5))

dt['target'].value_counts().plot.pie(ax=ax,autopct='%1.1f%%')#or we can have explode parameter to have some separation between two
fig,ax=plt.subplots(1,2,figsize=(12,5))

sns.countplot(data=dt,x='sex',hue='target',palette='Set1',ax=ax[0])

ax[0].set_xlabel("0 is for feamle and 1 is for male")

ax[0].set_ylabel("count")

dt['sex'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[0].set_title("sex vs survived")

ax[1].set_title("male vs female survived")
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(data=dt,x='cp',hue='target',palette="Set2",ax=ax[0])

ax[0].set_xlabel("chest pain category")

ax[0].set_ylabel("count")

ax[0].set_title("cp vs survival")

dt['cp'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[1].set_title("cp category vise percentage")
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(x='restecg',data=dt,hue='target',palette='Set1',ax=ax[0])#different sets gives different colors

ax[0].set_xlabel("resting electrocardiographic")

dt.restecg.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[1].set_title("resting electrocardiographic")
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(data=dt,x='fbs',hue='target',ax=ax[0],palette="Set3")

ax[0].set_xlabel("0 for fps <120 , 1 for fps>120")

ax[0].set_ylabel("count")

ax[0].set_title("fps vs heart disease")

dt['fbs'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(x='ca',data=dt,hue='target',palette='Set2',ax=ax[0])

ax[0].set_xlabel("number of major vessels colored by flourosopy")

dt['ca'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[1].set_title("number of major vessels colored by flourosopy")
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(data=dt,x='slope',palette='Set1',hue='target',ax=ax[0])

ax[0].set_xlabel('slope of the peak')

dt['slope'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[1].set_title('slope of peaks')
fig,ax=plt.subplots(1,2,figsize=(12,6))

sns.countplot(data=dt,x='thal',hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("thalassemia levels")

dt['thal'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%')

ax[1].set_title('thalassemia levels')
fig,ax=plt.subplots(2,2,figsize=(16,8))

sns.swarmplot(y='trestbps',x='ca',hue='target',data=dt,ax=ax[0,0])

sns.swarmplot(y='trestbps',x='sex',hue='target',data=dt,ax=ax[1,0])

sns.swarmplot(y='trestbps',x='cp',hue='target',data=dt,ax=ax[0,1])

sns.swarmplot(y='trestbps',x='exang',hue='target',data=dt,ax=ax[1,1])
fig,ax=plt.subplots(2,2,figsize=(14,8))

sns.swarmplot(y='chol',x='sex',hue='target',data=dt,ax=ax[0,0])

sns.swarmplot(y='chol',x='cp',hue='target',data=dt,ax=ax[1,0])

sns.swarmplot(y='chol',x='ca',hue='target',data=dt,ax=ax[0,1])

sns.scatterplot(x=dt['target'],y=dt['chol'],ax=ax[1,1])
sns.scatterplot(x=dt['target'],y=dt['oldpeak'])
fig,ax=plt.subplots(4,3,figsize=(16,16))

for i in range(12):

    plt.subplot(4,3,i+1)

    sns.kdeplot(data=dt.iloc[:,i],shade=True)
dt.isna().sum()
plt.figure(figsize=(16,16))

sns.heatmap(dt.corr(),annot=True)
dt.sex=dt.sex.astype('category')

dt.cp=dt.cp.astype('category')

dt.fbs=dt.fbs.astype('category')

dt.restecg=dt.restecg.astype('category')

dt.exang=dt.exang.astype('category')

dt.ca=dt.ca.astype('category')

dt.slope=dt.slope.astype('category')

dt.thal=dt.thal.astype('category')
model_label=dt['target']

model_label=pd.DataFrame(model_label)

del dt['target']
dt1=pd.get_dummies(dt,drop_first=True)

dt1.head()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dt1_scaled=MinMaxScaler().fit_transform(dt1)

dt1_scaled=pd.DataFrame(data=dt1_scaled, columns=dt1.columns)

dt1_scaled.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dt1_scaled,model_label,random_state=25,stratify=model_label)#stratify makes equal proportions to go onto both sets
from sklearn.metrics import confusion_matrix#matrix of true positives,true negatives,false positives,false negatives...

from sklearn.metrics import precision_recall_curve#x-axis->precision,y-axis->recall where precision means what fraction of +ve predictions are correct and recall means what fraction of positives are correctly identified.

from sklearn.metrics import average_precision_score#weighted mean of precision with weight as increase in the amount of recall from previous one

from sklearn.metrics import roc_curve#x-axis->False positives,y-axis->true positives

from sklearn.metrics import auc#Area under the Roc curve

from sklearn.model_selection import cross_val_score#cross validation score

from sklearn.metrics import f1_score#harmonic mean of precision and recall
def CrossValidate(dtX,dtY,model,cv=5):#defalut cv=5(folds),model means algorithm that we applied

    score=cross_val_score(model,dtX , dtY, cv=cv, scoring='accuracy')#return scores of all folds

    return(np.mean(score))#taking average of all.
def plotting(true_values,pred_values):

    fig,axe=plt.subplots(1,2,figsize=(12,6))

    precision,recall,threshold = precision_recall_curve(true_values,pred_values[:,1])#returns three arrays of precision,recalls and thresholds with respect to which those are attained

    axe[0].plot(precision,recall,'r--')

    axe[0].set_xlabel('Precision')

    axe[0].set_ylabel('Recall')

    axe[0].set_title("Average Precision Score : {}".format(average_precision_score(true_values,pred_values[:,1])))#probabilities of 1's-pred[:,1] means

    fpr,tpr,threshold = roc_curve(true_values,pred_values[:,1])#fpr->false positives,tpr->true positive

    axe[1].plot(fpr,tpr)

    axe[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))#area under curve

    axe[1].set_xlabel('False Positive Rate')

    axe[1].set_ylabel('True Positive Rate')
from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier(n_neighbors=18)#Checking different values of n_neighbors gives the maximum of them

knn_score=CrossValidate(X_train,np.ravel(y_train),knn_clf)

print(knn_score)

knn_clf.fit(X_train,np.ravel(y_train))

plotting(y_test,knn_clf.predict_proba(X_test))



#Now calculate F1 scores:-

fig=plt.figure(figsize=(10,5))

sns.heatmap(confusion_matrix(y_test,knn_clf.predict(X_test)),annot=True)

F1_knn=f1_score(y_test,knn_clf.predict(X_test))

print(F1_knn)
from sklearn.linear_model import LogisticRegression

lg_clf=LogisticRegression(C=10,solver='lbfgs')

lg_score=CrossValidate(X_train,np.ravel(y_train),lg_clf)

print(lg_score)

lg_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),lg_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,lg_clf.predict(X_test)),annot=True)

F1_lg=f1_score(y_test,lg_clf.predict(X_test))

print(F1_lg)
from sklearn.tree import DecisionTreeClassifier

dt_clf=DecisionTreeClassifier(max_depth=3)

dt_score=CrossValidate(X_train,np.ravel(y_train),dt_clf)

print(dt_score)

dt_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),dt_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,dt_clf.predict(X_test)),annot=True)

F1_dt=f1_score(y_test,dt_clf.predict(X_test))

print(F1_dt)
from sklearn.svm import SVC

svm_clf=SVC(kernel='rbf',C=0.1,gamma=0.1,probability=True)#To call Predict_proba we must keep probability as True which makes the output to be given as probability estimates.

svm_score=CrossValidate(X_train,np.ravel(y_train),svm_clf)

print(svm_score)

svm_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),svm_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,svm_clf.predict(X_test)),annot=True)

F1_svm=f1_score(y_test,svm_clf.predict(X_test))

print(F1_svm)
from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(max_features=3,random_state=25,n_estimators=10)

rf_score=CrossValidate(X_train,np.ravel(y_train),rf_clf)

print(rf_score)

rf_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),rf_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,rf_clf.predict(X_test)),annot=True)

F1_rf=f1_score(y_test,rf_clf.predict(X_test))

print(F1_rf)
from sklearn.naive_bayes import GaussianNB

nb_clf=GaussianNB()

nb_score=CrossValidate(X_train,np.ravel(y_train),nb_clf)

print(nb_score)

nb_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),nb_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,nb_clf.predict(X_test)),annot=True)

F1_nb=f1_score(y_test,nb_clf.predict(X_test))

print(F1_nb)
#we can also use batch gradient descent but it takes a lot more time when compared to this as weights gets updated after each batch.

from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier(alpha=0.3, random_state=37, penalty= "l2",loss='log')#applies l2 regulrazation,we need apply log as loss function to get probablities

sgd_score=CrossValidate(X_train,np.ravel(y_train),sgd_clf)

print(sgd_score)

sgd_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),sgd_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,sgd_clf.predict(X_test)),annot=True)

F1_sgd=f1_score(y_test,sgd_clf.predict(X_test))

print(F1_sgd)
from sklearn.ensemble import GradientBoostingClassifier

gb_clf=GradientBoostingClassifier(learning_rate=0.1,max_depth=6,n_estimators=150,random_state=0,tol=1e-10)#increasing learning rate will make model more complex.

gb_score=CrossValidate(X_train,np.ravel(y_train),gb_clf)

print(gb_score)

gb_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),gb_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,gb_clf.predict(X_test)),annot=True)

F1_gb=f1_score(y_test,gb_clf.predict(X_test))

print(F1_gb)

from sklearn.neural_network import MLPClassifier

nn_clf=MLPClassifier(hidden_layer_sizes=[150,100,100,100,100],solver='lbfgs',random_state=20,activation='tanh',alpha=0.01)

nn_score=CrossValidate(X_train,np.ravel(y_train),nn_clf)

print(nn_score)

nn_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),nn_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,nn_clf.predict(X_test)),annot=True)

F1_nn=f1_score(y_test,nn_clf.predict(X_test))

print(F1_nn)

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf_clf, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
fig= plt.figure(figsize=(10,10))

important_feature=pd.Series(rf_clf.feature_importances_, index=X_train.columns)

important_feature.sort_values().plot.bar()
models_accuracy=pd.Series(data=[knn_score,lg_score,dt_score,svm_score,rf_score,nb_score,sgd_score,gb_score,nn_score],index=["knn","lg","dt","svm","rf","nb","sgd","gb","nn"])

models_accuracy.sort_values().plot.bar()
models_accuracy=pd.Series(data=[F1_knn,F1_lg,F1_dt,F1_svm,F1_rf,F1_nb,F1_sgd,F1_gb,F1_nn],index=["knn","lg","dt","svm","rf","nb","sgd","gb","nn"])

models_accuracy.sort_values().plot.bar()
from sklearn.ensemble import VotingClassifier

vc_clf=VotingClassifier(estimators=[('knn',knn_clf),('lg',lg_clf),('sgd',sgd_clf),('svm',svm_clf),('nn',nn_clf)],voting='soft')

vc_score=CrossValidate(X_train,np.ravel(y_train),vc_clf)

print(vc_score)

vc_clf.fit(X_train,np.ravel(y_train))

plotting(np.ravel(y_test),vc_clf.predict_proba(X_test))



fig=plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix(y_test,vc_clf.predict(X_test)),annot=True)

F1_vc=f1_score(y_test,vc_clf.predict(X_test))

print(F1_vc)