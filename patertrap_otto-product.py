import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle



print("All imports OK")
%%time

#As usual, I get confused with the kaggle input. Just ../input/train_file.csv !!

#There is no target column in the test file, so I need to split my train and use part to validate

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

print("Everything looks fine so far")
train=shuffle(train)

train.head()

#Notice the id is index+1
#I want at the beginning of the bar plot: class 1, then 2, etc.

ordered=sorted(set(train['target']))



sns.countplot(x='target', data=train, order=ordered);



plt.title("Number of products per class");

plt.figure();

#check that there are no missing values

print(train.isnull().sum().sum())

#Double sum, to sum over all the list of rows. If it is not 0, remove on of the .sum() and find where they NaNs are.
X=train

Y=train.target

X=X.drop(['target','id'], axis=1)



from sklearn import preprocessing

LE=preprocessing.LabelEncoder()

LE.fit_transform(Y.values.tolist())

Labels=LE.transform(Y.astype(str))

print(LE.classes_)

print(Labels)
%%time



from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier



n_feat=30 #I´ll play a bit with this value and leave the one I like the most (45,30,15)

#The selection takes the same amount of time with 1 feature or with 100, it ranks all of htem and then selects the best ones

#45 is a lot but better than 30, you lose too much info. I'll put 30 for speed.



clasif=RandomForestClassifier()

rfe=RFE(clasif,n_feat)



fit=rfe.fit(X, Labels)



print("Num features: %d" %fit.n_features_)

print("Selected features: %s" %fit.support_)

print("Feature ranking: %s" %fit.ranking_) #Best is 1, worse is increasing



features=[]



#Now I´ll do a loop. If my feature is retained (==True), I keep it.

for i,j in zip(X.columns,fit.support_): #zip iterates on two lists in parallell, not only in one.

    if j==True:

        features.append(str(i))



%%time



X_RFC=X[features]

X_RFC.head() #As always, to check

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X_RFC,Y,random_state=13)



#Now I train my model with my data (No splitting necessary, it was already split in train and test files)



model_rfc=RandomForestClassifier(n_jobs=1, max_depth=13, random_state=17)

#I can add more parameters, but the default should be ok in this set

#I didn´t touch Y, X_smt I will keep on changing

model_rfc.fit(train_X,train_y)



pred_RFC=model_rfc.predict(val_X)

from sklearn.metrics import accuracy_score,confusion_matrix



acc_RFC=np.round(accuracy_score(val_y,pred_RFC),4)

print("Accuracy RFC is: ", acc_RFC)
mode=train_y.mode()

ls=[mode for i in range(len(val_y))]

#ls

acc_mode=np.round(accuracy_score(val_y,ls),4)

print("Accuracy With mode is: ", acc_mode)
labels = [sorted(train.target.unique())]

labels

cm = confusion_matrix(val_y, pred_RFC)

print(cm)

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
cm_N = cm/cm.sum(axis=1)

print(cm_N)

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax_N = ax.matshow(cm_N)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax_N)

ax.set_xticklabels([''] + labels)

#ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
recall_RFC=np.diag(cm)/np.sum(cm, axis=1)

#recall_RFC

precision_RFC=np.diag(cm)/np.sum(cm, axis=0)

#precision_RFC
#Calculate the weights



total_prod=len(train_X.index)

total_classes=train.groupby('target').size() #size() gives a list, count() gives a df

#total_classes

weights=0.75*total_classes/total_prod

#The 0.75 is because I´m doing the weights with the FULL train set, not with the train. The proportions should be the same, as you would expect if the splitting is random

#Then when I divide by the training set, result is > 1, which I correct with 0.75. It´s not a great solution, but it´s a solution.

#Anyway, the numbers are all irrelevant, this is not a real case.

av_recall_RFC=np.multiply(recall_RFC,weights).sum() #each*weight, and then sum all

av_prec_RFC=np.multiply(precision_RFC,weights).sum()



print("The average recall of the RFC is: ", av_recall_RFC)

print("The average precision of the RFC is: ", av_prec_RFC)
beta=1

F1_RFC=(1+beta)*(av_recall_RFC*av_prec_RFC)/(beta**2*av_prec_RFC+av_recall_RFC)

print("The F1 score for the RFC is: ",F1_RFC)
%%time



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,recall_score

#I do with 3 neighbors, then I do a loop with many for cross val



knn=KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

knn.fit(train_X,train_y)

pred_KNN_3=knn.predict(val_X)

#For completeness, I use the built in metrics, later I'll calculate the same thing as before, for consistency

acc_KNN_3=accuracy_score(val_y, pred_KNN_3)

print("The accuracy score from sklearn for KNN3 is: ",acc_KNN_3)





%%time



from sklearn.model_selection import cross_val_score

values_K = list(range(1,15)) #More than 10 neighbours is a bit too much



# empty list for scores

cv_scores = []



#I use accuracy as a metric. I have to choose something and then I'll do a proper fitting and cm and all that with the optimum.





# 5-fold cross validation

for k in values_K:

    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)

    scores = cross_val_score(knn, train_X, train_y, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())



print("Finished!")
print("The optimum number of neighbors is: ", cv_scores.index(max(cv_scores))+1) #Because Index starts at 0!!!

plt.plot(values_K, cv_scores)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Accuracy score')

plt.show()
%%time



model_knn=KNeighborsClassifier(n_neighbors=9,n_jobs=-1)

model_knn.fit(train_X,train_y)



pred_KNN_9=model_knn.predict(val_X)

#Now I do everything with confusion matrix, as before.



print("Trained and predicted")

cm_knn = confusion_matrix(val_y, pred_KNN_9)

print(cm_knn)

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

cax = ax.matshow(cm_knn)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
recall_KNN=np.diag(cm_knn)/np.sum(cm_knn, axis=1)

#recall_RFC

precision_KNN=np.diag(cm_knn)/np.sum(cm_knn, axis=0)

#precision_RFC
av_recall_KNN=np.multiply(recall_KNN,weights).sum() #each*weight, and then sum all

av_precision_KNN=np.multiply(precision_KNN,weights).sum()



print("The average recall of the RFC is: ", av_recall_KNN)

print("The average precision of the RFC is: ", av_precision_KNN)
%%time



from sklearn.ensemble import GradientBoostingClassifier



model_GBDT=GradientBoostingClassifier(loss='deviance', learning_rate=0.15, n_estimators=100, subsample=1, max_depth=8, random_state=13) #No n_jobs

model_GBDT.fit(train_X,train_y)

pred_GBDT=model_GBDT.predict(val_X)

#print("Model fitted and predictions done.")

#train_y.head()



cm_GBDT = confusion_matrix(val_y, pred_GBDT)

#print(cm_GBDT)

#fig = plt.figure(figsize=(20,10))

#ax = fig.add_subplot(111)

#cax = ax.matshow(cm_GBDT)

#plt.title('Confusion matrix of the classifier')

#fig.colorbar(cax)

#ax.set_xticklabels([''] + labels)

#ax.set_yticklabels([''] + labels)

#plt.xlabel('Predicted')

#plt.ylabel('True')

#plt.show()
recall_GBDT=np.diag(cm_GBDT)/np.sum(cm_GBDT, axis=1)

#recall_RFC

precision_GBDT=np.diag(cm_GBDT)/np.sum(cm_GBDT, axis=0)

#precision_RFC



av_recall_GBDT=np.multiply(recall_GBDT,weights).sum() #each*weight, and then sum all

av_precision_GBDT=np.multiply(precision_GBDT,weights).sum()



print("The average recall of the RFC is: ", av_recall_GBDT)

print("The average precision of the RFC is: ", av_precision_GBDT)
%%time



from sklearn import svm



model_SVC_lin=svm.SVC(C=1, kernel='linear')

model_SVC_lin.fit(train_X,train_y)

pred_SVC_lin=model_SVC_lin.predict(val_X)



cm_SVC_lin = confusion_matrix(val_y, pred_SVC_lin)



recall_SVC_lin=np.diag(cm_SVC_lin)/np.sum(cm_SVC_lin, axis=1)

#recall_RFC

precision_SVC_lin=np.diag(cm_SVC_lin)/np.sum(cm_SVC_lin, axis=0)

#precision_RFC



av_recall_SVC_lin=np.multiply(recall_SVC_lin,weights).sum() #each*weight, and then sum all

av_precision_SVC_lin=np.multiply(precision_SVC_lin,weights).sum()



print("The average recall of the SVC with linear kernel is: ", av_recall_SVC_lin)

print("The average precision of the SVC with linear kernel is: ", av_precision_SVC_lin)





%%time

#polynomial. I'll use degree 3-5 and decide manually





model_SVC_p=svm.SVC(C=1, kernel='poly', degree=3)

model_SVC_p.fit(train_X,train_y)

pred_SVC_p=model_SVC_p.predict(val_X)



cm_SVC_p = confusion_matrix(val_y, pred_SVC_p)



recall_SVC_p=np.diag(cm_SVC_p)/np.sum(cm_SVC_p, axis=1)

#recall_RFC

precision_SVC_p=np.diag(cm_SVC_p)/np.sum(cm_SVC_p, axis=0)

#precision_RFC



av_recall_SVC_p=np.multiply(recall_SVC_p,weights).sum() #each*weight, and then sum all

av_precision_SVC_p=np.multiply(precision_SVC_p,weights).sum()



print("The average recall of the SVC of degree 3 is: ", av_recall_SVC_p)

print("The average precision of the SVC of degree 3 is: ", av_precision_SVC_p)



#third degree: recall/prec = 0.751/0.751



#%%time

#polynomial. I'll use degree 3-5 and decide manually





#model_SVC_p=svm.SVC(C=1, kernel='poly', degree=5)

#model_SVC_p.fit(train_X,train_y)

#pred_SVC_p=model_SVC_p.predict(val_X)



#cm_SVC_p = confusion_matrix(val_y, pred_SVC_p)



#recall_SVC_p=np.diag(cm_SVC_p)/np.sum(cm_SVC_p, axis=1)

#recall_RFC

#precision_SVC_p=np.diag(cm_SVC_p)/np.sum(cm_SVC_p, axis=0)

#precision_RFC



#av_recall_SVC_p=np.multiply(recall_SVC_p,weights).sum() #each*weight, and then sum all

#av_precision_SVC_p=np.multiply(precision_SVC_p,weights).sum()



#print("The average recall of the SVC of degree 5 is: ", av_recall_SVC_p)

#print("The average precision of the SVC of degree 5 is: ", av_precision_SVC_p)



#third degree: recall/prec = 0.751/0.751
%%time

#RBF



model_SVC_rbf=svm.SVC(C=1, kernel='rbf')

model_SVC_rbf.fit(train_X,train_y)

pred_SVC_rbf=model_SVC_rbf.predict(val_X)



cm_SVC_rbf = confusion_matrix(val_y, pred_SVC_rbf)



recall_SVC_rbf=np.diag(cm_SVC_rbf)/np.sum(cm_SVC_rbf, axis=1)

#recall_RFC

precision_SVC_rbf=np.diag(cm_SVC_p)/np.sum(cm_SVC_rbf, axis=0)

#precision_RFC



av_recall_SVC_rbf=np.multiply(recall_SVC_rbf,weights).sum() #each*weight, and then sum all

av_precision_SVC_rbf=np.multiply(precision_SVC_rbf,weights).sum()



print("The average recall of the SVC with an rbf kernel is: ", av_recall_SVC_rbf)

print("The average precision of the SVC with an rbf kernel is: ", av_precision_SVC_rbf)



F1_RFC=round((1+beta)*(av_recall_RFC*av_prec_RFC)/(beta**2*av_prec_RFC+av_recall_RFC),4)

print("The F1 score for the RFC is: ",F1_RFC)



F1_KNN_9=round((1+beta)*(av_recall_KNN*av_precision_KNN)/(beta**2*av_precision_KNN+av_recall_KNN),4)

print("The F1 score for the KNN with 9 neighbors is: ",F1_KNN_9)



F1_GBDT=round((1+beta)*(av_recall_GBDT*av_precision_GBDT)/(beta**2*av_precision_GBDT+av_recall_GBDT),4)

print("The F1 score for the GBDT is: ",F1_GBDT)





F1_SVC_lin=round((1+beta)*(av_recall_SVC_lin*av_precision_SVC_lin)/(beta**2*av_precision_SVC_lin+av_recall_SVC_lin),4)

print("The F1 score for the SVM with linear kernel is: ",F1_SVC_lin)



F1_SVC_p=round((1+beta)*(av_recall_SVC_p*av_precision_SVC_p)/(beta**2*av_precision_SVC_p+av_recall_SVC_p),4)

print("The F1 score for the SVM with 3rd degree kernel is: ",F1_SVC_p)



F1_SVC_rbf=round((1+beta)*(av_recall_SVC_rbf*av_precision_SVC_rbf)/(beta**2*av_precision_SVC_rbf+av_recall_SVC_rbf),4)

print("The F1 score for the SVM with 3rd degree kernel is: ",F1_SVC_rbf)




