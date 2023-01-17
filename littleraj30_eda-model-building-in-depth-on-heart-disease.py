import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

color=sns.color_palette()
data=pd.read_csv('../input/heart.csv')

data.head()
fig,ax=plt.subplots(1, 2, figsize = (14,5))

sns.countplot(data=data, x='target', ax=ax[0],palette='Set2')

ax[0].set_xlabel("Disease Count \n [0]->No [1]->Yes")

ax[0].set_ylabel("Count")

ax[0].set_title("Heart Disease Count")

data['target'].value_counts().plot.pie(explode=[0.1,0.0],autopct='%1.1f%%',ax=ax[1],shadow=True, cmap='Greens')

plt.title("Heart Disease")
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='sex',data=data,hue='target',palette='Set1',ax=ax[0])

ax[0].set_xlabel("0 ->Female , 1 ->Male")

data.sex.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0], cmap='Reds')

ax[1].set_title("0 ->Female , 1 -> Male")
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='fbs',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("0-> fps <120 , 1-> fps>120",size=12)

data.fbs.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Oranges')

ax[1].set_title("0 -> fps <120 , 1 -> fps>120",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='restecg',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("resting electrocardiographic",size=12)

data.restecg.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,

                                     explode=[0.005,0.05,0.05],cmap='Blues')

ax[1].set_title("resting electrocardiographic",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='slope',data=data,hue='target',palette='Set1',ax=ax[0])

ax[0].set_xlabel("peak exercise ST segment",size=12)

data.slope.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,explode=[0.005,0.05,0.05],cmap='Blues')



ax[1].set_title("peak exercise ST segment ",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='ca',data=data,hue='target',palette='Set2',ax=ax[0])

ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)

data.ca.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')

ax[1].set_title("number of major vessels colored by flourosopy",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='thal',data=data,hue='target',palette='Set2',ax=ax[0])

ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)

data.thal.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greens')

ax[1].set_title("number of major vessels colored by flourosopy",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='cp',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("Chest Pain")

data.cp.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.01,0.01,0.01,0.01],shadow=True, cmap='Blues')

ax[1].set_title("Chest pain")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='trestbps',data=data,x='sex',hue='target',palette='Set2',ax=ax[0,0])

ax[0,0].set_title("Trestbps V/S Sex")

sns.factorplot(y='trestbps',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set2')

ax[0,1].set_title("Trestbps V/S Chest Pain")

sns.violinplot(y='trestbps',data=data,x='exang',hue='target',ax=ax[1,0],palette='Set2')

ax[1,0].set_title("Trestbps V/S Exang")

sns.swarmplot(y='trestbps',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set2')

ax[1,1].set_title("Trestbps V/S CA (Major Vessel Coloured)")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='chol',data=data,x='sex',hue='target',palette='Set3',ax=ax[0,0])

ax[0,0].set_title("Cholestrol V/S Sex")

sns.boxplot(y='chol',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set3')

ax[0,1].set_title("Cholestrol V/S Chest Pain")

sns.swarmplot(y='chol',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set3')

ax[1,0].set_title("Cholestrol V/S Thal")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='oldpeak',data=data,x='sex',hue='target',palette='Set1',ax=ax[0,0])

ax[0,0].set_title("oldpeak V/S Sex")

sns.boxplot(y='oldpeak',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set1')

ax[0,1].set_title("oldpeak V/S Chest Pain")

sns.swarmplot(y='oldpeak',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set1')

ax[1,0].set_title("oldpeak V/S Thal")

sns.factorplot(y='oldpeak',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set1')

ax[1,1].set_title("oldpeak V/S CA")
fig,ax=plt.subplots(4,3,figsize=(15,15))

for i in range(12):

    plt.subplot(4,3,i+1)

    sns.distplot(data.iloc[:,i],kde=True, color='blue')
fig,ax=plt.subplots(1,1,figsize=(15,5))

features = data.columns

sns.distplot(data[features].mean(axis=1),kde=True,bins=30,color='red')
fig,ax=plt.subplots(1,1,figsize=(15,5))

features = data.columns

sns.distplot(data[features].std(axis=1),kde=True,bins=30,color='green')
fig,ax=plt.subplots(figsize=(15,5))

sns.heatmap(data.isnull(), annot=True)
fig=plt.figure(figsize=(18,18))

sns.heatmap(data.corr(), annot= True, cmap='Blues')
data.sex=data.sex.astype('category')

data.cp=data.cp.astype('category')

data.fbs=data.fbs.astype('category')

data.restecg=data.restecg.astype('category')

data.exang=data.exang.astype('category')

data.ca=data.ca.astype('category')

data.slope=data.slope.astype('category')

data.thal=data.thal.astype('category')
data_label=data['target']

del data['target']

data_label=pd.DataFrame(data_label)
data=pd.get_dummies(data,drop_first=True)

data.head(),data_label.head()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_scaled=MinMaxScaler().fit_transform(data)

data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)
data_scaled.head()
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.20,

                                             stratify=data_label,random_state=9154)
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.ensemble import StackingClassifier Need to update sklearn to use inbuilt stacking classifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
def CrossVal(dataX,dataY,mode,cv=3):

    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='accuracy')

    return(np.mean(score))
def plotting(true,pred):

    fig,ax=plt.subplots(1,2,figsize=(10,5))

    precision,recall,threshold = precision_recall_curve(true,pred[:,1])

    ax[0].plot(recall,precision,'g--')

    ax[0].set_xlabel('Recall')

    ax[0].set_ylabel('Precision')

    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))

    fpr,tpr,threshold = roc_curve(true,pred[:,1])

    ax[1].plot(fpr,tpr)

    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))

    ax[1].plot([0,1],[0,1],'k--')

    ax[1].set_xlabel('False Positive Rate')

    ax[1].set_ylabel('True Positive Rate')
sgd=SGDClassifier(tol=1e-10, random_state=23,loss='log', penalty= "l2", alpha=0.2)

score_sgd=CrossVal(Xtrain,Ytrain,sgd)

print("Accuracy is : ",score_sgd)

sgd.fit(Xtrain,Ytrain)

plotting(Ytest,sgd.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,sgd.predict(Xtest)), annot= True, cmap='Oranges')

sgd_f1=f1_score(Ytest,sgd.predict(Xtest))

plt.title('F1 Score = {}'.format(sgd_f1))
k=KNeighborsClassifier(algorithm='auto',n_neighbors= 19)

score_k=CrossVal(Xtrain,Ytrain,k)

print("Accuracy is : ",score_k)

k.fit(Xtrain,Ytrain)

plotting(Ytest,k.predict_proba(Xtest))





fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,k.predict(Xtest)), annot= True, cmap='Reds')

k_f1=f1_score(Ytest,k.predict(Xtest))

plt.title('F1 Score = {}'.format(k_f1))
lr=LogisticRegression(class_weight='balanced', tol=1e-10)

score_lr=CrossVal(Xtrain,Ytrain,lr)

print("Accuracy is : ",score_lr)

lr.fit(Xtrain,Ytrain)

plotting(Ytest,lr.predict_proba(Xtest))





fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,lr.predict(Xtest)), annot= True, cmap='Greens')

lr_f1=f1_score(Ytest,lr.predict(Xtest))

plt.title('F1 Score = {}'.format(lr_f1))
dtc=DecisionTreeClassifier(max_depth=6)

score_dtc=CrossVal(Xtrain,Ytrain,dtc)

print("Accuracy is : ",score_dtc)

dtc.fit(Xtrain,Ytrain)

plotting(Ytest,dtc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,dtc.predict(Xtest)), annot= True, cmap='Blues')



dtc_f1=f1_score(Ytest,dtc.predict(Xtest))

plt.title('F1 Score = {}'.format(dtc_f1))
svc=SVC(C=0.2,probability=True,kernel='rbf',gamma=0.1)

score_svc=CrossVal(Xtrain,Ytrain,svc)

print("Accuracy is : ",score_svc)

svc.fit(Xtrain,Ytrain)

plotting(Ytest,svc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,svc.predict(Xtest)), annot= True, cmap='Greys')

svc_f1=f1_score(Ytest,svc.predict(Xtest))

plt.title('F1 Score = {}'.format(svc_f1))
rf=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=97)

score_rf= CrossVal(Xtrain,Ytrain,rf)

print('Accuracy is:',score_rf)

rf.fit(Xtrain,Ytrain)

plotting(Ytest,rf.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,rf.predict(Xtest)), annot= True, cmap='Oranges')



rf_f1=f1_score(Ytest,rf.predict(Xtest))

plt.title('F1 Score = {}'.format(rf_f1))
etc=ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=2)

score_etc= CrossVal(Xtrain,Ytrain,etc)

print('Accuracy is:',score_etc)

etc.fit(Xtrain,Ytrain)

plotting(Ytest,etc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,etc.predict(Xtest)), annot= True, cmap='Greens')



etc_f1=f1_score(Ytest,etc.predict(Xtest))

plt.title('F1 Score = {}'.format(etc_f1))
abc=AdaBoostClassifier(sgd,n_estimators=100, random_state=343, learning_rate=0.012)

score_ada= CrossVal(Xtrain,Ytrain,abc)

print('Accuracy is:',score_ada)

abc.fit(Xtrain,Ytrain)

plotting(Ytest,abc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,abc.predict(Xtest)), annot= True, cmap='Reds')



abc_f1=f1_score(Ytest,abc.predict(Xtest))

plt.title('F1 Score = {}'.format(abc_f1))
gbc=GradientBoostingClassifier(n_estimators=100, random_state=43, learning_rate = 0.01)

score_gbc= CrossVal(Xtrain,Ytrain,gbc)

print('Accuracy is:',score_gbc)

gbc.fit(Xtrain,Ytrain)

plotting(Ytest,gbc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,gbc.predict(Xtest)), annot= True, cmap='Blues')



gbc_f1=f1_score(Ytest,gbc.predict(Xtest))

plt.title('F1 Score = {}'.format(gbc_f1))
bc=BaggingClassifier(lr,max_samples=23, bootstrap=True, n_jobs= -1)

score_bc= CrossVal(Xtrain,Ytrain,gbc)

print('Accuracy is:',score_bc)

bc.fit(Xtrain,Ytrain)

plotting(Ytest,bc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,bc.predict(Xtest)), annot= True, cmap='Greys')



bc_f1=f1_score(Ytest,bc.predict(Xtest))

plt.title('F1 Score = {}'.format(bc_f1))
fig= plt.figure(figsize=(10,10))

important=pd.Series(rf.feature_importances_, index=Xtrain.columns)

sns.set_style('whitegrid')

important.sort_values().plot.barh()

plt.title('Feature Importance')
model_accuracy = pd.Series(data=[score_sgd, score_k, score_lr, score_dtc, score_svc, score_rf, score_etc, 

                           score_ada, score_gbc, score_bc], 

                           index=['Stochastic GD','KNN','logistic Regression','decision tree', 'SVM', 'Random Forest',

                            'Extra Tree', 'Ada Boost' , 'Gradient Boost','Bagging Classfier'])

fig= plt.figure(figsize=(8,8))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')
model_f1_score = pd.Series(data=[sgd_f1, k_f1, lr_f1, dtc_f1, svc_f1, rf_f1, etc_f1, 

                           abc_f1, gbc_f1, bc_f1], 

                           index=['Stochastic GD','KNN','logistic Regression','decision tree', 'SVM', 'Random Forest',

                                'Extra Tree', 'Ada Boost' , 'Gradient Boost', 'Bagging Classfier'])

fig= plt.figure(figsize=(8,8))

model_f1_score.sort_values().plot.barh()

plt.title('Model F1 Score Comparison')
vc=VotingClassifier(estimators=[('knn',k),('SGD',sgd),('lr',lr)],

                    voting='soft')

score_vc= CrossVal(Xtrain,Ytrain,vc)

print('Accuracy is:',score_vc)

vc.fit(Xtrain,Ytrain)

plotting(Ytest,vc.predict_proba(Xtest))



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,vc.predict(Xtest)), annot= True, cmap='Greys')



vc_f1=f1_score(Ytest,vc.predict(Xtest))

plt.title('F1 Score = {}'.format(vc_f1))
from sklearn.model_selection import StratifiedKFold

k=StratifiedKFold(n_splits= 5, shuffle=False, random_state=6)
def stacking(model, Xtrain, Ytrain, Xtest, name):

    prediction_train = np.zeros(len(Xtrain))

    prediction_test = np.zeros((len(Xtest)))

    for train_index, test_index in k.split(Xtrain,Ytrain):

        trainset, trainset_label =  Xtrain.iloc[train_index,:], Ytrain.iloc[train_index]

        cv_set, cv_label =  Xtrain.iloc[test_index,:], Ytrain.iloc[test_index]

        

        model.fit(trainset, trainset_label)

        prediction_train[test_index] = model.predict(cv_set)

        

    prediction_test = model.predict(Xtest)

    return (pd.DataFrame({name:prediction_train}),pd.DataFrame({name:prediction_test}))                               
# stacking SGD , Logistic regression, voting classifier

sgd_train, sgd_test = stacking(sgd, Xtrain, Ytrain, Xtest, 'sgd')

lr_train, lr_test = stacking(lr, Xtrain, Ytrain, Xtest, 'logistic')

vc_train, vc_test = stacking(vc, Xtrain, Ytrain, Xtest, 'voting') 
# Combining prediction made by all the three classifiers

trainset = pd.concat([sgd_train,lr_train,vc_train],axis=1)

testset = pd.concat([sgd_test,lr_test,vc_test],axis=1)



# checking correlation 

sns.heatmap(trainset.corr(), annot =True, cmap='Greens')
# meta classifeir

lr=LogisticRegression(class_weight='balanced', tol=1e-20)

score_lr=CrossVal(trainset,Ytrain,lr)

print("Accuracy is : ",score_lr)

lr.fit(trainset,Ytrain)

plotting(Ytest,lr.predict_proba(testset))





fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,lr.predict(testset)), annot= True, cmap='Greens')

lr_f1=f1_score(Ytest,lr.predict(testset))

plt.title('F1 Score = {}'.format(lr_f1))