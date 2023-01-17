# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing,metrics,manifold
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_val_predict
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import collections
import keras as k
import matplotlib.patches as mpatches

%matplotlib inline
from sklearn.preprocessing import RobustScaler
import xgboost
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,r2_score,recall_score,confusion_matrix,precision_recall_curve
from collections import Counter
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD,PCA
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(df)
print(df.describe())
print(df.columns)
#check for NULL
print(df.isnull().any())
#frauds
frauds=round(df['Class'].value_counts()[0]/len(df)*100,3)
non_frauds=round(df['Class'].value_counts()[1]/len(df)*100,3)
print(frauds)
print(non_frauds)
pd.value_counts(df['Class']).plot.bar()
plt.title("Fraud detection freq vs class")
plt.xlabel("Class")
plt.ylabel("Freq")
df['Class'].value_counts()
plt.show()
#check for skewed
fig,ax= plt.subplots(1,2,figsize=(20,4))
amount=df['Amount'].values
time=df['Time'].values
sns.distplot(amount,ax=ax[0],color='g')
ax[0].set_title("Amount")
ax[0].set_xlim(min(df['Amount']),max(df['Amount']))

sns.distplot(time,ax=ax[1],color='b')
ax[1].set_title("Time")
ax[1].set_xlim(min(df['Time']),max(df['Time']))



plt.show()
robscaler=RobustScaler()
df['Amount']=robscaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time']=robscaler.fit_transform(df['Time'].values.reshape(-1,1))

X=df.drop('Class',axis=1)
Y=df['Class']


#dimen reduction
#tsne=TSNE(n_components=2,random_state=42).fit_transform(X.values)
#pca=PCA(n_components=2,random_state=42).fit_transform(X.values)
#tuncated_svd=TruncatedSVD(n_components=2,algorithm='randomized',random_state=42).fit_transform(X.values)

#f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
#f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


#blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
#red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
#ax1.scatter(tsne[:,0],tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
#ax1.scatter(tsne[:,0],tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
#ax1.set_title('t-SNE', fontsize=14)

#ax1.grid(True)

#ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
#ax2.scatter(pca[:,0],pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
#ax2.scatter(pca[:,0],pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
#ax2.set_title('PCA', fontsize=14)

#ax2.grid(True)

#ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
#ax3.scatter(truncated_svd[:,0], truncated_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
#ax3.scatter(truncated_svd[:,0], truncated_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
#ax3.set_title('Truncated SVD', fontsize=14)

#ax3.grid(True)

#ax3.legend(handles=[blue_patch, red_patch])

#plt.show()



print("X shape",X.shape)
print("Y shape",Y.shape)
#sandard split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#stratified split
s=StratifiedKFold(n_splits=5,random_state=0,shuffle=False)
for train_idx,test_idx in s.split(X,Y):
  orig_Xtrain,orig_Xtest=X.iloc[train_idx],X.iloc[test_idx]
  orig_Ytrain,orig_Ytest=Y.iloc[train_idx],Y.iloc[test_idx]
orig_Xtrain=orig_Xtrain.values
orig_Xtest=orig_Xtest.values
orig_Ytrain=orig_Ytrain.values
orig_Ytest=orig_Ytest.values

train_unique_label,train_counts_label=np.unique(orig_Ytrain,return_counts=True)
test_unique_label,test_counts_label=np.unique(orig_Ytest,return_counts=True)
print(train_unique_label)
print(test_unique_label)
print(train_counts_label/len(orig_Ytrain))
print(test_counts_label/len(orig_Ytest))



import sys
#oversampling
print(df.columns)
x=np.array(df.iloc[:,df.columns != 'Class'])
y=np.array(df.iloc[:,df.columns == 'Class'])



print("X shape",X.shape)
print("Y shape",Y.shape)
#sandard split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print(X_train.shape)




#smote
smote=SMOTE(random_state=2)
x_residual_train,y_residual_train=smote.fit_resample(X_train,Y_train)
print("Up sampled smote")
print(x_residual_train.shape)
print(y_residual_train.shape)
print("no of frauds and non frauds")
print(sum(y_residual_train==1))
#print(y_residual_train.value_counts()[1])

#adasyn sampling
adasyn=ADASYN(sampling_strategy="minority",random_state=420, n_neighbors=5)
x_residual_train_adasyn,y_residual_train_adasyn=adasyn.fit_resample(X_train,Y_train)
print("Up sampled adasyn")
print(x_residual_train_adasyn.shape)
print(y_residual_train_adasyn.shape)
print("no of frauds and non frauds")
print(sum(y_residual_train==1))







models=[]
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('DT',DecisionTreeClassifier()))
#models.append(('SVC',SVC()))
model_result=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    results=cross_val_score(model,x_residual_train_adasyn,y_residual_train_adasyn,cv=kfold,scoring=scoring)
    print("Classifiers: ",name, "Has a training score of", round(results.mean(), 2) * 100, "% accuracy score")
    model_result.append(results.mean())
    
    
    





#hyperparameter tuning 
logistic_reg_params={"penalty":['l1','l2'],'C':[0.0001,0.001,0.01,1,10,100,1000]}
gridlog_search=GridSearchCV(LogisticRegression(),logistic_reg_params)
gridlog_search.fit(x_residual_train_adasyn,y_residual_train_adasyn)
log_reg=gridlog_search.best_estimator_

knc_params={"n_neighbors":list(range(2,5,1)),'algorithm':['auto','ball_tree','kd_tree','brute']}
gridknc_search=GridSearchCV(KNeighborsClassifier(),knc_params)
gridknc_search.fit(x_residual_train_adasyn,y_residual_train_adasyn)
knc_grid=gridknc_search.best_estimator_

#svc_params={'C':[0.5,0.7,0.9,0.1],'kernel':['rbf','poly','sigmoid','linear']}
# gridsvc_search=GridSearchCV(SVC(),svc_params)
# gridsvc_search.fit(x_residual_train_adasyn,y_residual_train_adasyn)
# svc_grid=gridsvc_search.best_estimator_

dec_tree_params={'criterion':['gini','entropy'],'maxdepth':list(range(2,4,1))}
griddec_search=GridSearchCV(DecisionTreeClassifier(),dec_tree_params)
griddec_search.fit(x_residual_train_adasyn,y_residual_train_adasyn)
dectree_grid=grddec_search.best_estimator_


models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
#model1s.append(('DT',DecisionTreeClassifier()))
#models1.append(('SVC',SVC()))


for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    acc=cross_val_predict(model,x_residual_train_adasyn,y_residual_train_adasyn,cv=kfold,method="decision_function")
    print("Classifiers: ",name, "Has a training score of", round(acc.mean(), 2) * 100, "% accuracy score")
           
        

for name,model in models:
    model.fit(x_residual_train_adasyn,y_residual_train_adasyn)
    y_pred_adasyn=model.predict(x_residual_train_adasyn)
    cnf_matrix=confusion_matrix(y_residual_train_adasyn,y_pred_adasyn)
    print(classification_report(y_residual_train_adasyn,y_pred_adasyn))
    #validation
    y_pred_test_adasyn=model.predict(X_test)
    cnf_matrix=confusion_matrix(Y_test,y_pred_test_adasyn)
    print(classification_report(Y_test,y_pred_test_adasyn))

    
models1=[]
models1.append(('DT',DecisionTreeClassifier()))
models1.append(('SVC',SVC()))
for name,models in models1:
    models.fit(x_residual_train_adasyn,y_residual_train_adasyn)
    y_predict=models.predict(X_test)
    cnf_mat=confusion_matrix(Y_test,y_predict)
    print(classification_report(Y_test,y_predict))


#deep learning 4 intermediate hidden
from sklearn.metrics import accuracy_score 

from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy,binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

n_inp=np.array(X_train)

model=Sequential([Dense(units=16,input_dim=30,activation="relu"),
                  Dense(units=32,activation="relu"),
                  Dropout(0.5),
                  Dense(units=20,activation="relu"),
                  Dense(units=16,activation="relu"),
                  Dense(units=1,activation="sigmoid")
                 
                 ])


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_residual_train_adasyn,y_residual_train_adasyn,batch_size=25,epochs=5,shuffle=False,verbose=2)
score=model.evaluate(X_test,Y_test)
print("Score")
print(score)
y_deep_predict=model.predict(X_test)
print("Confusion matrix")
print(confusion_matrix(Y_test,y_deep_predict.round()))
print(1-accuracy_score(Y_test,y_deep_predict.round()))
                                                                        
#xbgboost classifier
#boosting trees

from sklearn.metrics import accuracy_score 
from xgboost import XGBClassifier as xg
model_xgb= xg(n_estimators=100,random_state=42)
model_xgb.fit(x_residual_train_adasyn,y_residual_train_adasyn)
y_pred_xgb=model_xgb.predict(X_test)
print("Confusion matrix")
print(confusion_matrix(Y_test,y_pred_xgb.round()))
print(accuracy_score(Y_test,y_pred_xgb.round()))
                                                                        


#LGBMClassifier
from sklearn.metrics import accuracy_score 
from lightgbm import LGBMClassifier as lg
model_lgbm= lg(n_estimators=100,random_state=42)
model_lgbm.fit(x_residual_train_adasyn,y_residual_train_adasyn)
y_pred_lgbm=model_lgbm.predict(X_test)
print("Confusion matrix")
print(confusion_matrix(Y_test,y_pred_lgbm))
print(accuracy_score(Y_test,y_pred_lgbm.round()))
                                                                        
