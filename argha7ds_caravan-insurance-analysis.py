import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model, preprocessing, model_selection
plt.style.use('fivethirtyeight')
train = pd.read_csv("../input/ticdata2000.csv")
test = pd.read_csv("../input/ticeval2000.csv")
result_true=pd.read_csv("../input/tictgts2000.csv")
#df = pd.read_csv('caravan-insurance-challenge.csv')
np.random.seed(42)
test.shape
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
target=train.iloc[:,-1].values
train_t=train.iloc[:,0:85].values
target=target.reshape(5821,1)
target.shape
train_t.shape
target_df=pd.DataFrame(target)
train_df=pd.DataFrame(train_t)
target.shape,train_t.shape
type(train_df),type(target_df)
train_so=train_df.iloc[:,0:43].values
train_po=train_df.iloc[:,43:86].values
test_so=test.iloc[:,0:43].values
test_po=test.iloc[:,43:86].values
test_po.shape
mms=MinMaxScaler()
train_so=mms.fit_transform(train_so)
train_po=mms.fit_transform(train_po)
test_so=mms.fit_transform(test_so)
test_po=mms.fit_transform(test_po)
plt.figure(figsize=[21,15])
plt.boxplot(train_so,vert=False,patch_artist=True)
plt.show()
plt.figure(figsize=[21,15])
plt.boxplot(test_so,vert=False,patch_artist=True)
plt.show()
pca=PCA()
pca.fit(train_so)
exp_var=pca.explained_variance_  # Which feature is having maximum variance based on Probality score
exp_var
exp_ration=pca.explained_variance_ratio_  #probablity score
exp_ration
plt.bar(range(0,43),exp_ration)
#Cumulutative plot
plt.plot(np.cumsum(exp_ration))    
plt.show()
pca_1=PCA(10)
pca_1.fit(train_so)
num_com=pca_1.n_components_
num_com
train_so_pca=pca_1.fit_transform(train_so)
test_so_pca=pca_1.fit_transform(test_so)
train_so_pca.shape
pca.fit(train_po)
exp_var2=pca.explained_variance_ 
exp_var2
exp_ration2=pca.explained_variance_ratio_  #probablity score
exp_ration2
plt.bar(range(0,42),exp_ration2)
#Cumulutative plot
plt.plot(np.cumsum(exp_ration2))    
plt.show()
pca2=PCA(2)
train_po_pca=pca2.fit_transform(train_po)
test_po_pca=pca2.fit_transform(test_po)
train_po_pca
train_new=np.concatenate((train_so_pca,train_po_pca),axis=1)
test_new=np.concatenate((test_so_pca,test_po_pca),axis=1)
train_new.shape,test_new.shape
model_log=LogisticRegression(C=10)
model_knn=KNeighborsClassifier(n_neighbors=5)
model_svc= SVC(C=10,kernel='rbf',probability=True)
model_dt=DecisionTreeClassifier()
model_rf=RandomForestClassifier(n_estimators=100)
model_log.fit(train_new,target)
model_knn.fit(train_new,target)
model_svc.fit(train_new,target)
model_dt.fit(train_new,target)
model_rf.fit(train_new,target)
result_true_arr=result_true.iloc[:].values
result_true_arr.shape,test_new.shape
target_pred_log=model_log.predict(test_new)
target_pred_knn=model_knn.predict(test_new)
target_pred_svc=model_svc.predict(test_new)
target_pred_dt=model_dt.predict(test_new)
target_pred_rf=model_rf.predict(test_new)
cm_log=confusion_matrix(result_true_arr,target_pred_log)
cr_log=classification_report(result_true_arr,target_pred_log)
cm_knn=confusion_matrix(result_true_arr,target_pred_knn)
cr_knn=classification_report(result_true_arr,target_pred_knn)
cm_svc=confusion_matrix(result_true_arr,target_pred_svc)
cr_svc=classification_report(result_true_arr,target_pred_svc)
cm_dt=confusion_matrix(result_true_arr,target_pred_dt)
cr_dt=classification_report(result_true_arr,target_pred_dt)
cm_rf=confusion_matrix(result_true_arr,target_pred_rf)
cr_rf=classification_report(result_true_arr,target_pred_rf)
print('=====Log Reg========')
plt.figure(figsize=(8,4))
sns.heatmap(cm_log,annot=True,cbar=False)
plt.show()
print('=====KNN========')
plt.figure(figsize=(8,4))
sns.heatmap(cm_knn,annot=True,cbar=False)
plt.show()
print('=====SVC========')
plt.figure(figsize=(8,4))
sns.heatmap(cm_svc,annot=True,cbar=False)
plt.show()
print('=====Discreet Tree Class========')
plt.figure(figsize=(8,4))
sns.heatmap(cm_dt,annot=True,cbar=False)
plt.show()
print('=====Random Forest========')
plt.figure(figsize=(8,4))
sns.heatmap(cm_rf,annot=True,cbar=False)
plt.show()
print('=====cr_log========')
print(cr_log)
print('=====cr_knn=========')
print(cr_knn)
print('=====cr_svc========')
print(cr_svc)
print('=====cr_dt=========')
print(cr_dt)
print('=====cr_rf========')
print(cr_rf)