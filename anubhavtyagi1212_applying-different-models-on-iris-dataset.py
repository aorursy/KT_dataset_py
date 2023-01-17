import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('../input/iris/Iris.csv')
df.head()
df.shape
df.describe()
df.info()
for i in df.columns[1:5]:
    sns.kdeplot(df[i],color='b')
    plt.show()
    
sns.countplot(df['Species'])
plt.show()
### we can see that all the species are equally distributed
plt.figure(figsize=(8,8))
sns.boxplot(y=df['PetalLengthCm'],x=df['Species'])#Iris virginca has the largest petal length followed by versicolor and setosa
plt.show()
df.drop('Id',axis=1).boxplot(by='Species',figsize=(12,6))
heat=df.drop('Id',axis=1).corr()
sns.heatmap(heat,annot=True,cmap='PuBu')
plt.show()
x=df.drop(['Id','Species'],axis=1)
x.head()
sc=StandardScaler()
num_sc=sc.fit_transform(x)
num_sc=pd.DataFrame(num_sc,columns=x.columns)
num_sc.head()
from sklearn.model_selection import train_test_split
y=df['Species']
xtrain,xtest,ytrain,ytest=train_test_split(num_sc,y,test_size=0.3,random_state=0,)
lr=LogisticRegression(solver='saga',multi_class='multinomial')
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
from sklearn import metrics
metrics.accuracy_score(ytest,ypred)
metrics.confusion_matrix(ytest,ypred)
metrics.classification_report(ytest,ypred)
from sklearn.metrics import roc_curve,auc
ypred_prob=lr.predict_proba(xtest)[:,1]
ypred_prob
lr=LogisticRegression()
kf=KFold(shuffle=True,n_splits=5,random_state=0)
res=cross_val_score(lr,num_sc,y,cv=kf,scoring='f1_weighted')
print('Bias error is ',np.mean(1-res))
print('variance error is',np.std(res))
from sklearn.model_selection import GridSearchCV
knn=KNeighborsClassifier()
param={'n_neighbors':np.arange(1,100),'weights':['uniform','distance']}
Kf=KFold(shuffle=True,n_splits=5,random_state=0)
Gs=GridSearchCV(knn,param,cv=kf,scoring='f1_weighted')
Gs.fit(num_sc,y)
Gs.best_params_
knn=KNeighborsClassifier(n_neighbors=26,weights='distance')
kf=KFold(shuffle=True,n_splits=5,random_state=0)
res_knn=cross_val_score(knn,num_sc,y,cv=kf,scoring='f1_weighted')
print('bias error is',np.mean(1-res_knn))
print('variance error is',np.std(res_knn))
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(xtrain,ytrain)
ypred=nb.predict(xtest)
metrics.accuracy_score(ytest,ypred)
## cross validating
kf=KFold(shuffle=True,n_splits=5,random_state=0)
nb_res=cross_val_score(nb,num_sc,y,cv=kf,scoring='f1_weighted')
print('bias error is ',np.mean(1-nb_res))
print('variance error is ',np.std(nb_res))
1-0.04666971190235871  # mean score accross the folds
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
param={'max_depth':np.arange(1,100),'criterion':['gini','entropy']}
kf=KFold(shuffle=True,n_splits=5,random_state=0)
gs=GridSearchCV(dt_model,param,cv=kf,scoring='f1_weighted')
gs.fit(num_sc,y)
gs.best_params_
dt_reg=DecisionTreeClassifier(max_depth=6,criterion='gini')
kf=KFold(shuffle=True,n_splits=5,random_state=0)
dt_reg_res=cross_val_score(dt_reg,num_sc,y,cv=kf,scoring='f1_weighted')
print('Bias error is',np.mean(1-dt_reg_res))
print('varaince error is ',np.std(dt_reg_res))
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
rf=RandomForestClassifier()
kf=KFold(shuffle=True,n_splits=5,random_state=0)
rf_res=cross_val_score(rf,num_sc,y,cv=kf,scoring='f1_weighted')
print('bias error is ',np.mean(1-rf_res))
print('variance error is ',np.std(rf_res))
rf_be=[]
rf_ve=[]
for n in np.arange(1,100):
    rf=RandomForestClassifier(n_estimators=n,criterion='entropy',random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    roc_auc=cross_val_score(rf,num_sc,y,cv=kf,scoring='f1_weighted')
    rf_be.append(np.mean(1-roc_auc))
    rf_ve.append(np.std(roc_auc))
np.min(rf_ve),np.argmin(rf_ve),rf_be[43]  #n_estimators=44
x_axis=np.arange(len(rf_ve))
plt.plot(x_axis,rf_ve)
plt.xlabel('range of rf_ve')
plt.ylabel('Rf_ve')
plt.show()
rf=RandomForestClassifier(n_estimators=44,criterion='entropy',random_state=0)
kf=KFold(shuffle=True,n_splits=5,random_state=0)
auc=cross_val_score(rf,num_sc,y,cv=kf,scoring='f1_weighted')
rf_be=np.mean(1-auc)
rf_ve=np.std(auc)
print(rf_be,rf_ve)
LR=LogisticRegression(solver='saga',multi_class='multinomial')
NB=GaussianNB()
KNN=KNeighborsClassifier(n_neighbors=26,weights='distance')
LR_ve=[]
LR_be=[]
for n in np.arange(1,100):
    Lr_bag=BaggingClassifier(base_estimator=LR,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(Lr_bag,num_sc,y,cv=kf,scoring='f1_weighted')
    LR_ve.append(np.mean(1-auc))
    LR_be.append(np.std(auc))
np.min(LR_ve),np.argmin(LR_ve),LR_be[14]          # n_estimator=15
x_axis=np.arange(len(LR_ve))
plt.plot(x_axis,LR_ve)
plt.show()
NB_ve=[]
NB_be=[]
for n in np.arange(1,100):
    NB_bag=BaggingClassifier(base_estimator=NB,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(NB_bag,num_sc,y,cv=kf,scoring='f1_weighted')
    NB_be.append(np.mean(1-auc))
    NB_ve.append(np.std(auc))
np.min(NB_ve),np.argmin(NB_ve),NB_be[46]
x_axis=np.arange(len(NB_ve))
plt.plot(x_axis,NB_ve)
plt.show()
# nestimators=47
KNN_ve=[]
KNN_be=[]
for n in np.arange(1,100):
    KNN_bag=BaggingClassifier(base_estimator=KNN,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(KNN_bag,num_sc,y,cv=kf,scoring='f1_weighted')
    KNN_be.append(np.mean(1-auc))
    KNN_ve.append(np.std(auc))
np.min(KNN_ve),np.argmin(KNN_ve)
x_axis=np.arange(len(KNN_ve))
plt.plot(x_axis,KNN_ve)
## n_estimators=7
LR=LogisticRegression(solver='saga',multi_class='multinomial')
LR_bag=BaggingClassifier(base_estimator=LR,n_estimators=15,random_state=0)
knn=KNeighborsClassifier(n_neighbors=26,weights='distance')
knn_bag=BaggingClassifier(base_estimator=KNN,n_estimators=7)
NB=GaussianNB()
NB_bag=BaggingClassifier(base_estimator=NB,n_estimators=47)
dt_reg=DecisionTreeClassifier(max_depth=9,criterion='gini')
rf=RandomForestClassifier(n_estimators=44,criterion='entropy',random_state=0)
models=[]
models.append(('LR',LR))
models.append(('LR_bag',LR_bag))
models.append(('knn',knn))
models.append(('knn_bag',knn_bag))
models.append(("NB",NB))
models.append(('nb_bag',NB_bag))
models.append(('decisiontree_reg',dt_reg))
models.append(('rf',rf))
results=[]
names=[]
for name,model in models:
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(model,num_sc,y,cv=kf,scoring='f1_weighted')
    results.append(auc)
    names.append(name)
    print('%s:%f(%f)' %(name,np.mean(1-auc),np.std(auc,ddof=1)))
fig=plt.figure()
fig.suptitle('Alogrith,m comaparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,rotation=90)
plt.show()
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
GB_ve=[]
GB_be=[]
for n in np.arange(1,100):
    gb=GradientBoostingClassifier(n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(gb,num_sc,y,cv=kf,scoring='f1_weighted')
    GB_be.append(np.mean(1-auc))
    GB_ve.append(np.std(auc))
np.min(GB_be),np.argmin(GB_be),GB_ve[54]
x_axis=np.arange(len(GB_be))
plt.plot(x_axis,GB_be)
plt.show()
LR=LogisticRegression(solver='saga',multi_class='multinomial')
AB_ve=[]
AB_be=[]
for n in np.arange(1,100):
    Ab=AdaBoostClassifier(base_estimator=LR,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(LR,num_sc,y,cv=kf,scoring='f1_weighted')
    AB_be.append(np.mean(1-auc))
    AB_ve.append(np.std(auc))
np.min(AB_be),np.argmin(AB_be)
## therefore no scope of boosting as n estimators is 1
NB=GaussianNB()
gb_be=[]
gb_ve=[]
for n in np.arange(1,100):
    ada=AdaBoostClassifier(base_estimator=NB,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(NB,num_sc,y,cv=kf,scoring='f1_weighted')
    gb_be.append(np.mean(1-auc))
    gb_ve.append(np.std(auc))
np.min(gb_be),np.argmin(gb_be)
RF=RandomForestClassifier()
rf_ve=[]
rf_be=[]
for n in np.arange(1,100):
    ad=AdaBoostClassifier(base_estimator=RF,n_estimators=n,random_state=0)
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(RF,num_sc,y,cv=kf,scoring='f1_weighted')
    rf_be.append(np.mean(1-auc))
    rf_ve.append(np.std(auc))
np.min(rf_be),np.argmin(rf_be)  # nestimators=23
LR=LogisticRegression(solver='saga',multi_class='multinomial')
LR_bag=BaggingClassifier(base_estimator=LR,n_estimators=23,random_state=0)
knn=KNeighborsClassifier(n_neighbors=26,weights='distance')
knn_bag=BaggingClassifier(base_estimator=KNN,n_estimators=7)
NB=GaussianNB()
NB_bag=BaggingClassifier(base_estimator=NB,n_estimators=47)
dt_reg=DecisionTreeClassifier(max_depth=9,criterion='gini')
rf=RandomForestClassifier(n_estimators=44,criterion='entropy',random_state=0)
RF_BOOST=RandomForestClassifier(n_estimators=23,criterion='entropy',random_state=0)
models=[]
models.append(('LR',LR))
models.append(('LR_bag',LR_bag))
models.append(('knn',knn))
models.append(('knn_bag',knn_bag))
models.append(("NB",NB))
models.append(('nb_bag',NB_bag))
models.append(('decisiontree_reg',dt_reg))
models.append(('rf',rf))
models.append(('RF_BOOST',RF_BOOST))
results=[]
names=[]
for name,model in models:
    kf=KFold(shuffle=True,n_splits=5,random_state=0)
    auc=cross_val_score(model,num_sc,y,cv=kf,scoring='f1_weighted')
    results.append(auc)
    names.append(name)
    print('%s:%f(%f)' %(name,np.mean(1-auc),np.std(auc,ddof=1)))
fig=plt.figure()
fig.suptitle('Alogrith,m comaparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names,rotation=90)
plt.show()
