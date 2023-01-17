import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

import shap

import xgboost
df_train=pd.read_csv("../input/novartis-data/Train.csv")

df_test=pd.read_csv("../input/novartis-data/Test.csv")
df_train.head()
df_train.isnull().sum()
df_test.isnull().sum()
plt.figure(figsize=(20,18))

df_train.iloc[:,2:-1].boxplot()
corr = df_train.iloc[:,2:-1].corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn.impute import KNNImputer
k=int(round(len(df_train)**0.5,0))

if k%2==0:

    k=k+1

k
imputer = KNNImputer(n_neighbors=k)
X=df_train.iloc[:,2:-1].values

Y=df_train.iloc[:,-1].values

x_test=df_test.iloc[:,2:].values
X=imputer.fit_transform(X)

x_test=imputer.transform(x_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(X)

x_test=sc.transform(x_test)

np.corrcoef(X[:,9],X[:,11])

np.corrcoef(X[:,1],X[:,2])
X_reformed=np.delete(X,(2,11),axis=1)

x_test_reformed=np.delete(x_test,(2,11),axis=1)
from sklearn.cluster import KMeans
iner=[]

count=[]
for i in range(1,8):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(X_reformed)

    inertia=kmeans.inertia_

    count.append(i)

    iner.append(inertia)

    
count=np.array(count)

iner=np.array(iner)
plt.plot(count,iner)
from sklearn.decomposition import PCA

explained_variance=[]

count1=[]
pca=PCA(n_components=10)

pca.fit(X_reformed)

explained_variance1=pca.explained_variance_ratio_

plt.plot(explained_variance1)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
p=sum(Y)/len(Y)

print(p)
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, Y_resampled = ros.fit_resample(X_reformed, Y)
p=sum(Y_resampled)/len(Y_resampled)

print(p)
lr.fit(X_resampled,Y_resampled)
y=lr.predict(X_resampled)
from sklearn.metrics import accuracy_score, auc, confusion_matrix,f1_score, roc_curve, roc_auc_score
confusion_matrix(Y_resampled,y)
from sklearn.model_selection import cross_val_score

from sklearn import metrics

scores = cross_val_score(lr,X_resampled, Y_resampled, cv=10, scoring='recall')
scores
fpr, tpr, thresholds = metrics.roc_curve(Y_resampled, y)

roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='logistic')

display.plot()  

plt.show() 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
param= {

    'bootstrap': [True],

    'max_depth': [3,4,5,6,7,8],

    'max_features': [5,6,7,8,9,10],

    'min_samples_leaf': [5,6,7,8,9,10],

    'min_samples_split': [20,25,50],

    'n_estimators': [500,1000],

    'criterion':["gini","entropy"]

}
random=RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=param,n_iter=10,cv=3,n_jobs=-1)

random.fit(X_resampled,Y_resampled)
search=random.fit(X_resampled,Y_resampled)

search.best_params_
y=random.predict(X_resampled)
scores = cross_val_score(random,X_resampled, Y_resampled, cv=10, scoring='recall',n_jobs=-1)

scores
fpr, tpr, thresholds = metrics.roc_curve(Y_resampled, y)

roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='randomforest')

display.plot()  

plt.show()  
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=1000)
ada.fit(X_resampled,Y_resampled)
scores = cross_val_score(ada,X_resampled, Y_resampled, cv=10, scoring='recall',n_jobs=-1)

scores
y=ada.predict(X_resampled)

fpr, tpr, thresholds = metrics.roc_curve(Y_resampled, y)

roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='Adaboost')

display.plot()  

plt.show() 
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=1000)

gbc.fit(X_resampled,Y_resampled)
scores = cross_val_score(gbc,X_resampled, Y_resampled, cv=10, scoring='recall',n_jobs=-1)

scores
y=ada.predict(x_test_reformed)

xcv={"INCIDENT_ID":df_test.iloc[:,0],"MULTIPLE_OFFENSE":y}

sample=pd.DataFrame(xcv)

sample.to_csv("Sample.csv",index=False)