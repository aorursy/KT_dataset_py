import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

import plotly as py

from plotly.offline import init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

cf.go_offline()
from sklearn.datasets import load_breast_cancer
b_cancer=load_breast_cancer()
# b_cancer
b_cancer.target_names
b_cancer["feature_names"]
df_cancer=pd.DataFrame(np.c_[b_cancer["data"],b_cancer["target"]],columns=np.append(b_cancer["feature_names"],["target"]))
df_cancer.head()
sns.countplot(df_cancer["target"]) # 0=ma 1=bei
plt.figure(figsize=(20,10))

sns.heatmap(df_cancer.corr(),annot=True)
X=df_cancer.drop(["target"],axis=1) # Features
X.head()
y=df_cancer["target"] #Targets
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
cancer_model=SVC()
cancer_model.fit(X_train,y_train)
predict=cancer_model.predict(X_test)
predict
cm=confusion_matrix(y_test,predict)
sns.heatmap(cm,annot=True)

cm
print(classification_report(y_test,predict))
min_test=X_test.min()

range_test=(X_test - min_test).max()



X_test_scaled=(X_test - min_test)/range_test
min_train=X_train.min()

range_train=(X_train - min_train).max()



X_train_scaled=(X_train - min_train)/range_train
normalized_model=SVC()
normalized_model.fit(X_train_scaled,y_train)
normalized_model_predit=normalized_model.predict(X_test_scaled)
cm1=confusion_matrix(y_test,normalized_model_predit)
sns.heatmap(cm1,annot=True)

cm1
print(classification_report(y_test,normalized_model_predit))
par={"C":[0.1,1,5,10,100],"gamma":[1,0.1,0.01,0.001],"kernel":["rbf"]}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),par,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
gridcv_predict=grid.predict(X_test_scaled)
cm2=confusion_matrix(y_test,gridcv_predict)
sns.heatmap(cm2,annot=True)

cm2
print(classification_report(y_test,gridcv_predict))