#Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
cancer.keys()
print(cancer['feature_names'])
cancer['data'].shape
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
df_cancer.head()
sns.countplot(df_cancer['target'])
#Seperating X and Y values
X=df_cancer.values[:,:-1]
Y=df_cancer.values[:,-1]
#Seperating Train and Test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#Performing feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#Model Building
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=10,gamma=0.1)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(Y_pred)
#Checking confusion matrix and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print('classification report:')

print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc)
#Heatmap of confusion matrix
sns.heatmap(cfm,annot=True)

