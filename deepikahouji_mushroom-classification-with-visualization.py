import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
mush = pd.read_csv("../input/mushrooms.csv")
mush.head()
sns.heatmap(mush.isna(),cmap='coolwarm')
# there is no missing data
mush.describe()
X = mush.drop('class',axis=1)
y = mush['class']
y.head()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in X.columns:
    X[col] = labelencoder.fit_transform(X[col])
X.head()
y = labelencoder.fit_transform(y)
y
# poisonous =1
# edible =0
X = pd.get_dummies(X, columns=X.columns)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
pred = dtc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
dtc.fit(X_train,y_train)
preddtc = dtc.predict(X_test)
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(16,8))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,dtc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set Decision Tree")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(16,8))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,dtc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set Decision Tree")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
predrfc = rfc.predict(X_test)
print(confusion_matrix(y_test,predrfc))
print(classification_report(y_test,predrfc))
accuracy_score(y_test,predrfc)
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(16,8))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,rfc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set Random Forest")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(16,8))
from matplotlib.colors import ListedColormap
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,rfc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set Random Forest")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svcpred = svc.predict(X_test)
print(confusion_matrix(y_test,svcpred))
print(classification_report(y_test,svcpred))
accuracy_score(y_test,svcpred)
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,svc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set Support Vector Classifier")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,svc.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set Support Vector Classifier")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train,y_train)
predlg = lg.predict(X_test)
print(confusion_matrix(y_test,predlg))
print(classification_report(y_test,predlg))
accuracy_score(y_test,predlg)
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,lg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set Logistic Regression")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,lg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set Logistic Regression")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,y_train)
prednb = NB.predict(X_test)
print(confusion_matrix(y_test,prednb))
print(classification_report(y_test,prednb))
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,NB.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set Naive bayes")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,NB.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set Naive bayes")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
predknn = knn.predict(X_test)
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))
accuracy_score(y_test,predknn)
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,knn.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Training set KNN")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
sns.set_context('notebook',font_scale=2)
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)     )
plt.contourf(X1,X2,knn.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.5,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[j,0],X_set[j,1],cmap=ListedColormap(('red','green'))(i),label=j)
plt.title("Test set KNN")
plt.xlabel('PC 1')
plt.ylabel('PC 2')
