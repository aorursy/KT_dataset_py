# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')



from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,accuracy_score
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.describe()
df.info()
df.isnull().any()
plt.figure(figsize=(7,5))

sns.countplot('target',data=df,palette='viridis')

df['target'].value_counts()
plt.figure(figsize=(14,6))

sns.countplot('age',hue='target',data=df,palette='viridis')
plt.figure(figsize=(14,6))

sns.heatmap(df.corr(),annot=True,linecolor='black',linewidths=0.01)
fig, axis=plt.subplots(1,2,figsize=(14,6))

_=sns.countplot('sex',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['sex'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
fig, axis=plt.subplots(1,2,figsize=(14,6))

_=sns.countplot('cp',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['cp'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
fig, axis=plt.subplots(1,2,figsize=(14,6))

_=sns.countplot('fbs',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['fbs'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
explode=[0.03,0.03,0.03]

fig, axis=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('restecg',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['restecg'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%',explode=explode)
fig, axis=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('exang',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['exang'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
fig, axis=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('slope',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['slope'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%',explode=explode)
fig, axis=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('ca',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['ca'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
fig, axis=plt.subplots(1,2,figsize=(14,7))

_=sns.countplot('thal',hue='target',data=df,palette='viridis',ax=axis[0])

_=df['thal'].value_counts().plot.pie(ax=axis[1],autopct='%.2f%%')
plt.figure(figsize=(10,6))

plt.scatter(df.age[df['target']==0],df.thalach[df['target']==0])

plt.scatter(df.age[df['target']==1],df.thalach[df['target']==1])
plt.figure(figsize=(10,6))

sns.scatterplot(x='chol',y='thalach',data=df,hue='target')
plt.figure(figsize=(10,6))

sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
data=pd.get_dummies(df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
scaler=MinMaxScaler()

data_scaled=scaler.fit_transform(data)

data_scaled=pd.DataFrame(data_scaled, columns=data.columns)
X=data_scaled.drop('target',axis=1)

y=data_scaled['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
knn=KNeighborsClassifier()

params={'n_neighbors':range(1,21),'weights':['uniform','distance'],'leaf_size':range(1,21),'p':[1,2,3,4,5,6,7,8,9,10]}
gs_knn=GridSearchCV(knn,param_grid=params,n_jobs=-1)
gs_knn.fit(X_train,y_train)

gs_knn.best_params_
prediction=gs_knn.predict(X_test)
acc_knn=accuracy_score(y_test,prediction)

print(acc_knn)

cm=confusion_matrix(y_test,prediction)
class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(cm),annot=True,cmap='GnBu',fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for K-Nearest Neighbors Model',y=1.1)
probability=gs_knn.predict_proba(X_test)[:,1]
fpr_knn,tpr_knn,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(10,6))

plt.title('Receiver Operating Characteristic Curve')

plt.plot(fpr_knn,tpr_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='0.5')

plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)
lr=LogisticRegression()

params={'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100],'class_weight':['balanced',None]}
gs_r=GridSearchCV(lr,param_grid=params,n_jobs=-1)
gs_r.fit(X_train,y_train)

gs_r.best_params_
prediction=gs_r.predict(X_test)
acc_lr=accuracy_score(y_test,prediction)

print(acc_lr)

cm=confusion_matrix(y_test,prediction)
class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(cm),annot=True,cmap='GnBu',fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression',y=1.1)
probability=gs_r.predict_proba(X_test)[:,1]
fpr_lr,tpr_lr,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(10,6))

plt.title('Receiver Operating Characteristic Curve')

plt.plot(fpr_lr,tpr_lr)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='0.5')

plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)
rfc=RandomForestClassifier()

params={'max_features':['auto','sqrt','log2'],'min_samples_split':[2,3,4,5,6,7,8,9,10],

        'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
gs_rfc=GridSearchCV(rfc,param_grid=params,n_jobs=-1)
gs_rfc.fit(X_train,y_train)

gs_rfc.best_params_
prediction=gs_rfc.predict(X_test)
acc_rfc=accuracy_score(y_test,prediction)

print(acc_knn)

cm=confusion_matrix(y_test,prediction)
class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(cm),annot=True,cmap='GnBu',fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Random Forest Classifier',y=1.1)
probability=gs_rfc.predict_proba(X_test)[:,1]
fpr_rfc,tpr_rfc,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(10,6))

plt.title('Receiver Operating Characteristic Curve')

plt.plot(fpr_rfc,tpr_rfc)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='0.5')

plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)
tree=DecisionTreeClassifier()

params={'max_features':['auto','sqrt','log2'],'min_samples_split':[2,3,4,5,6,7,8,9,10],

        'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
gs_tree=GridSearchCV(tree,param_grid=params,n_jobs=-1)
gs_tree.fit(X_train,y_train)

gs_tree.best_params_
prediction=gs_tree.predict(X_test)
acc_tree=accuracy_score(y_test,prediction)

print(acc_tree)

cm=confusion_matrix(y_test,prediction)
class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(cm),annot=True,cmap='GnBu',fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Decision Tree Classifier',y=1.1)
probability=gs_tree.predict_proba(X_test)[:,1]
fpr_tree,tpr_tree,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(10,6))

plt.title('Receiver Operating Characteristic Curve')

plt.plot(fpr_tree,tpr_tree)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='0.5')

plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)
result=pd.DataFrame({'Models':['K-Neighbors Classifiers','Logistic Regression','Random Forest Classifier','Decision Tree Classifier'],

                    'Score':[acc_knn,acc_lr,acc_rfc,acc_tree]})

result.sort_values(by='Score',ascending=False)