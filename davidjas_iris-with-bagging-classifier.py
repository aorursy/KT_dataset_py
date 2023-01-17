import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
iris=load_iris()

data=pd.DataFrame(iris.data,columns=iris.feature_names)

data['species']=iris.target

data.head()
#getting to know missing values

import missingno as mn

mn.matrix(data)
data.columns
plt.figure()

fig,ax=plt.subplots(2,2,figsize=(15,6))





sns.distplot(data['sepal length (cm)'],ax=ax[0][0],hist=True,kde=True,

            bins='auto',color='darkblue',

            hist_kws={'edgecolor':'black'},

            kde_kws={'linewidth':4})

sns.distplot(data['sepal width (cm)'],ax=ax[0][1],hist=True,kde=True,

            bins='auto',color='darkblue',

            hist_kws={'edgecolor':'black'},

            kde_kws={'linewidth':4})

sns.distplot(data['petal length (cm)'],ax=ax[1][0],hist=True,kde=True,

            bins='auto',color='darkblue',

            hist_kws={'edgecolor':'black'},

            kde_kws={'linewidth':4})

sns.distplot(data['petal width (cm)'],ax=ax[1][1],hist=True,kde=True,

            bins='auto',color='darkblue',

            hist_kws={'edgecolor':'black'},

            kde_kws={'linewidth':4})
formatter=plt.FuncFormatter(lambda i,*args: iris.target_names[int(i)])

plt.figure(figsize=(15,8))

plt.scatter(np.array(data.iloc[:,0]),np.array(data.iloc[:,1]),c=data.species)

plt.colorbar(ticks=[0,1,2],format=formatter)

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Sepal Width (cm)')

plt.show()
X=data.iloc[:,0:4].values

y=data.iloc[:,4].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

pipeline=make_pipeline(StandardScaler(),DecisionTreeClassifier(criterion='entropy',max_depth=4))
pipeline.fit(X_train,y_train)
pipeline.score(X_train,y_train),pipeline.score(X_test,y_test)
pipeline=make_pipeline(StandardScaler(),DecisionTreeClassifier(criterion='entropy',max_depth=4))
from sklearn.ensemble import BaggingClassifier

bgclf=BaggingClassifier(base_estimator=pipeline,n_estimators=100,max_samples=10,random_state=1,n_jobs=5)
bgclf.fit(X_train,y_train)
bgclf.score(X_train,y_train),bgclf.score(X_test,y_test)
y_train_pred=bgclf.predict(X_train)

y_test_pred=bgclf.predict(X_test)

y_test_pred
cm_train=metrics.confusion_matrix(y_train_pred,y_train)

print(cm_train)

sns.heatmap(cm_train,annot=True)
cm_test=metrics.confusion_matrix(y_test_pred,y_test)

print(cm_test)

sns.heatmap(cm_test,annot=True,cmap='Blues')
metrics.accuracy_score(y_test_pred,y_test)
import graphviz
!pip install pydotplus
import pydotplus

from IPython.display import Image
clf=DecisionTreeClassifier(min_samples_leaf=20,max_depth=5)

clf.fit(X_train,y_train)

from sklearn import tree

dot_data=tree.export_graphviz(clf,out_file=None,feature_names=iris.feature_names,filled=True)
graph=pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
clftree2=DecisionTreeClassifier(min_samples_leaf=20,max_depth=5)

clftree2.fit(X_train,y_train)

dot_data=tree.export_graphviz(clftree2,out_file=None,feature_names=iris.feature_names,filled=True)

graph=pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())