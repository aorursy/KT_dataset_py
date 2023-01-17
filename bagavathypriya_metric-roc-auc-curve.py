# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import make_classification
x,y=make_classification(n_samples=2000,n_features=20,n_classes=2,weights=[1,1],random_state=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
rfmod=RandomForestClassifier()
rfmod.fit(xtrain,ytrain)
ytrainpred=rfmod.predict_proba(xtrain)
print("Random Forest train roc-auc score {}".format(roc_auc_score(ytrain,ytrainpred[:,1])))
ytestpred=rfmod.predict_proba(xtest)
print("Random Forest train roc-auc score {}".format(roc_auc_score(ytest,ytestpred[:,1])))
from sklearn.linear_model import LogisticRegression
logmod=LogisticRegression()
logmod.fit(xtrain,ytrain)
ytrainpred=logmod.predict_proba(xtrain)
print("Logistic regression train roc-auc score {}".format(roc_auc_score(ytrain,ytrainpred[:,1])))
ytestpred=logmod.predict_proba(xtest)
print("Logistic regression train roc-auc score {}".format(roc_auc_score(ytest,ytestpred[:,1])))
from sklearn.ensemble import AdaBoostClassifier
adamod=AdaBoostClassifier()
adamod.fit(xtrain,ytrain)
ytrainpred=adamod.predict_proba(xtrain)
print("Ada Boost train roc-auc score {}".format(roc_auc_score(ytrain,ytrainpred[:,1])))
ytestpred=adamod.predict_proba(xtest)
print("Ada Boost train roc-auc score {}".format(roc_auc_score(ytest,ytestpred[:,1])))
from sklearn.neighbors import KNeighborsClassifier
knnmod=KNeighborsClassifier()
knnmod.fit(xtrain,ytrain)
ytrainpred=knnmod.predict_proba(xtrain)
print("K Nearest Neighbor Classifier  train roc-auc score {}".format(roc_auc_score(ytrain,ytrainpred[:,1])))
ytestpred=knnmod.predict_proba(xtest)
print("K Nearest Neighbor Classifier train roc-auc score {}".format(roc_auc_score(ytest,ytestpred[:,1])))
pred=[]
for model in [rfmod,logmod,adamod,knnmod]:
    pred.append(pd.Series(model.predict_proba(xtest)[:,1]))
finalpred=pd.concat(pred,axis=1).mean(axis=1)
print("Ensemble models test roc-auc score {}".format(roc_auc_score(ytest,finalpred)))
fpr,tpr,threshold= roc_curve(ytest,finalpred)
threshold
from sklearn.metrics import accuracy_score
acc=[]
for thres in threshold:
    ypred=np.where(finalpred>thres,1,0)
    acc.append(accuracy_score(ytest,ypred,normalize=True))
acc=pd.concat([pd.Series(acc),pd.Series(threshold)],axis=1)
acc.columns=['Accuracy','Threshold']
acc.sort_values(by='Accuracy',ascending=False,inplace=True)
acc.head()
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
plot_roc_curve(fpr,tpr)
