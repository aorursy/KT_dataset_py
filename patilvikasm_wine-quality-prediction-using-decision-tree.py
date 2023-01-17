# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dir_name,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dir_name,filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wine=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
wine.head()
wine.tail()
wine.shape
wine.columns
wine.info()
wine.describe()
import seaborn as sns
sns.countplot(wine["quality"])
import pandas_profiling 
pandas_profiling.ProfileReport(wine)
#dataset has 240 15% duplicate rows so remove this rows
wine.drop_duplicates(inplace=True)
#1599-240
wine.shape
#insight ofdata after removing duplicate rows
pandas_profiling.ProfileReport(wine)
#find correlation between features
corr=wine.corr()
sns.heatmap(corr,annot=True)
def qualitymap(num):
        if num<6.5:
            return "bad"
        return "good"
            
wine["Qualitymap"]=wine["quality"].apply(qualitymap)        

wine["Qualitymap"].value_counts()
qual=pd.get_dummies(wine["Qualitymap"],drop_first=True)

wine["Qual"]=qual
wine.drop(["quality","Qualitymap"],inplace=True,axis=1)
sns.countplot(x=wine.Qual,data=wine)
X=wine.iloc[:,:-1]
y=wine.iloc[:,-1:]
X.shape,y.shape
#create train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=14,test_size=0.30)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_leaf_nodes=3)
model.fit(x_train,y_train)
model.score(x_test,y_test)

from sklearn.model_selection import cross_val_score
cross_val_score(model, x_test,y_test,cv=10)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, model.predict(x_test))
cm

from sklearn.metrics import classification_report
classification_report(y_test, model.predict(x_test))
from sklearn.metrics import auc, roc_auc_score, roc_curve
#find roc_auc_score
y_train_prob=model.predict_proba(x_train)
y_test_prob=model.predict_proba(x_test)
print("Train probability is: {}".format(roc_auc_score(y_train,y_train_prob[:,1:])))
print("Test probability is: {}".format(roc_auc_score(y_test,y_test_prob[:,1:])))
fpr, tpr, thresholds=roc_curve(y_test, y_test_prob[:,1:])

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,linestyle="--",label="Decision Tree AUROC=%0.3f" % roc_auc_score(y_test, y_test_prob[:,1:]))
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
import graphviz
from sklearn import tree
dot_data=tree.export_graphviz(model)
graph=graphviz.Source(dot_data)
graph
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
rf_y_train_prob=rf.predict_proba(x_train)
rf_y_test_prob=rf.predict_proba(x_test)
print("Train probability is: {}".format(roc_auc_score(y_train,rf_y_train_prob[:,1:])))
print("Test probability is: {}".format(roc_auc_score(y_test,rf_y_test_prob[:,1:])))
fpr, tpr, thresholds=roc_curve(y_test, rf_y_test_prob[:,1:])
import matplotlib.pyplot as plt
plt.plot(fpr,tpr,linestyle="--",label="Random Forest AUROC=%0.3f" % roc_auc_score(y_test, rf_y_test_prob[:,1:]))
plt.plot([0,1],[0,1])
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
thresholds
from sklearn.metrics import accuracy_score
acc_ls=[]
for thres in thresholds:
  y_pred=np.where(y_test_prob[:,1:]>thres,1,0)
  acc_ls.append(accuracy_score(y_test,y_pred, normalize=True))
acc_ls
acc_ls1=pd.concat([pd.Series(thresholds),pd.Series(acc_ls)],axis=1)
acc_ls1
acc_ls1.columns=["Threshold","Accuracy"]
acc_ls1.sort_values(by="Accuracy",ascending=False,inplace=True)
acc_ls1.head()
