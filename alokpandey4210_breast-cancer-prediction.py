# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

df.head()
df['diagnosis'].value_counts()
df.corr(method='pearson')
X=df.drop(['diagnosis'],axis=1)
y=df['diagnosis']
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y=lb.fit_transform(y)

type(y)



X.isnull().sum()
X['mean_radius'].hist(bins=50)
X.boxplot(column='mean_radius',by='mean_texture')
X['mean_radius'].plot('density',color='Red')
X['mean_texture'].plot('density',color='Green')
X['mean_texture'].plot('density',color='Pink')
X['mean_perimeter'].hist(bins=50)
X.boxplot(column='mean_perimeter',by='mean_texture')
X['mean_texture'].plot('density',color='Blue')
X['mean_area'].hist(bins=50)
X.boxplot(column='mean_area',by='mean_perimeter')
X['mean_area'].plot('density',color='Pink')
X.isnull().sum()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

                                
from sklearn.svm import SVC

p=SVC()

p.fit(X_train,y_train)

p.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

k=DecisionTreeClassifier()

k.fit(X_train,y_train)

k.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

n=RandomForestClassifier()

n.fit(X_train,y_train)

n.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesClassifier

o=ExtraTreesClassifier()

o.fit(X_train,y_train)

o.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

LG=LogisticRegression()

LG.fit(X_train,y_train)

LG.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

r=KNeighborsClassifier()

r.fit(X_train,y_train)

r.score(X_test,y_test)
#save model

import pickle 

file_name='Cancer.sav'

tuples=(n,X)

pickle.dump(tuples,open(file_name,'wb'))
from sklearn.metrics import confusion_matrix

ycm1=p.predict(X_test)

result=confusion_matrix(y_test,ycm1)

result
from sklearn.metrics import confusion_matrix

ycm2=k.predict(X_test)

result1=confusion_matrix(y_test,ycm2)

result1
from sklearn.metrics import confusion_matrix

ycm3=n.predict(X_test)

result2=confusion_matrix(y_test,ycm3)

result2
from sklearn.metrics import confusion_matrix

ycm4=o.predict(X_test)

result3=confusion_matrix(y_test,ycm4)

result3
from sklearn.metrics import confusion_matrix

ycm5=LG.predict(X_test)

result4=confusion_matrix(y_test,ycm5)

result4
from sklearn.metrics import confusion_matrix

ycm6=r.predict(X_test)

result5=confusion_matrix(y_test,ycm6)

result5
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,k.predict_proba(X_test)[:,1])
fpr
tpr
threshold
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr1,tpr1,threshold1=roc_curve(y_test,n.predict_proba(X_test)[:,1])
fpr1
tpr1
threshold1
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr2,tpr2,threshold2=roc_curve(y_test,o.predict_proba(X_test)[:,1])
fpr2
tpr2
threshold2
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr3,tpr3,threshold3=roc_curve(y_test,LG.predict_proba(X_test)[:,1])
fpr3
tpr3
from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr4,tpr4,threshold4=roc_curve(y_test,r.predict_proba(X_test)[:,1])
fpr4
tpr4
threshold4
roc_auc1=roc_auc_score(y_test,k.predict(X_test))

roc_auc2=roc_auc_score(y_test,n.predict(X_test))

roc_auc3=roc_auc_score(y_test,o.predict(X_test))

roc_auc4=roc_auc_score(y_test,LG.predict(X_test))

roc_auc5=roc_auc_score(y_test,r.predict(X_test))

print(roc_auc1,'',roc_auc2,'',roc_auc3,'',roc_auc4,'',roc_auc5)
yPPPPP=o.predict_proba(X_test)
yPPPPP
from sklearn.preprocessing import binarize

wcv=binarize(yPPPPP,0.60)

wcv
LOC=wcv[:,1]
TYPE=LOC.astype(int)
from sklearn.metrics import confusion_matrix

CF=confusion_matrix(y_test,TYPE)
CF
from sklearn.preprocessing import binarize

wcv=binarize(yPPPPP,0.70)

wcv
LOC=wcv[:,1]
TYPE=LOC.astype(int)
from sklearn.metrics import confusion_matrix

CF=confusion_matrix(y_test,TYPE)

CF
from sklearn.preprocessing import binarize

wcv=binarize(yPPPPP,0.70)

wcv
from sklearn.preprocessing import binarize

wcv=binarize(yPPPPP,0.80)

wcv
LOC=wcv[:,1]
TYPE=LOC.astype(int)
from sklearn.metrics import confusion_matrix

CF=confusion_matrix(y_test,TYPE)

CF
from sklearn.preprocessing import binarize

wcv=binarize(yPPPPP,0.90)

wcv
LOC=wcv[:,1]
TYPE=LOC.astype(int)
from sklearn.metrics import confusion_matrix

CF=confusion_matrix(y_test,TYPE)

CF