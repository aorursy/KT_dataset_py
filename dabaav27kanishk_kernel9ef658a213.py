# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
train.describe()
train.info()
songtit=pd.get_dummies(pd.concat((train['songtitle'],test['songtitle']),axis=0),prefix='songtitle',drop_first=True)

artistna=pd.get_dummies(pd.concat((train['artistname'],test['artistname']),axis=0),prefix='artistname',drop_first=True)

sID=pd.get_dummies(pd.concat((train['songID'],test['songID']),axis=0),prefix='songID',drop_first=True)

aID=pd.get_dummies(pd.concat((train['artistID'],test['artistID']),axis=0),prefix='artistID',drop_first=True)
train
y_train=train.Top10
train=train.drop(['songtitle','songID','artistname','artistID','Top10'],1)
test=test.drop(['songtitle','songID','artistname','artistID'],1)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

train=pd.DataFrame(ss.fit_transform(train),columns=train.columns)

test=pd.DataFrame(ss.fit_transform(test),columns=test.columns)
train.head()
test.head()
train.info()
test.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import roc_auc_score,confusion_matrix
X_train,X_test,Y_train,Y_test=train_test_split(train,y_train,test_size=0.001,random_state=42)
lr=LogisticRegression(C=0.3,random_state=42)

lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)
cm=confusion_matrix(Y_test,Y_pred)

cm
roc_auc_score(Y_test,Y_pred)
from sklearn import metrics
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 4))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    #plt.xlim([0.0, 1.0])

    #plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(Y_test,Y_pred)
y_pred=lr.predict(test)

final=pd.DataFrame({})

final["songID"]=pd.read_csv("../input/test.csv").songID

final["Top10"]=y_pred.T
final.index+=1

final
final.to_csv("final_sub5.csv",index=False)