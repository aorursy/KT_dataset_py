# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/creditcard.csv")
df.head(3)
df.shape
frac =0.85

x_train = df.sample(frac=frac)
x_test=df.loc[~df.index.isin(x_train.index)]
x_test.shape
y_train=x_train['Class']

x_train=x_train.drop(['Class'],axis=1)

y_test=x_test['Class']

x_test=x_test.drop(['Class'],axis=1)
x_train.shape,x_test.shape
deep=np.array(range(1,20))

xvalMSE =[]

score_hist=[]

for ideep in deep:

    dt=DecisionTreeClassifier(max_depth = ideep)

    dt.fit(x_train,y_train)

    pred=dt.predict(x_test)

    pred=np.array(pred,dtype=np.float32)

    y_tst=np.array(y_test,dtype=np.float32)

    error=[y_tst[r] - pred[r] for r in range(len(y_tst))]

    o_error=sum(e*e for e in error)

    mse=o_error/len(y_test)

    #print(mse)

    xvalMSE.append(mse)

    score=roc_auc_score(y_test,pred)

    #print(score)

    score_hist.append(score) 
plt.figure()

plt.plot(deep,xvalMSE)

plt.axis("tight")

plt.xlabel("Tree Depth")

plt.ylabel("MSE")

opt_depth = deep[np.argmin(xvalMSE)]

plt.plot(opt_depth,np.amin(xvalMSE),"or")
plt.figure()

plt.plot(deep,score_hist)

plt.axis("tight")

plt.xlabel("Tree Depth")

plt.ylabel("ROC AUC Scores")

sc_max=deep[np.argmax(score_hist)]

plt.plot(sc_max,np.amax(score_hist),"r^")

plt.show()