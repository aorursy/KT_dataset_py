# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
train =  pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train=train[0:20000]
X= train.drop(['label'],axis=1)

y= train['label']
del train
sns.countplot(y)

y.value_counts()
X.isnull().any().describe()

test.isnull().any().describe()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


clf=MLPClassifier(solver='lbfgs',hidden_layer_sizes=350,alpha=1e-04).fit(X_train,y_train)

y_pred= clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print('The accuracy is : ', round(accuracy*100,2))

result = clf.predict(test)

result = pd.Series(result,name='label')

name=pd.Series(range(1,28001))

submission = pd.DataFrame({'ImageID':name,'label':result})

submission.to_csv('submission.csv',index=False)