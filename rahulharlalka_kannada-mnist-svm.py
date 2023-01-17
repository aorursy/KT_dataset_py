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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np

import pandas as pd

train_dataset=pd.read_csv("../input/Kannada-MNIST/train.csv")

test_dataset=pd.read_csv("../input/Kannada-MNIST/test.csv")

ids=test_dataset[['id']]

test_dataset.drop(labels='id',axis=1,inplace=True)
X=train_dataset.iloc[:,1:].values

Y=train_dataset.iloc[:,0].values
sns.countplot(Y)
from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(X,Y,test_size=0.25,random_state=0)
sns.countplot(y_train)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)

x_test = sc_X.transform(x_test)

sc_test = sc_X.transform(test_dataset)

from sklearn.svm import SVC

print('SVM Classifier with gamma = 0.1; Kernel = Polynomial')

classifier = SVC(gamma=0.1, kernel='poly', random_state = 0,C=2,verbose=1)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix

model_acc = classifier.score(x_test, y_test)

test_acc = accuracy_score(y_test, y_pred)

conf_mat = confusion_matrix(y_test,y_pred)
print(test_acc)

print(conf_mat)
submit_label=classifier.predict(sc_test)



result=pd.DataFrame(submit_label)

#print(result)

result.rename(columns={0:"label"},inplace=True)

result.head()

submission=pd.concat([ids,result],axis=1)



submission.to_csv('submission.csv',index=False)