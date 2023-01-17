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
train = pd.read_csv('../input/train.csv') 

test= pd.read_csv('../input/test.csv')
train.head(20)
x= train.drop(['id', 'species'], axis=1).values

y= train['species']
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
X_train,X_test,y_train,y_test =train_test_split(x,y)

le = LabelEncoder().fit(train['species'])

scaler=MinMaxScaler()

X_tranf=scaler.fit_transform(X_train)

X_testf=scaler.transform(X_test)

classes= list(le.classes_)
classes
print(X_train.shape)

print(y_train.shape)
mlp = MLPClassifier(solver='adam',hidden_layer_sizes=[20,20],random_state=1).fit(X_tranf,y_train)

y_pred=mlp.predict(X_testf)
accuracy=accuracy_score(y_pred,y_test)

print('The accuracy is : ',round(accuracy*100,2))
test.head()
test_ids=test['id']
test=test.drop(['id'],axis=1)
result = mlp.predict_proba(test)

Submit = pd.DataFrame(result, columns=classes)

Submit.insert(0, 'id', test_ids)

Submit.reset_index()





#submission.to_csv('submission.csv', index = False)

Submit.to_csv('leaf_classification.csv', index = False)

Submit.tail()