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
data = pd.read_csv('../input/train.csv')

# print(data.head(10))
y = data['label']
X = data.drop(['label'],axis=1)


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

train_X , test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42)

clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(200,200,200),random_state=1)

print(clf)

clf.fit(train_X,train_y)

preds = clf.predict(test_X)



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y,preds))

test_data = pd.read_csv('../input/test.csv')
# print(test_data.index.values)

sub_preds = clf.predict(test_data)

subs_data = pd.DataFrame({'Label':sub_preds})
subs_data.index+=1
subs_data.index.name='ImageId'

subs_data.to_csv('submission.csv',index=True)
