# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
import os

os.listdir('../input/ghouls-goblins-and-ghosts-boo')

train_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip', compression='zip')

test_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip', compression='zip')



print ('Training data features:')

print (train_data.columns)

print ('\nTest data features:')

print (test_data.columns)
#Training Dataset

print (train_data.shape,'\n')

print (train_data.head(),'\n')

print (train_data.tail(),'\n')

print (train_data.describe(),'\n')
#Test Dataset

print (test_data.shape,'\n')

print (test_data.head(),'\n')

print (test_data.tail(),'\n')

print (test_data.describe(),'\n')
trainid=train_data.values[:,0]

testid=test_data.values[:,0]



y= train_data['type']

x = pd.get_dummies(train_data.drop(['color','type','id'], axis = 1))



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



X=np.array(X_train)

y=np.array(y_train)
#MLP Classifier

alg=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=0).fit(X,y)

print(alg)

y_pred=alg.predict(X_test)
#Training Accuracy

y_pred=alg.predict(X_test)

alg.score(X_test,y_test)
#Validation Accuracy

from sklearn.model_selection import cross_val_score

scores= cross_val_score(alg, X, y, cv=5)

scores

scores.mean()
test=test_data.drop(['color','id'],axis=1)

prediction=alg.predict(test)



submission=pd.DataFrame({'id':testid, 'type': prediction})

submission.to_csv("submission.csv",index=False)
