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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train.info()
test.info()
def feat_eng(data):
  features = data[['Pclass', 'Sex', 'Age','SibSp', 'Parch']].copy()
  features['Age'].fillna(features['Age'].mean(),inplace = True)
  features[['Sex']] = features[['Sex']].astype('category')
  features = pd.get_dummies(features, columns=["Sex"], prefix=["ohe"] ) 

  return features.values

def evaluate(model,X_train,X_test,y_train,y_test):
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
  #Recall
  recall = tp/(tp+fn)
  #Precision
  precision = tp / (tp+fp)
  #F-Measure
  f1 = 2*precision*recall/(precision+recall)
  print(type(model).__name__)
  print('Recall : {}'.format(recall))
  print('Precision : {}'.format(precision))
  print('F1 : {}'.format(f1))
  
  return recall,precision,f1
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dt = DecisionTreeClassifier()
nb = GaussianNB()
nn = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500,activation='logistic')

X_training = feat_eng(train)
y_training = train[['Survived']].copy().values.reshape(-1,)
X_test = feat_eng(test)
y_test = gender[['Survived']].copy().values.reshape(-1,)

_,_,_ = evaluate(dt,X_training,X_test,y_training,y_test)
_,_,_ = evaluate(nb,X_training,X_test,y_training,y_test)
_,_,_ = evaluate(nn,X_training,X_test,y_training,y_test)
cv  = KFold(n_splits=5)

fold = 1
dt_eval = []
nb_eval = []
nn_eval = []
for train_index,test_index in cv.split(X_training):
  print('*******************************')
  print('FOLD :' + str(fold))
  print('*******************************')
  X_train,y_train = X_training[train_index],y_training[train_index]
  X_val,y_val = X_training[test_index],y_training[test_index]
  y_train = y_train.reshape(len(y_train),)
  y_val = y_val.reshape(len(y_val),)

  clf1 = dt 
  clf2 = nb
  clf3 = nn

  recall_1,precision_1,f1_1 = evaluate(clf1,X_train,X_val,y_train,y_val)
  recall_2,precision_2,f1_2 = evaluate(clf2,X_train,X_val,y_train,y_val)
  recall_3,precision_3,f1_3 = evaluate(clf3,X_train,X_val,y_train,y_val)  

  dt_eval.append([recall_1,precision_1,f1_1])
  nb_eval.append([recall_2,precision_2,f1_2])
  nn_eval.append([recall_3,precision_3,f1_3])
  fold +=1
# at each fold
pd.DataFrame([np.mean(dt_eval,axis=0),np.mean(nb_eval,axis=0),np.mean(nn_eval,axis=0)],columns=['Average Recall','Average Precision','Average F1'],
             index = ['Decision Tree', 'Naive Bayes', 'MLP']
            )
gender_submission = (pd.DataFrame(list(zip(test['PassengerId'],clf3.predict(X_test))),columns=['PassengerId','Survived']))
gender_submission.to_csv('gender_submission.csv',index = False)