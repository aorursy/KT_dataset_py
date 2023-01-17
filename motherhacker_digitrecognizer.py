# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print(train.shape)
train.head()
X = train.iloc[:,1:].values
y = train.iloc[:,0].values
test = test.values
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
from sklearn.model_selection import train_test_split
#the optimal ratio for us is 80/20 (Train/Test) 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = seed)
X_shaped = X.reshape(X.shape[0], 28, 28)
plt.imshow(X_shaped[3])
classifier.predict([X[3]])
plt.hist(y, rwidth=0.8)
plt.boxplot(y)
from sklearn.svm import SVC
param_grid = [
  {'C': [1, 10, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly']},
 ]
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(), param_grid, cv=2, n_jobs = 1, verbose = 2)
clf
classifier = SVC(kernel='poly', C=1, gamma=0.001, random_state=seed)
classifier.fit(Xtrain, ytrain)
y_pred = classifier.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_pred)
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=200, random_state=seed)
rfclf.fit(Xtrain, ytrain)
y_pred2 = rfclf.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_pred2)
#respectively, we have good accuracy - 
classifier_final = SVC(kernel='poly', C=1, gamma=0.001, random_state=seed)
classifier_final.fit(X, y)
y_pred_final = classifier.predict(test)

df = pd.DataFrame(y_pred_final)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)