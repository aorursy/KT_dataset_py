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
from sklearn.metrics import  confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
data= pd.read_csv("/kaggle/input/dishonest-internet-users-dataset-data-set/main.csv")
df = pd.DataFrame(data)
df
df.info()
print('ctrust',df.ctrust.unique())
print('cuntrust',df.cuntrust.unique())
print('last',df['last'].unique())
print('context',df.context.unique())
print('score',df.score.unique())
df['score'].replace({'untrustworthy':0,'trustworthy':1},inplace=True)
df
df['context'].replace({'sport':1,'game':2,'ECommerce':3,'holiday':4},inplace=True)
df
y=data.score.values
x_data=data.drop("score",axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=37)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
confu_svm=confusion_matrix(y_test,y_pred)
accuracy_svm=accuracy_score(y_test,y_pred)
precision_svm=precision_score(y_test,y_pred)
recall_svm=recall_score(y_test,y_pred)
f1_svm=f1_score(y_test,y_pred)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
confu_nb=confusion_matrix(y_test,y_pred)
accuracy_nb=accuracy_score(y_test,y_pred)
precision_nb=precision_score(y_test,y_pred)
recall_nb=recall_score(y_test,y_pred)
f1_nb=f1_score(y_test,y_pred)
from tabulate import tabulate
a='SVM'
b='Naive Bayes'
result1=(a,accuracy_svm,precision_svm,recall_svm,f1_svm)
result2=(b,accuracy_nb,precision_nb,recall_nb,f1_nb)
result=(result1,result2)
print('confusion matrix SVM')
print(confu_svm)

print('confusion matrix Naive Bayes')
print(confu_nb)

print(tabulate(result, headers=["accuracy", "precision", "recall","f1_score"]))
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model
def get_models():
    models = dict()
    models['Meta Learner LR'] = LogisticRegression()
    models['Base Learner SVM'] = SVC()
    models['Base Learner Bayes'] = GaussianNB()
    models['Final stacking'] = get_stacking()
    return models
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)
    return scores
models = get_models()
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    
    
    print('>%s accuracy=%.3f ' % (name, mean(scores)))
