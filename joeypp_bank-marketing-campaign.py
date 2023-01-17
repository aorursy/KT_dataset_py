# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

import scikitplot as skplt

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics



!pip install imblearn

from imblearn.over_sampling import SMOTE





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        inp= os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.



inp_file = pd.read_csv(inp, sep=';')

for i in ['job', 'marital', 'education', 'contact']:

    plt.figure(figsize=(10,4))

    sns.countplot(x=i,hue='y', data=inp_file)

    

corr = inp_file.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr, annot=True)
inp_file = pd.get_dummies(inp_file, columns=['job', 'marital', 'education', 'default', 'housing', 'loan',

       'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True)

labels = inp_file['y'].unique().tolist()

mapping = dict( zip(labels,range(len(labels))) )

inp_file.replace({'y': mapping},inplace=True)



#Train basic logistic regression model.



train, test = train_test_split(inp_file, test_size=0.2, random_state=0, 

                               stratify=inp_file['y'])

train_x=train.drop(columns={'y'})

train_y=train['y']



test_x=test.drop(columns={'y'})

test_y=test['y']



basemodel = LogisticRegression(solver='lbfgs',max_iter=10000)

basemodel.fit(train_x, train_y)



predictions_bm=basemodel.predict(test_x)

score_bm = basemodel.score(test_x, test['y'])

print("Score of base model is:"+str(score_bm))



y_probas = basemodel.predict_proba(test_x)

skplt.metrics.plot_roc(test_y, y_probas)

plt.show()



cm_bm = metrics.confusion_matrix(test_y, predictions_bm, [0,1])

print("Confusion Matrix of base model:")

print(cm_bm)
# Apply grid search cv and run logistic reg



logistic = LogisticRegression(solver='lbfgs')

penalty = ['l2']

max_iter=[10000]

    

# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)

model_gs = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

best_model = model_gs.fit(train_x,train_y)



print('Best C:', best_model.best_estimator_.get_params()['C'])



score_best = best_model.score(test_x, test_y)

print("Best model score:"+str(score_best))



predictions_best=best_model.predict(test_x)

y_probas = best_model.predict_proba(test_x)

skplt.metrics.plot_roc(test_y, y_probas)

plt.show()



cm_best = metrics.confusion_matrix(test_y, predictions_best, [0,1])

print("Confusion matrix of best model:")

print(cm_best)

#Oversampling as true positve rate is low for 'yes' and then fit basic log reg



X_resampled, y_resampled = SMOTE().fit_resample(train_x, train_y)

basemodel = LogisticRegression(solver='lbfgs',max_iter=10000)

basemodel.fit(X_resampled, y_resampled)

predictions_bm=basemodel.predict(test_x)

score_bm = basemodel.score(test_x, test['y'])

print("Score of base model after oversampling is:"+str(score_bm))



y_probas = basemodel.predict_proba(test_x)

skplt.metrics.plot_roc(test_y, y_probas)

plt.show()



cm_bm = metrics.confusion_matrix(test_y, predictions_bm, [0,1])

print("Confusion Matrix of base model after oversampling:")

print(cm_bm)



#Resampling, hyperparameter tuning with grid search cv to pick best model.



logistic = LogisticRegression(solver='lbfgs')

penalty = ['l2']

max_iter=[10000]

    

# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)

model_gs = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

best_model = model_gs.fit(X_resampled,y_resampled)



print('Best C:', best_model.best_estimator_.get_params()['C'])



score_best = best_model.score(test_x, test_y)

print("Best model score:"+str(score_best))



predictions_best=best_model.predict(test_x)

y_probas = best_model.predict_proba(test_x)

skplt.metrics.plot_roc(test_y, y_probas)

plt.show()



cm_best = metrics.confusion_matrix(test_y, predictions_best, [0,1])

print("Confusion matrix of best model:")

print(cm_best)