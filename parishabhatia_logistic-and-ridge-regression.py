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

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
data.describe()
data.info()
corr= data.corr()

fig = plt.figure()

ax = fig.add_subplot()

cax = ax.matshow(corr, cmap='coolwarm')

fig.colorbar(cax)

ticks = np.arange(0,len(corr.columns),1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

plt.xticks(rotation=90)

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)

plt.show()
#binning the quality column into into parts i.e good quality and bad quality 

#bins = np.linspace(3, 8, 3)

bin_name = ['bad','good']

bins = (2,6.5,8)

data['quality'] = pd.cut(data['quality'],bins=bins,labels=bin_name)
le = LabelEncoder()

le.fit(data['quality'])

data['quality'] = le.fit_transform(data['quality'])
x = data.drop('quality',axis=1).values

y = data['quality'].values.reshape(-1,1)

y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print("x_train shape: ",x_train.shape)

print("x_test shape: ",x_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)
from sklearn import linear_model

logreg = linear_model.LogisticRegression()

logregCV = linear_model.LogisticRegressionCV()

passiveAg = linear_model.PassiveAggressiveClassifier()

percept = linear_model.Perceptron()

ridge = linear_model.RidgeClassifier()

ridgeCV = linear_model.RidgeClassifierCV()

sgd = linear_model.SGDClassifier()
models = [logreg,logregCV, passiveAg, percept, ridge, ridgeCV,sgd]
from sklearn.model_selection import cross_val_score

def get_cv_scores(model):

    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')
for model in models:

    print(model)

    get_cv_scores(model)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

Cs = [1,2,3,4,5]

fit_intercept = [True, False]

cv = [4,5]

dual = [True, False]

penalty = ['l1', 'l2','elasticnet']

solver = ['liblinear', 'saga', 'newton-cg', 'sag', 'lbfgs']

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]





param_grid = dict(penalty=penalty,

                  class_weight=class_weight,

                  solver=solver,

                  fit_intercept=fit_intercept,

                  dual=dual)



grid = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)

grid_result = grid.fit(x_train, y_train)



print('Best Score: ', grid_result.best_score_)

print('Best Params: ', grid_result.best_params_)
logreg = linear_model.LogisticRegression(solver ='lbfgs', class_weight = {1: 0.7, 0: 0.3}, dual= False, fit_intercept= True, penalty ='l2')

get_cv_scores(logregCV)
logreg = linear_model.LogisticRegression()

logreg.fit(x_train,y_train) 

y_pred=logreg.predict(x_test) 
lreg_cm = confusion_matrix(y_pred,y_test)

ax = sns.heatmap(lreg_cm,annot=True)

ax.set(xlabel='predict', ylabel='true')

lreg_as = accuracy_score(y_pred,y_test)

print("logistic regression accuracy score: ",lreg_as*100,'%')
alpha = [1,0.1,10,0.01,100]

fit_intercept = [True, False]

normalize=[True, False]

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

solver = ['auto','svd','lsqr','sag','saga']









param_grid = dict(alpha = alpha ,

                  fit_intercept=fit_intercept,

                  normalize = normalize,

                  class_weight=class_weight,

                  solver=solver)



grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)

grid_result = grid.fit(x_train, y_train)



print('Best Score: ', grid_result.best_score_)

print('Best Params: ', grid_result.best_params_)
ridge = linear_model.RidgeClassifier(alpha=  0.1, class_weight = {1: 0.7, 0: 0.3}, fit_intercept = True, normalize =True, solver ='sag')

get_cv_scores(ridge)
ridge.fit(x_train,y_train) 

y_pred_ridge=ridge.predict(x_test) 

rreg_cm = confusion_matrix(y_pred_ridge,y_test)

ax = sns.heatmap(rreg_cm,annot=True)

ax.set(xlabel='predict', ylabel='true')

rreg_as = accuracy_score(y_pred_ridge,y_test)

print("Ridge regression accuracy score: ",rreg_as*100,'%')
my_submission = pd.DataFrame({'Y_Pred': y_pred_ridge})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)