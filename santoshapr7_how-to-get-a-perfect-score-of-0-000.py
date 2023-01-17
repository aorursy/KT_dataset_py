from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'



import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import cross_val_score
import os

os.listdir("../input")

train=pd.read_csv("../input/ghouls-goblins-and-ghosts-boo/train.csv")

test=pd.read_csv("../input/ghouls-goblins-and-ghosts-boo/test.csv")
train.shape 

train.head()





train.drop('id',axis=1,inplace=True)

test1=test.drop('id',axis=1)

train.describe() 
train.isnull().sum()

train.dtypes
train['type'].value_counts()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

train['color']=label.fit_transform(train['color'])

train['type']=label.fit_transform(train['type'])



test1['color']=label.fit_transform(test1['color'])







sns.pairplot(train,hue='type') 
sns.heatmap(train.corr(),annot=True,vmin=-1,vmax=1,cmap='RdYlGn')
train.drop('color',axis=1,inplace=True)

test1.drop('color',axis=1,inplace=True)
X=train.drop('type',axis=1)

y=train['type']
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.tree import DecisionTreeClassifier



from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

from xgboost import XGBClassifier



Classification_models=[('LogisticRegression',LogisticRegression()),('StochasticGDC',SGDClassifier()),('KNC',KNeighborsClassifier()),('SVC',SVC()),

                       ('LinearSVC',LinearSVC()),('GNaiveBayes',GaussianNB()),('MNaiveBayes',MultinomialNB()),('DTree',DecisionTreeClassifier()),

                       ('MLPerceptronC',MLPClassifier()),('RF',RandomForestClassifier()),('ET',ExtraTreesClassifier()),('AdaBoostC',AdaBoostClassifier()),

                       ('GBC',GradientBoostingClassifier()),('XGBC',XGBClassifier())]
result=[]

names=[]

for name,model in Classification_models:

    cvresult=cross_val_score(model,X,y,cv=5,n_jobs=-1,scoring = 'accuracy')

    result.append(cvresult.mean())

    names.append(name)

    print("%s gives %f " % (name, cvresult.mean()))
params={'C':[0.01,0.1,1],'gamma':[1,0.1,0.01],'kernel':['linear', 'poly', 'rbf'] }

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(SVC(),param_grid=params,n_jobs=-1,cv=5)

gridfit=grid.fit(X,y)
gridfit.best_score_

gridfit.best_params_



vc=VotingClassifier(estimators=[('Support Vector Classifier',SVC(C=1, gamma=1, kernel='linear')),

                                ('Gaussian Naive Bayes',GaussianNB())])



vc.fit(X,y)

predictions = vc.predict(test1)
submission = pd.DataFrame({'id':test['id'], 'type':predictions})
submission['type']=label.inverse_transform(submission['type'])
submission.to_csv('submission.csv', index=False)