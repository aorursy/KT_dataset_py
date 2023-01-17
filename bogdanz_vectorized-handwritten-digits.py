# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#ignore warnings for convenience

warnings.filterwarnings("ignore")



import os



print(os.listdir("../input/vectordigits"))



# Any results you write to the current directory are saved as output.
#extracting training data and obtaining the basic info



train_data = pd.read_csv("../input/vectordigits/training.csv")

train_data.info()
y = train_data['label']

train_data = train_data.drop(['label'], axis = 1)
train_data.describe()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

scaled_data = pd.DataFrame(ss.fit_transform(train_data.values), columns=train_data.columns, index=train_data.index)

scaled_data.head()
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pca = PCA(n_components = len(train_data.columns), whiten = True)

pca.fit(train_data)

plt.bar(range(len(train_data.columns)), pca.explained_variance_ratio_)
import seaborn as sns



g = sns.heatmap(train_data.corr(), cmap = "coolwarm")
from sklearn import tree

model_1 = tree.DecisionTreeClassifier(random_state = 1) #decision tree

model_1.fit(train_data, y)

from sklearn.model_selection import cross_val_score

cross_val_score(model_1, train_data, y, scoring = 'accuracy', cv = 5).mean()
from sklearn.model_selection import GridSearchCV

pars = {

    'criterion': ['gini', 'entropy'],

    'max_features': ['auto', 'log2', None],

    'min_samples_split': [2, 4, 5, 10],

    'max_depth': [None, 5, 10, 15]

}



gs = GridSearchCV(estimator = tree.DecisionTreeClassifier(random_state = 1), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(train_data, y)



gs.best_estimator_
gs.best_score_
model_1 = gs.best_estimator_

model_1.fit(train_data, y)

cross_val_score(model_1, train_data, y, scoring = 'accuracy', cv = 5).mean()
model_1_s = gs.best_estimator_

model_1_s.fit(scaled_data, y)

cross_val_score(model_1_s, scaled_data, y, scoring = 'accuracy', cv = 5).mean()



scores = dict({}) #holds scores of different algorithms

scores['Decision tree'] = cross_val_score(model_1_s, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'criterion': ['gini', 'entropy'],

    'max_features': ['auto', 'log2', None],

    'min_samples_split': [2, 4, 5, 10],

    'max_depth': [None, 5, 10, 15]

}



gs = GridSearchCV(estimator = tree.DecisionTreeClassifier(random_state = 1), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
from sklearn.ensemble import RandomForestClassifier

model_2 = RandomForestClassifier(random_state = 1)

model_2.fit(scaled_data, y)

cross_val_score(model_2, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
from matplotlib.pyplot import plot

accuracy = []

for i in range(30):

    a = cross_val_score(RandomForestClassifier(random_state = 1, n_estimators = i+1), 

                           scaled_data, y, scoring = 'accuracy', cv = 5).mean()

    accuracy.append(a)

plot(accuracy)
pars = {

    'n_estimators': [10, 50, 100, 250],

    'criterion': ['gini', 'entropy'],

    'min_samples_split': [2, 4, 5, 10],

    'max_depth': [None, 5, 10, 15]

}



gs = GridSearchCV(estimator = RandomForestClassifier(random_state = 1), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_2 = gs.best_estimator_

scores['Random forest'] = cross_val_score(model_2, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
from sklearn.linear_model import LogisticRegression

model_3 = LogisticRegression(random_state = 1)

model_3.fit(scaled_data, y)

cross_val_score(model_3, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],

    'max_iter': [50, 100, 250],

}



gs = GridSearchCV(estimator = LogisticRegression(random_state = 1, penalty = 'elasticnet', solver = 'saga'), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_3 = gs.best_estimator_

scores['Logistic regression'] = cross_val_score(model_3, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
from sklearn.ensemble import AdaBoostClassifier

model_4 = AdaBoostClassifier(random_state = 1)

model_4.fit(scaled_data, y)

cross_val_score(model_4, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'n_estimators': [100, 250, 500],

    'learning_rate': [0.25, 0.5, 0.75, 1.0, 1.5],

}



gs = GridSearchCV(estimator = AdaBoostClassifier(random_state = 1), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_4 = gs.best_estimator_

scores['AdaBoost'] = cross_val_score(model_4, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
accuracy = []

for i in range(50):

    a = cross_val_score(AdaBoostClassifier(random_state = 1, n_estimators = i+1, learning_rate = 0.5), 

                           scaled_data, y, scoring = 'accuracy', cv = 5).mean()

    accuracy.append(a)

plot(accuracy)
from sklearn.ensemble import GradientBoostingClassifier

model_5 = GradientBoostingClassifier(random_state = 1)

model_5.fit(scaled_data, y)

cross_val_score(model_5, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'n_estimators': [15, 25, 35, 45],

    'learning_rate': [0.05, 0.1, 0.15, 0.2],

    'max_depth':[1,2,3]

}



gs = GridSearchCV(estimator = GradientBoostingClassifier(random_state = 1, warm_start = True), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_5 = gs.best_estimator_

scores['GradientBoost'] = cross_val_score(model_5, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
accuracy = []

for i in range(50):

    a = cross_val_score(GradientBoostingClassifier(random_state = 1, n_estimators = i+1, learning_rate = 0.15, max_depth = 1), 

                           scaled_data, y, scoring = 'accuracy', cv = 5).mean()

    accuracy.append(a)

plot(accuracy)
from xgboost import XGBClassifier

model_6 = XGBClassifier(random_state = 1)

model_6.fit(scaled_data, y)

cross_val_score(model_6, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    #'lambda': [0, 0.5, 1],

    'learning_rate':[0.05, 0.1, 0.15, 0.2, 0.3],

    'max_depth':[6, 8, 10],

    'min_child_weight':[0.25, 0.5, 0.75, 1],

    'n_estimators':[100, 250, 500]

}



#best results - learning_rate = 0.05, max_depth = 6, min_child_weight = 0.75, n_estimators = 250



#gs = GridSearchCV(estimator = XGBClassifier(random_state = 1), param_grid = pars, 

#                  scoring = 'accuracy', cv = 5)



#gs.fit(scaled_data, y)



#gs.best_estimator_
model_6 = XGBClassifier(learning_rate = 0.05, max_depth = 6, min_child_weight = 0.75, n_estimators = 250)

model_6.fit(scaled_data, y)

scores['XGBoost'] = cross_val_score(model_6, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
accuracy = []

for i in range(50):

    a = cross_val_score(XGBClassifier(random_state = 1, n_estimators = i+1, learning_rate = 0.05, max_depth = 6, min_child_weight = 0.75), 

                           scaled_data, y, scoring = 'accuracy', cv = 5).mean()

    accuracy.append(a)

plot(accuracy)
from sklearn.svm import SVC

model_7 = SVC() 

model_7.fit(scaled_data, y)

cross_val_score(model_7, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],

    'C':[0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

}



gs = GridSearchCV(estimator = SVC(), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_7 = gs.best_estimator_

model_7.fit(scaled_data, y)

scores['SVC'] = cross_val_score(model_7, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
from sklearn.naive_bayes import BernoulliNB

model_8 = BernoulliNB(binarize=0.0)

model_8.fit(scaled_data, y)

cross_val_score(model_8, scaled_data, y, scoring = 'accuracy', cv = 5).mean()
pars = {

    'alpha':[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

}



gs = GridSearchCV(estimator = BernoulliNB(), param_grid = pars, 

                  scoring = 'accuracy', cv = 5)



gs.fit(scaled_data, y)



gs.best_estimator_
gs.best_score_
model_8 = gs.best_estimator_

model_8.fit(scaled_data, y)

scores['BernoulliNB'] = cross_val_score(model_8, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)
from sklearn.naive_bayes import GaussianNB

model_9 = GaussianNB()

model_9.fit(scaled_data, y)

scores['GaussianNB'] = cross_val_score(model_9, scaled_data, y, scoring = 'accuracy', cv = 5).mean()

print(scores)