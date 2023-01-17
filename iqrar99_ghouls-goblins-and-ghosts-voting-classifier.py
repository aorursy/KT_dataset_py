# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder



import os

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dtrain = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

dtest = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
dtrain.head()
dtrain.info()
dtrain.shape
dtrain.describe()
f,ax = plt.subplots(1,4, figsize=(15,3))



sns.distplot(dtrain['bone_length'], ax = ax[0])

sns.distplot(dtrain['rotting_flesh'], ax = ax[1], color = 'r')

sns.distplot(dtrain['hair_length'], ax = ax[2], color = 'g')

sns.distplot(dtrain['has_soul'], ax = ax[3], color = 'purple')



plt.show()
f,ax = plt.subplots(1,4, figsize=(15,3))



sns.distplot(np.log(dtrain['bone_length']), ax = ax[0])

sns.distplot(np.log(dtrain['rotting_flesh']), ax = ax[1], color = 'r')

sns.distplot(np.log(dtrain['hair_length']), ax = ax[2], color = 'g')

sns.distplot(np.log(dtrain['has_soul']), ax = ax[3], color = 'purple')



plt.show()
sns.pairplot(data=dtrain.iloc[:,1:], hue = 'type')
sns.heatmap(dtrain.drop('id', axis = 1).corr(), square = True, annot = True)
dtrain['type'].unique()
label_encoder = LabelEncoder()

dtrain['type'] = label_encoder.fit_transform(dtrain['type'])

# 0 : Ghost

# 1 : Ghoul

# 2 : Goblin
dtrain['type'].unique()
dtrain['color'].unique()
d_color = dtrain['color']
dtrain['clear'] = [1 if d_color[i] == 'clear' else 0 for i in range(len(d_color))]

dtrain['green'] = [1 if d_color[i] == 'green' else 0 for i in range(len(d_color))]

dtrain['black'] = [1 if d_color[i] == 'black' else 0 for i in range(len(d_color))]

dtrain['white'] = [1 if d_color[i] == 'white' else 0 for i in range(len(d_color))]

dtrain['blue'] = [1 if d_color[i] == 'blue' else 0 for i in range(len(d_color))]

dtrain['blood'] = [1 if d_color[i] == 'blood' else 0 for i in range(len(d_color))]
d_color2 = dtest['color']
dtest['clear'] = [1 if d_color2[i] == 'clear' else 0 for i in range(len(d_color2))]

dtest['green'] = [1 if d_color2[i] == 'green' else 0 for i in range(len(d_color2))]

dtest['black'] = [1 if d_color2[i] == 'black' else 0 for i in range(len(d_color2))]

dtest['white'] = [1 if d_color2[i] == 'white' else 0 for i in range(len(d_color2))]

dtest['blue'] = [1 if d_color2[i] == 'blue' else 0 for i in range(len(d_color2))]

dtest['blood'] = [1 if d_color2[i] == 'blood' else 0 for i in range(len(d_color2))]
dtrain.head()
dtest.head()
train = dtrain.copy()

y = train['type']

x = train.drop(['id', 'color','type'], axis = 1)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 0) 

fold = KFold(n_splits = 5)
scoring = {'accuracy' : make_scorer(accuracy_score), 

           'precision' : make_scorer(precision_score),

           'recall' : make_scorer(recall_score), 

           'f1_score' : make_scorer(f1_score)}
ensembles=[]

ensembles.append(('rfc',RandomForestClassifier(n_estimators=10)))

ensembles.append(('abc',AdaBoostClassifier(n_estimators=10)))

ensembles.append(('bc',BaggingClassifier(n_estimators=10)))

ensembles.append(('etc',ExtraTreesClassifier(n_estimators=10)))



results=[]

names=[]

for name,model in ensembles:

    result = cross_val_score(model,x_train,y_train,cv=fold,scoring='accuracy')

    results.append(result)

    names.append(name)

    msg="%s : %f (%f)"%(name,result.mean(),result.std())

    print(msg)
# Random Forest Tuning

n_estimators=[10,20,30,40,50]

max_depth =  [4,6,8,10,12,24]



param_grid=dict(n_estimators=n_estimators, max_depth=max_depth)



model=RandomForestClassifier()



fold=KFold(n_splits=10,random_state=0)



grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)

grid_result=grid.fit(x_train,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
rf_best_params = grid_result.best_params_
# AdaBoost Tuning

n_estimators=[10,20,30,40,50]

learning_rate =  [1.0, 0.1, 0.05, 0.01, 0.001]

param_grid=dict(n_estimators=n_estimators, learning_rate=learning_rate)





model=AdaBoostClassifier()



grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)

grid_result=grid.fit(x_train,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
ab_best_params = grid_result.best_params_
# Bagging Tuning

n_estimators=[10,20,30,40,50]

max_features =  [2,4,6,8,10]



param_grid=dict(n_estimators=n_estimators, max_features=max_features)



model=BaggingClassifier()



grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)

grid_result=grid.fit(x_train,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
bc_best_params = grid_result.best_params_
# Extra Trees Tuning

n_estimators=[10,20,30,40,50]

max_depth =  [4,6,8,10,12,24]



param_grid=dict(n_estimators=n_estimators, max_depth=max_depth)



model=ExtraTreesClassifier()



grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)

grid_result=grid.fit(x_train,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
et_best_params = grid_result.best_params_
rf = RandomForestClassifier(**rf_best_params)

et = ExtraTreesClassifier(**et_best_params)

bc = BaggingClassifier(**bc_best_params)

ab = AdaBoostClassifier(rf, **ab_best_params)



vc = VotingClassifier(estimators=[('et', et), ('rf', rf), ('bc', bc), ('ab', ab)], voting='soft')

vc.fit(x_train, y_train)

y_pred = vc.predict(x_test)

vc_acc = accuracy_score(y_pred, y_test)

print("Accuracy score: ", vc_acc)
# It's time to predict the data

y_pred = vc.predict(dtest.drop(['id','color'], axis = 1))

y_pred
sub = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip')

sub['type'] = y_pred

sub['type'] = sub['type'].map({

    0:'Ghost',

    1:'Ghoul',

    2:'Goblin'

})
sub
# submit

sub.to_csv('submission.csv', index = False)