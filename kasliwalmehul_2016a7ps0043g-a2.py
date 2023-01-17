import pandas as pd

import numpy as np

from math import sqrt

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import LinearRegression,SGDRegressor,LassoLars

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer,accuracy_score

from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier, AdaBoostRegressor,AdaBoostClassifier,ExtraTreesRegressor,RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,VotingClassifier

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer

import matplotlib.pyplot as plt

from sklearn.svm import SVR

from sklearn.utils import resample,class_weight

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
dfx = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df = dfx.iloc[:,1:]

print(df.shape)

print(df['class'].value_counts())

print(df['class'].nunique())

df_w6 = df[df['class']!=6]

print(df_w6['class'].value_counts())

df_6 = df[df['class']==6]

print(df_6['class'].value_counts())
corr = df.iloc[:,:10].corr()

corr.style.background_gradient(cmap='coolwarm')
X = df_w6.iloc[:,:9].values

Y = df_w6.iloc[:,-1].values

print(X.shape,Y.shape)

for x in range(0,2):

    smote = SMOTE('minority')

    X,Y = smote.fit_sample(X,Y)

    print(X.shape,Y.shape)

    print(np.unique(Y,return_counts=True))

X6 = df_6.iloc[:,:9].values

Y6 = df_6.iloc[:,-1].values

X = np.concatenate((X,X6))

Y = np.concatenate((Y,Y6))

print(X.shape,Y.shape)

p = np.random.permutation(len(Y))

Y = np.asarray(Y)

X,Y = X[p], Y[p]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()

#scaler = RobustScaler()

#scaler = Normalizer()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val) 

scorer = make_scorer(accuracy_score)



param_distETC = {

    'max_depth': range(80,90),

    'n_estimators': [100,150,200,250,300,350,400,450,500,550,600],

    'class_weight':['balanced']

}

param_distADB = {

    'n_estimators': [100,150,200,250,300,350,400,450,500,550,600],

    "learning_rate": [0.15, 0.20, 0.25, 0.30 ]

}

param_distXGB = {

    "learning_rate"    : [0.15, 0.20, 0.25, 0.30 ] ,

    "max_depth"        : [ 3, 4, 5, 6, 8],

    "min_child_weight" : [ 1, 3],

    "gamma"            : [ 0.0, 0.1, 0.2],

    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],

    "class_weight" : ['balanced']

}

param_distLGBM = {

    "learning_rate": [0.15, 0.20, 0.25, 0.30 ] ,

    "max_depth": [1,2,3, 4, 5, 6, 8],

    "class_weight": ['balanced']

}
dict = {}

scores_list = []

random_search = GridSearchCV(estimator=XGBClassifier(),param_grid=param_distXGB,cv=3,scoring=scorer, verbose=2)

random_result = random_search.fit(X_train,Y_train)

best_params = random_result.best_params_

print(best_params)

score1 = random_result.best_score_

scores_list.append(score1)

dict[score1] = random_result.best_estimator_

random_search = GridSearchCV(estimator=ExtraTreesClassifier(),param_grid=param_distETC,cv=3,scoring=scorer, verbose=2)

random_result = random_search.fit(X_train,Y_train)

best_params = random_result.best_params_

print(best_params)

score2 = random_result.best_score_

scores_list.append(score2)

dict[score2] = random_result.best_estimator_

random_search = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_distETC,cv=3,scoring=scorer, verbose=2)

random_result = random_search.fit(X_train,Y_train)

best_params = random_result.best_params_

print(best_params)

score3 = random_result.best_score_

scores_list.append(score3)

dict[score3] = random_result.best_estimator_

random_search = GridSearchCV(estimator=LGBMClassifier(),param_grid=param_distLGBM,cv=3,scoring=scorer, verbose=2)

random_result = random_search.fit(X_train,Y_train)

best_params = random_result.best_params_

print(best_params)

score4 = random_result.best_score_

scores_list.append(score4)

dict[score4] = random_result.best_estimator_

scores_list.sort()
ss = pd.read_csv('../input/eval-lab-2-f464/test.csv')

X_test_submit = ss.iloc[:,1:]

X = scaler.fit_transform(X)

X_test_submit = scaler.transform(X_test_submit)

p = np.random.permutation(len(X))

X,Y = X[p], Y[p]

print(X.shape,Y.shape)
estimators = [('m1',dict[scores_list[0]]),('m2',dict[scores_list[1]]),('m3',dict[scores_list[2]])]

parameters = {'weights':[[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X,Y)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_

best_clf_sv.fit(X,Y)

Y_predicted_submit = best_clf_sv.predict(X_test_submit)

out = pd.DataFrame({'id':ss.iloc[:,0],'class':Y_predicted_submit})
estimators = [('m1',dict[scores_list[1]]),('m2',dict[scores_list[2]]),('m3',dict[scores_list[3]])]

parameters = {'weights':[[1,1,1],[1,1,2],[1,2,1],[2,1,1],[1,2,2],[2,1,2],[2,2,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X,Y)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_

best_clf_sv.fit(X,Y)

Y_predicted_submit = best_clf_sv.predict(X_test_submit)

out = pd.DataFrame({'id':ss.iloc[:,0],'class':Y_predicted_submit})
"""feature_importances = pd.DataFrame(best_clf_sv.feature_importances_,index = range(0,9),columns=['importance']).sort_values('importance',ascending=False)

objects = feature_importances.index

y_pos = np.arange(len(objects))

performance = feature_importances['importance']



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.show()

"""