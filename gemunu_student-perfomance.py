import pandas as pd

%matplotlib inline

import seaborn as sns

from matplotlib import pyplot as plt
rawData = pd.read_csv('../input/xAPI-Edu-Data.csv')
rawData.tail().T
rawData.columns


rawData.info()
import numpy as np



np.where(pd.isnull(rawData))
rawData.describe()
cat_Vars = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID','Relation', 'GradeID', 'Topic', 'Semester', 'StudentAbsenceDays']
fig = plt.figure(figsize=(20, 30))

fig.subplots_adjust(hspace=.3, wspace=0.2)



for i in range(1,len(cat_Vars)+1,1):

    ax = fig.add_subplot(5, 2, i,)

    sns.countplot(rawData[cat_Vars[i-1]])

    ax.xaxis.label.set_size(20)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    total = float(len(rawData))

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,height + 5,'{:1.1f}%'.format(100 * height/total),ha="center")
num_Vars = [ 'raisedhands','VisITedResources', 'AnnouncementsView', 'Discussion', 'Class']



# fig = plt.figure(figsize=(10, 10))

# for i in range(1,len(num_Vars)+1,1):

#     ax = fig.add_subplot(2, 2, i)

#     sns.distplot(rawData[num_Vars[i-1]],kde= None)
sns.pairplot(rawData[num_Vars], hue='Class')
target = rawData.pop('Class') # target



# Drop the features not relevant to the student perfomance

rawData.drop(rawData[['ParentAnsweringSurvey', 'ParentschoolSatisfaction']], axis=1, inplace=True)



X = pd.get_dummies(rawData) # get numeric dummy variables for categorical data
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

y = le.fit_transform(target) # encode target labels with a value
from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40) # split data set into train and set

X_train.shape, X_test.shape
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 



param_grid = { 

    'n_estimators': [50, 100, 200],

    'max_features': ['auto', 'sqrt', 'log2'],

    'min_samples_leaf' : [1, 5,10, 50],

    

}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, refit=True)

CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
#CV_rfc.grid_scores_
from sklearn.metrics import accuracy_score



rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True)

rfc.fit(X_train, y_train)

pred = rfc.predict(X_test)

accuracy_score(y_test, pred)
feature_importance = pd.DataFrame(rfc.feature_importances_, index=X_train.columns, columns=["Importance"])

feature_importance.head(8)
feature_importance.sort('Importance', ascending=False, inplace=True)
fig = plt.figure(figsize=(25, 10))

ax = sns.barplot(feature_importance.index, feature_importance['Importance'])

fig.add_axes(ax)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')[1]
import xgboost as xgb



xg_train = xgb.DMatrix(X_train, label=y_train)

xg_test = xgb.DMatrix(X_test, label=y_test)
param = {}



# use softmax multi-class classification

param['objective'] = 'multi:softmax'



# set xgboost parameter values

param['eta'] = 0.1

param['max_depth'] = 9

param['silent'] = 1

param['nthread'] = 3

param['num_class'] = 3



num_round = 100

bst = xgb.train(param, xg_train, num_round)



# get prediction

pred = bst.predict( xg_test ).astype(int)

accuracy_score(y_test, pred)
xgb.plot_importance(bst, height=0.5)