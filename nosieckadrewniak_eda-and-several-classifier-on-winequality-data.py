# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scikitplot as skplt

import seaborn as sns



from distutils.version import LooseVersion as Version

from sklearn import __version__ as sklearn_version

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import confusion_matrix



if Version(sklearn_version)<'0.18':

    from sklearn.cross_validation import train_test_split

else:

    from sklearn.model_selection import train_test_split

    

if Version(sklearn_version)<'0.18':

    from sklearn.grid_search import GridSearchCV

else:

    from sklearn.model_selection import GridSearchCV



df=pd.read_csv('/kaggle/input/wine-quality-from-chemical-properties/winequality.csv')
df.head()
df.info()
df['quality'].value_counts()
sns.set(style='whitegrid', context='notebook')

cols1=['residual_sugar','chlorides','free_S_dioxide','total_S_dioxide','density','pH','sulphates','alcohol','quality']

sns.pairplot(df[cols1])

plt.show()
cols=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_S_dioxide','total_S_dioxide','density','pH','sulphates']

cm=np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':7},yticklabels=cols,xticklabels=cols)

plt.show()
sns.violinplot(data=df[cols],palette='Set3',inner='points')

plt.xticks(rotation=90)

plt.show()
X=df[cols]

y=df['quality']



X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)

print(y_train.shape)
clf1=LogisticRegression(penalty='l2',multi_class='auto', random_state=1,max_iter=250)

pipe1=Pipeline([('scl', StandardScaler()),

               ('clf1', clf1)])

    

parameters1={'clf1__C': [0.001, 0.1, 1, 10, 100],

            'clf1__solver': ['newton-cg', 'lbfgs']}

gs1=GridSearchCV(estimator=pipe1, param_grid=parameters1, scoring='accuracy', cv=9, iid=True)

gs1=gs1.fit(X_train, y_train)

y_pred_clf1 = gs1.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf1,normalize=True, figsize=(7,7))

plt.show()

print('linear regresion accuracy score: %.3f' %gs1.best_score_)

print('Best parameters:')

print(gs1.best_params_)
clf2=DecisionTreeClassifier(criterion='entropy', random_state=0)

gs2=GridSearchCV(estimator=clf2, param_grid=[

        {'max_depth':[1,2,3,4,5,6,7,8,None]}], scoring='accuracy',cv=9,iid=True)

gs2=gs2.fit(X_train, y_train)

y_pred_clf2 = gs2.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf2,normalize=True, figsize=(7,7))

plt.show()

print('decision tree classifier accuracy: %.3f' %gs2.best_score_)

print('Best parameters:')

print(gs2.best_params_)
clf3=KNeighborsClassifier(algorithm='auto')

pipe3=Pipeline([('scl', StandardScaler()),

                ('clf3', clf3)])

parameters3={'clf3__n_neighbors':[1,2,3,4,5,6,7,8,9,10],

             'clf3__leaf_size':[1,5,10]}

gs3=GridSearchCV(estimator=pipe3, param_grid=parameters3, scoring='accuracy', cv=9, iid=True)

gs3=gs3.fit(X_train, y_train)

y_pred_clf3 = gs3.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf3,normalize=True, figsize=(7,7))

plt.show()

print('KNeighbors classifier accuracy: %.3f'%gs3.best_score_)

print('Best parameters:')

print(gs3.best_params_)
clf4=RandomForestClassifier(n_estimators=100, criterion='entropy')

clf4.fit(X_train, y_train)

y_pred_clf4 = clf4.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf4,normalize=True, figsize=(7,7))

plt.show()

#print(clf4.feature_importances_)

print('Random forest classifier accuracy: %.3f' %clf4.score(X_test,y_test))

print('Features importances:')

for i, item in enumerate(clf4.feature_importances_): 

    print(cols[i]+': %.3f' %item)
mv_clf=VotingClassifier(estimators=[('dt',gs2), ('kn', gs3), ('rf', clf4)], voting='hard', weights=[1,1,3])

mv_clf.fit(X_train, y_train)

y_pred_mv_clf = mv_clf.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_mv_clf,normalize=True, figsize=(7,7))

plt.show()

print('Voiting classifier accuracy: %.3f' %mv_clf.score(X_test,y_test))
def logistic_regresion_classifier(X_train, y_train, X_test, y_test, cv):

    clf=LogisticRegression(penalty='l2',multi_class='auto', random_state=1,max_iter=250)

    pipe=Pipeline([('scl', StandardScaler()),

               ('clf', clf)])

    

    parameters={'clf__C': [0.001, 0.1, 1, 10, 100],

            'clf__solver': ['newton-cg', 'lbfgs']}

    gs=GridSearchCV(estimator=pipe, param_grid=parameters, scoring='accuracy', cv=cv, iid=True)

    gs=gs.fit(X_train, y_train)

    y_pred_clf = gs.predict(X_test)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf,normalize=True, figsize=(7,7))

    plt.show()

    print('linear regresion accuracy score: %.3f' %gs.best_score_)

    
def decision_tree_classifier(X_train, y_train, X_test, y_test, cv):

    clf=DecisionTreeClassifier(criterion='entropy', random_state=0)

    gs=GridSearchCV(estimator=clf, param_grid=[

        {'max_depth':[1,2,3,4,5,6,7,8,None]}], scoring='accuracy',cv=cv,iid=True)

    gs=gs.fit(X_train, y_train)

    y_pred_clf = gs.predict(X_test)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf,normalize=True, figsize=(7,7))

    plt.show()

    print('decision tree classifier accuracy: %.3f' %gs.best_score_)

    return clf
def KNeighbors_classifier(X_train, y_train, X_test, y_test,cv):

    clf=KNeighborsClassifier(algorithm='auto')

    pipe=Pipeline([('scl', StandardScaler()),

                ('clf', clf)])

    parameters={'clf__n_neighbors':[1,2,3,4,5,6,7,8,9,10],

             'clf__leaf_size':[1,5,10]}

    gs=GridSearchCV(estimator=pipe, param_grid=parameters, scoring='accuracy', cv=cv, iid=True)

    gs=gs.fit(X_train, y_train)

    y_pred_clf = gs.predict(X_test)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf,normalize=True, figsize=(7,7))

    plt.show()

    print('KNeighbors classifier accuracy: %.3f'%gs.best_score_)

    return gs
def Random_forest_classifier(X_train, y_train, X_test, y_test):

    clf=RandomForestClassifier(n_estimators=300, criterion='entropy')

    clf.fit(X_train, y_train)

    print('Random forest classifier accuracy: %.3f' %clf.score(X_test,y_test))

    y_pred_clf = clf.predict(X_test)

    skplt.metrics.plot_confusion_matrix(y_test, y_pred_clf,normalize=True, figsize=(7,7))

    plt.show()

    return clf
cols=['chlorides','total_S_dioxide','sulphates','alcohol']

X1=df[cols]

y1=df['quality']



X_train1, X_test1, y_train1, y_test1=train_test_split(X1,y1,test_size=0.3, random_state=42)
logistic_regresion_classifier(X_train1, y_train1, X_test1, y_test1, 9)
gs21=decision_tree_classifier(X_train1, y_train1, X_test1, y_test1, 9)
gs31=KNeighbors_classifier(X_train1, y_train1, X_test1, y_test1, 9)
clf41=Random_forest_classifier(X_train1, y_train1, X_test1, y_test1)
mv_clf1=VotingClassifier(estimators=[('dt1',gs21), ('kn1', gs31), ('rf1', clf41)], voting='hard')

mv_clf1.fit(X_train1, y_train1)

y_pred_mv_clf1 = mv_clf1.predict(X_test1)

skplt.metrics.plot_confusion_matrix(y_test1, y_pred_mv_clf1,normalize=True, figsize=(7,7))

plt.show()

print('voting classifier accuracy: %.3f' %mv_clf1.score(X_test1,y_test1))
newdf=df.drop(df[df['total_S_dioxide']>200].index)
nX=newdf[cols]

ny=newdf['quality']



nX_train, nX_test, ny_train, ny_test=train_test_split(nX,ny,test_size=0.3, random_state=42)
logistic_regresion_classifier(nX_train, ny_train, nX_test, ny_test, 8)
ngs2=decision_tree_classifier(nX_train, ny_train, nX_test, ny_test, 8)
ngs3=KNeighbors_classifier(nX_train, ny_train, nX_test, ny_test, 8)
nclf4=Random_forest_classifier(nX_train, ny_train, nX_test, ny_test)
nmv_clf=VotingClassifier(estimators=[('ndt',ngs2), ('nkn', ngs3), ('nrf', nclf4)], voting='hard', weights=[1,1,3])

nmv_clf.fit(nX_train, ny_train)

ny_pred_mv_clf = nmv_clf.predict(nX_test)

skplt.metrics.plot_confusion_matrix(ny_test, ny_pred_mv_clf,normalize=True, figsize=(7,7))

plt.show()

print('Voiting classifier accuracy: %.3f' %nmv_clf.score(nX_test,ny_test))
dat=df.drop(df[df['quality']<5].index)

data=dat.drop(dat[dat['quality']>7].index)

data['quality'].value_counts()
newX=data[cols]

newy=data['quality']



newX_train, newX_test, newy_train, newy_test=train_test_split(newX,newy,test_size=0.3, random_state=42)
logistic_regresion_classifier(newX_train, newy_train, newX_test, newy_test, 8)
newgs2=decision_tree_classifier(newX_train, newy_train, newX_test, newy_test, 8)
newgs3=KNeighbors_classifier(newX_train, newy_train, newX_test, newy_test, 8)
newclf4=Random_forest_classifier(newX_train, newy_train, newX_test, newy_test)
newmv_clf=VotingClassifier(estimators=[('newdt',newgs2), ('newkn', newgs3), ('newrf', newclf4)], voting='hard', weights=[1,1,3])

newmv_clf.fit(newX_train, newy_train)

newy_pred_mv_clf = newmv_clf.predict(newX_test)

skplt.metrics.plot_confusion_matrix(newy_test, newy_pred_mv_clf,normalize=True, figsize=(7,7))

plt.show()

print('Voiting classifier accuracy: %.3f' %newmv_clf.score(newX_test,newy_test))