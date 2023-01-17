import numpy as np

import scipy as sp

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import time



from sklearn.dummy import DummyClassifier as DC

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression as LR

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.svm import SVC, LinearSVC as LSVC

from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC

from sklearn.ensemble import GradientBoostingClassifier as GBC, BaggingClassifier as BC, ExtraTreesClassifier as ETC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv')

todrop = list()

train.isnull().sum()
print('Age')

print('Fraction of Values Missing: ', train['Age'].isnull().sum()/train.shape[0])

train['Age'].hist()

plt.show()
#replacing with mean age

train['Age'] = train['Age'].fillna(np.mean(train['Age']))
#discard cabin - most values missing

train = train.drop('Cabin',axis=1)

todrop.append('Cabin')



print('Embarked')

print('Fraction of Values Missing: ', train['Embarked'].isnull().sum()/train.shape[0])

train['Embarked'].value_counts().plot(kind='bar')

plt.show()
#replacing with most frequent value 

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])



train['Survived'].value_counts().plot(kind='bar')

plt.title('Class Distribution of Survived')

plt.show()



#dropping Name and Ticket

todrop.append('Name')

todrop.append('Ticket')

train = train.drop(['Name','Ticket'],axis=1)
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train,ci=None)

plt.show()

sns.barplot(x='SibSp',y='Survived',hue='Sex',data=train,ci=None)

plt.show()

sns.barplot(x='Parch',y='Survived',hue='Sex',data=train,ci=None)

plt.show()

sns.distplot(train.Fare[train['Survived'] == 1])

sns.distplot(train.Fare[train['Survived'] == 0])

plt.legend({'Survived':1,'Not Survived':2})

plt.show()
def cleanup(df):

    df = df.drop(todrop,axis=1)

    df['Age'] = df['Age'].fillna(np.mean(df['Age']))

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])

    df['Fare'] = np.log1p(df['Fare'])

    return df
X = train.drop(['PassengerId','Survived'],axis=1)

y = train['Survived']

X['Fare'] = np.log1p(X['Fare']) #fare is highly skewed



le_sex = LabelEncoder()

le_sex.fit(X['Sex'])

X['Sex'] = le_sex.transform(X['Sex'])



le_embarked = LabelEncoder()

le_embarked.fit(X['Embarked'])

X['Embarked'] = le_embarked.transform(X['Embarked'])



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=23)



toscale=['Age','Fare']

ss = StandardScaler()

ss.fit(X_train[toscale])

X_train[toscale]=ss.transform(X_train[toscale])

X_test[toscale]=ss.transform(X_test[toscale])
#optimize KNN neighbors

t0=time.time()

param_grid = {'n_neighbors':np.linspace(2,30,num=29,dtype=int)}

opt_knn = GridSearchCV(KNN(),param_grid=param_grid,cv=5)

opt_knn.fit(X_train,y_train)

sc = opt_knn.score(X_test,y_test)

n_neighbors = opt_knn.best_params_.get('n_neighbors')

print('Optimum number of neighbors: ', n_neighbors)

print('Test Score: %0.3f'%sc)

print('Train Score: %0.3f'%opt_knn.score(X_train,y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
#optimizing SVC - rbf kernel

t0=time.time()

param_grid = {'C':np.logspace(-3,3,num=7),'gamma':np.logspace(-5,-1,num=5)}

opt_svc = GridSearchCV(SVC(kernel='rbf'),param_grid=param_grid)

opt_svc.fit(X_train,y_train)

C_opt_svc = opt_svc.best_params_.get('C')

gamma_opt = opt_svc.best_params_.get('gamma')

print('Optimum C: ',C_opt_svc)

print('Optimum gamma: ',gamma_opt)

print('Test Score: %0.3f'%opt_svc.score(X_test, y_test))

print('Train Score: %0.3f'%opt_svc.score(X_train, y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
#optimizing LinearSVC - rbf kernel

t0=time.time()

param_grid = {'C':np.logspace(-3,3,num=7)}

opt_lsvc = GridSearchCV(LSVC(),param_grid=param_grid)

opt_lsvc.fit(X_train,y_train)

C_opt_lsvc = opt_lsvc.best_params_.get('C')

print('Optimum C: ',C_opt_lsvc)

print('Test Score: %0.3f'%opt_svc.score(X_test, y_test))

print('Train Score: %0.3f'%opt_svc.score(X_train, y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
t0=time.time()

classifiers = [DC(),KNN(n_neighbors=n_neighbors),SVC(kernel='rbf',C=C_opt_svc,gamma=gamma_opt),

              LSVC(C=C_opt_lsvc),LR(),DTC(),RFC(),ABC(),GBC(),LDA(),QDA(),BC(),ETC()]

columns=['Classifier','Test Score','Train Score']

scores = pd.DataFrame(columns=columns)

for clf in classifiers:

    clf.fit(X_train,y_train)

    tst_sc = clf.score(X_test,y_test)

    trn_sc = clf.score(X_train,y_train)

    name = clf.__class__.__name__

    print()

    print('Classifier: ', name)

    print('Test Score: %0.3f'%tst_sc)

    print('Train Score: %0.3f'%trn_sc)

    temp = pd.DataFrame([[name,tst_sc,trn_sc]],columns=columns)

    scores = scores.append(temp)



sns.barplot(x='Test Score',y='Classifier',data=scores)

plt.xlabel('Test Score')

plt.xlim(0,1)

plt.show()



sns.barplot(x='Train Score',y='Classifier',data=scores)

plt.xlabel('Train Score')

plt.xlim(0,1)

plt.show()

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
t0=time.time()

#BaggingClassifier Tuning

param_grid = {'n_estimators':np.linspace(10,20,num=11,dtype=int),

             'max_samples':np.linspace(1,10,num=10,dtype=int),

             'max_features':np.linspace(1,X_train.shape[1],num=10,dtype=int)}

opt_bc = GridSearchCV(BC(),param_grid=param_grid,cv=5)

opt_bc.fit(X_train,y_train)



opt_n_estimators_bc = opt_bc.best_params_.get('n_estimators')

opt_max_samples_bc = opt_bc.best_params_.get('max_samples')

opt_max_features_bc = opt_bc.best_params_.get('max_features')



print('Optimum n_estimators: ',opt_n_estimators_bc)

print('Optimum max_samples: ', opt_max_samples_bc)

print('Optimum max_features: ', opt_max_features_bc)

print('Optimum Test Score: ', opt_bc.score(X_test, y_test))

print('Optimum Train Score: ', opt_bc.score(X_train,y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
t0=time.time()

#DecisionTree Tuning

param_grid = {'max_depth':np.linspace(5,15,num=11,dtype=int),

             'min_samples_split':np.linspace(2,10,num=9,dtype=int),

             'min_samples_leaf':np.linspace(2,20,num=19,dtype=int)}

opt_dtc = GridSearchCV(DTC(),param_grid = param_grid)

opt_dtc.fit(X_train,y_train)



opt_max_depth_dtc = opt_dtc.best_params_.get('max_depth')

opt_min_samples_split_dtc = opt_dtc.best_params_.get('min_samples_split')

opt_min_samples_leaf_dtc = opt_dtc.best_params_.get('min_samples_leaf')



print('Optimum max_depth: ',opt_max_depth_dtc)

print('Optimum min_samples_split: ', opt_min_samples_split_dtc)

print('Optimum min_samples_leaf: ', opt_min_samples_leaf_dtc)

print('Optimum Test Score: ', opt_dtc.score(X_test, y_test))

print('Optimum Train Score: ', opt_dtc.score(X_train,y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
t0=time.time()

#RandomForest tuning

t0=time.time()

param_grid = {'max_depth':np.linspace(3,10,num=8,dtype=int),

             'min_samples_split':np.linspace(4,12,num=9,dtype=int),

             'min_samples_leaf':np.linspace(4,12,num=9,dtype=int)}



opt_rfc = GridSearchCV(RFC(n_estimators=20),param_grid = param_grid,cv=5)

opt_rfc.fit(X_train,y_train)



opt_max_depth_rfc = opt_rfc.best_params_.get('max_depth')

opt_min_samples_split_rfc = opt_rfc.best_params_.get('min_samples_split')

opt_min_samples_leaf_rfc = opt_rfc.best_params_.get('min_samples_leaf')



print('Optimum max_depth: ',opt_max_depth_rfc)

print('Optimum min_samples_split: ', opt_min_samples_split_rfc)

print('Optimum min_samples_leaf: ', opt_min_samples_leaf_rfc)

print('Optimum Test Score: ', opt_rfc.score(X_test, y_test))

print('Optimum Train Score: ', opt_rfc.score(X_train,y_train))

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
t0=time.time()

classifiers = [DC(),KNN(n_neighbors=n_neighbors),SVC(kernel='rbf',C=C_opt_svc,gamma=gamma_opt),

              LSVC(C=C_opt_lsvc),LR(),

               DTC(max_depth=opt_max_depth_dtc,min_samples_split=opt_min_samples_split_dtc,

                  min_samples_leaf=opt_min_samples_leaf_dtc),

               RFC(n_estimators=20,max_depth=opt_max_depth_rfc,min_samples_split=opt_min_samples_split_rfc,

                  min_samples_leaf=opt_min_samples_leaf_rfc),

               ABC(),GBC(),LDA(),QDA(),

               BC(n_estimators=opt_n_estimators_bc,max_samples=opt_max_samples_bc,max_features=opt_max_features_bc),

               ETC()]

columns=['Classifier','Test Score','Train Score']

scores = pd.DataFrame(columns=columns)

for clf in classifiers:

    clf.fit(X_train,y_train)

    tst_sc = clf.score(X_test,y_test)

    trn_sc = clf.score(X_train,y_train)

    name = clf.__class__.__name__

    print()

    print('Classifier: ', name)

    print('Test Score: ', tst_sc)

    print('Train Score: ', trn_sc)

    temp = pd.DataFrame([[name,tst_sc,trn_sc]],columns=columns)

    scores = scores.append(temp)



sns.barplot(x='Test Score',y='Classifier',data=scores)

plt.xlabel('Test Score')

plt.xlim(0,1)

plt.show()



sns.barplot(x='Train Score',y='Classifier',data=scores)

plt.xlabel('Train Score')

plt.xlim(0,1)

plt.show()

t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
scores.nlargest(5,columns='Test Score')
t0=time.time()

#Model with the best 5 estimators. Prediction determined by majority voting

model = [GBC(),SVC(kernel='rbf',C=C_opt_svc,gamma=gamma_opt),

        DTC(max_depth=opt_max_depth_dtc,min_samples_split=opt_min_samples_split_dtc,

                  min_samples_leaf=opt_min_samples_leaf_dtc),

        RFC(n_estimators=20,max_depth=opt_max_depth_rfc,min_samples_split=opt_min_samples_split_rfc,

                  min_samples_leaf=opt_min_samples_leaf_rfc), ABC()]

model_clf = list()



y_all_model_train = np.zeros((y_train.shape[0],5))

y_all_model_test = np.zeros((y_test.shape[0],5))

for i,clf in enumerate(model):

    clf.fit(X_train,y_train)

    model_clf.append(clf)

    y_all_model_train[:,i] = clf.predict(X_train)

    y_all_model_test[:,i] = clf.predict(X_test)



y_model_train = np.zeros((y_train.shape[0]))

y_model_test = np.zeros((y_test.shape[0]))



y_model_train[:] = np.round(np.sum(y_all_model_train,axis=1)/5)

y_model_test[:] = np.round(np.sum(y_all_model_test,axis=1)/5)



tst_sc = accuracy_score(y_test,y_model_test)

trn_sc = accuracy_score(y_train,y_model_train)

print('Test Score: %0.3f'%tst_sc)

print('Train Score: %0.3f'%trn_sc)

temp = pd.DataFrame([['5 Best Voting',tst_sc,trn_sc]],columns=columns)

scores = scores.append(temp)



sns.barplot(x='Test Score',y='Classifier',data=scores)

plt.xlabel('Test Score')

plt.xlim(0,1)

plt.show()



sns.barplot(x='Train Score',y='Classifier',data=scores)

plt.xlabel('Train Score')

plt.xlim(0,1)

plt.show()



t1=time.time()

print('Time taken: %.2f'%(t1-t0),' seconds.')
scores.nlargest(3,columns='Test Score')
#submit predictions from these 3 models

gbc = GBC()

gbc.fit(X_train,y_train)

print('Gradient Boosting Classifier')

print('Test Score: ', gbc.score(X_test,y_test))

print('Train Score: ',gbc.score(X_train,y_train))



print()

svc = SVC(kernel='rbf',C=C_opt_svc,gamma=gamma_opt)

svc.fit(X_train,y_train)

print('SVC')

print('Test Score: ', svc.score(X_test,y_test))

print('Train Score: ',svc.score(X_train,y_train))
test = pd.read_csv('../input/test.csv')

test_X = cleanup(test)

test_X['Fare'] = test_X['Fare'].fillna(np.mean(test_X['Fare']))

test_X = test_X.drop('PassengerId',axis=1)

test_X['Sex'] = le_sex.transform(test_X['Sex'])

test_X['Embarked'] = le_embarked.transform(test_X['Embarked'])

test_X[toscale] = ss.transform(test_X[toscale])



#gradient boosting submission

y_gradient = gbc.predict(test_X)

gbc_submission = pd.DataFrame()

gbc_submission['PassengerId'] = test['PassengerId']

gbc_submission['Survived'] = y_gradient

gbc_submission.to_csv('gbc_submission.csv',index=False)



#SVC boosting submission

y_svc = svc.predict(test_X)

svc_submission = pd.DataFrame()

svc_submission['PassengerId'] = test['PassengerId']

svc_submission['Survived'] = y_svc

svc_submission.to_csv('svc_submission.csv',index=False)



#5-best submission

y_5best_all = np.zeros((test_X.shape[0],5)) 

for i,clf in enumerate(model_clf):

    y_5best_all[:,i] = clf.predict(test_X)



y_5best = np.zeros((test_X.shape[0]))

y_5best[:] = np.round(np.sum(y_5best_all,axis=1)/5)



fivebest_submission = pd.DataFrame()

fivebest_submission['PassengerId'] = test['PassengerId']

fivebest_submission['Survived'] = y_5best.astype(int)

fivebest_submission.to_csv('fivebest_submission.csv',index=False)