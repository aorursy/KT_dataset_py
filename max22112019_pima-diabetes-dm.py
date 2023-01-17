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

import matplotlib.pyplot as plt

import numpy as np
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
import seaborn as sns

sns.countplot(x='Outcome', data=df)

plt.show()
diabetes_count = len(df.loc[df['Outcome'] == 1])

no_diabetes_count=len(df.loc[df['Outcome']==0])

(diabetes_count, no_diabetes_count)
#distribution of various parameters in the dataset except the target variable

cols=['Pregnancies','Glucose','BloodPressure','SkinThickness',

      'Insulin','BMI','DiabetesPedigreeFunction','Age']

num=df[cols]

for i in num.columns:

    plt.hist(num[i])

    plt.title(i)

    plt.show()
print("total number of rows : {0}".format(len(df)))

print("number of rows with 0 Pregnancies: {0}".format(len(df.loc[df['Pregnancies'] == 0])))

print("number of rows with 0 Glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))

print("number of rows with 0 BloodPressure: {0}".format(len(df.loc[df['BloodPressure'] == 0])))

print("number of rows with 0 SkinThickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))

print("number of rows with 0 Insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))

print("number of rows with 0 BMI: {0}".format(len(df.loc[df['BMI'] == 0])))

print("number of rows with 0 DiabetesPedigreeFunction: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))

print("number of rows with 0 Ages: {0}".format(len(df.loc[df['Age'] == 0])))
from sklearn.impute import SimpleImputer

zcol=['Glucose','BloodPressure','SkinThickness',

      'Insulin','BMI']

zcols=df[zcol]

imputer = SimpleImputer(missing_values=0, strategy="mean", verbose=0)

imputed_df = pd.DataFrame(imputer.fit_transform(zcols))

imputed_df.columns = zcols.columns

temp=imputed_df.copy()

zcols=temp.copy()
zcols.head()
df.drop(['Glucose','BloodPressure','SkinThickness',

      'Insulin','BMI'], axis=1, inplace=True)
df=df.join(zcols)

df.head()
df.dtypes
outcome=df['Outcome']

df.drop(['Outcome'], axis=1, inplace=True)

df=df.join(outcome)

df.head()
X=df.iloc[:,:-1]

y=df.iloc[:,-1]
X.head()
y.head()
# stratify the outcome

from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)

#stratify=y
X_train.head()
X_test.head()
from sklearn.model_selection import cross_val_score

from sklearn import metrics
#Scaling the training and test dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=8)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
'''lr = LogisticRegression(max_iter = 2000, random_state=0)

cv = cross_val_score(lr,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

lr.fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(y_pred_lr,y_test)*100)



[0.76388889 0.73611111 0.69444444 0.77777778 0.81944444 0.73611111

 0.77777778 0.83333333]

76.73611111111111

The accuracy of the Logistic Regression is 75.0'''
'''#Hyperparameter Tuning 

lr = LogisticRegression(random_state=0)

param_grid = {'max_iter' : [2000,4000],

              'penalty' : ['l1', 'l2','elasticnet'],

              'C' : np.logspace(-4, 4, 50),

              'solver' : ['newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag','saga']

                }



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)

best_clf_lr = clf_lr.fit(X_train,y_train)

best_clf_lr.best_estimator_'''
lr = LogisticRegression(C=0.8286427728546842, max_iter=2000, solver='saga', random_state=0)

cv = cross_val_score(lr,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

lr.fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(y_pred_lr,y_test)*100)

cm=confusion_matrix(y_test, y_pred_lr)

print(cm)

classification_report(y_test, y_pred_lr)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

cv = cross_val_score(gnb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

gnb.fit(X_train,y_train)

y_pred_gnb=gnb.predict(X_test)

print('The accuracy of the Naive Bayes is',metrics.accuracy_score(y_pred_gnb,y_test)*100)

cm=confusion_matrix(y_test, y_pred_gnb)

print(cm)

classification_report(y_test, y_pred_gnb)
'''from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

cv = cross_val_score(rf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_rf = rf.predict(X_test)

print('The accuracy of the RandomForestClassifier is',metrics.accuracy_score(y_pred_rf,y_test)*100)



[0.68055556 0.80555556 0.77777778 0.76388889 0.79166667 0.70833333

 0.76388889 0.70833333]

75.0

The accuracy of the RandomForestClassifier is 77.08333333333334'''
'''#Hyperparameter Tuning

rf = RandomForestClassifier(random_state = 0)

param_grid =  {'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800], 

                                  'bootstrap': [True,False],

                                  'max_depth': [3,4,5,6,7,8,9,10,15,20,50,None],

                                  'max_features': [3,'auto','sqrt','log2'],

                                  'bootstrap': [False, True],

                                  'criterion': ['gini', 'entropy'],

                                  'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],

                                  'min_samples_split': [2 ,3,4,5,6,7,8,9,10]}

                                  

clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 300, 

cv = kfold, verbose = True, n_jobs = -1)

best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)

best_clf_rf_rnd.best_estimator_'''
'''param_grid =  {'n_estimators': [500,600,700,800], 

                                  'bootstrap': [True,False],

                                  'max_depth': [40,50,60,70],

                                  'max_features': ['log2'],

                                  'bootstrap': [True],

                                  'criterion': ['entropy'],

                                  'min_samples_leaf': [3,4,5,6],

                                  'min_samples_split': [7,8,9,10]}

clf_rf_gr = GridSearchCV(rf, param_grid = param_grid, 

cv = kfold, verbose = True, n_jobs = -1)

best_clf_rf_gr = clf_rf_gr.fit(X_train,y_train)

best_clf_rf_gr.best_estimator_



RandomForestClassifier(criterion='entropy', max_depth=40, max_features='log2',

                       min_samples_leaf=4, min_samples_split=9,

                       n_estimators=500, random_state=0)'''
'''param_grid =  {'n_estimators': [400,500,600,700,800], 

                                  'bootstrap': [True,False],

                                  'max_depth': [40,50,60,70],

                                  'max_features': ['log2'],

                                  'bootstrap': [True],

                                  'criterion': ['entropy'],

                                  'min_samples_leaf': [3,4,5,6],

                                  'min_samples_split': [7,8,9,10]}

clf_rf_gr = GridSearchCV(rf, param_grid = param_grid, 

cv = kfold, verbose = True, n_jobs = -1)

best_clf_rf_gr = clf_rf_gr.fit(X_train,y_train)

best_clf_rf_gr.best_estimator_'''
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', max_depth=40, max_features='log2',

                       min_samples_leaf=4, min_samples_split=9,

                       n_estimators=500, random_state=0)

rf.fit(X_train, y_train)

cv = cross_val_score(rf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_rf = rf.predict(X_test)

print('The accuracy of the RandomForestClassifier is',metrics.accuracy_score(y_pred_rf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_rf)

print(cm)

classification_report(y_test, y_pred_rf)
'''from sklearn.svm import SVC

svc = SVC(random_state = 0, probability=True)

param_grid =  {'kernel' :['linear', 'rbf’, ‘poly'],

               'gamma' :[0.1,0.5, 1,5, 10,20,50,70, 100],

               'C' :[0.1, 1, 10, 100, 1000],

               'degree' :[0, 1, 2, 3, 4, 5, 6]}

clf_svc_rnd = RandomizedSearchCV(svc, param_distributions = param_grid, n_iter = 200, 

cv = kfold, verbose = True, n_jobs = -1)

best_clf_svc_rnd = clf_svc_rnd.fit(X_train,y_train)

best_clf_svc_rnd.best_estimator_'''            
'''from sklearn.svm import SVC

svc = SVC(C=100, gamma=20, kernel='linear', probability=True, random_state=0)

svc.fit(X_train, y_train)

cv = cross_val_score(svc,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_svc = svc.predict(X_test)

print('The accuracy of the Linear SVC is',metrics.accuracy_score(y_pred_svc,y_test)*100)'''
#Linear SVC

from sklearn.svm import SVC

svcl = SVC(kernel = 'linear', random_state = 0, probability=True)

svcl.fit(X_train, y_train)

cv = cross_val_score(svcl,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_svcl = svcl.predict(X_test)

print('The accuracy of the Linear SVC is',metrics.accuracy_score(y_pred_svcl,y_test)*100)

cm=confusion_matrix(y_test, y_pred_svcl)

print(cm)

classification_report(y_test, y_pred_svcl)
#rbf SVC

from sklearn.svm import SVC

svck = SVC(kernel = 'rbf', random_state = 0, probability=True)

svck.fit(X_train, y_train)

cv = cross_val_score(svck,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_svck = svck.predict(X_test)

print('The accuracy of the Kernel SVC is',metrics.accuracy_score(y_pred_svck,y_test)*100)

cm=confusion_matrix(y_test, y_pred_svck)

print(cm)

classification_report(y_test, y_pred_svck)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 0)

dt.fit(X_train, y_train)

cv = cross_val_score(dt,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_dt = dt.predict(X_test)

print('The accuracy of the Decision Tree Classifier is',metrics.accuracy_score(y_pred_dt,y_test)*100)

cm=confusion_matrix(y_test, y_pred_dt)

print(cm)

classification_report(y_test, y_pred_dt)
'''from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

cv = cross_val_score(knn,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_knn = knn.predict(X_test)

print('The accuracy of the K-Neighbors Classifier is',metrics.accuracy_score(y_pred_knn,y_test)*100)'''
'''from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : [3,5,7,9,11,13,15,16,17,19],

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2,3,4,5,6,7,8,9,10]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)

best_clf_knn = clf_knn.fit(X_train,y_train)

best_clf_knn.best_estimator_'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15, weights='distance')

knn.fit(X_train, y_train)

cv = cross_val_score(knn,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_knn = knn.predict(X_test)

print('The accuracy of the K-Neighbors Classifier is',metrics.accuracy_score(y_pred_knn,y_test)*100)

cm=confusion_matrix(y_test, y_pred_knn)

print(cm)

classification_report(y_test, y_pred_knn)
from xgboost import XGBClassifier
'''from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =0)

xgb.fit(X_train, y_train)

cv = cross_val_score(xgb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_xgb = xgb.predict(X_test)

print('The accuracy of the XGB Classifier is',metrics.accuracy_score(y_pred_xgb,y_test)*100)



[0.73611111 0.79166667 0.70833333 0.73611111 0.81944444 0.68055556

 0.79166667 0.72222222]

74.82638888888889

The accuracy of the XGB Classifier is 75.52083333333334'''
'''xgb = XGBClassifier(random_state = 0)



param_grid = {

    'n_estimators': [20, 50, 100, 250,300,400, 500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],

    'colsample_bytree': [0.2,0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1],

    'max_depth': [2, 5, 8,10, 15, 20, 25, None],

    'reg_alpha': [0, 0.5, 1],

    'reg_lambda': [1, 1.5, 2,2.5,3,4],

    'subsample': [0.2,0.3,0.4,0.5,0.6, 0.7, 0.8,0.9],

    'learning_rate':[.01,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.9],

    'gamma':[0,.01,.1,.5,1,10,20,30,40,50,70,100],

    'min_child_weight':[0,.01,0.05,0.1,1,10,100],

    'sampling_method': ['uniform', 'gradient_based']

}



clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 200, 

cv = kfold, verbose = True, n_jobs = -1)

best_clf_xgb_rnd = clf_xgb_rnd.fit(X_train,y_train)'''
#best_clf_xgb_rnd.best_estimator_
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.01, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.01, max_delta_step=0, max_depth=10,

              min_child_weight=10, monotone_constraints='()',

              n_estimators=1200, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0.5, reg_lambda=2, sampling_method='uniform',

              scale_pos_weight=1, subsample=0.7, tree_method='exact',

              validate_parameters=1, verbosity=None)

xgb.fit(X_train, y_train)

cv = cross_val_score(xgb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_xgb = xgb.predict(X_test)

print('The accuracy of the XGB Classifier is',metrics.accuracy_score(y_pred_xgb,y_test)*100)

cm=confusion_matrix(y_test, y_pred_xgb)

print(cm)

classification_report(y_test, y_pred_xgb)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr', lr),('gnb',gnb),('knn',knn),

                                            ('rf',rf),('svck',svck),('svcl',svcl),

                                            ('xgb',xgb)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('rf',rf),('svck',svck),('knn',knn),

                                            ('xgb',xgb)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('svck',svck),

                                            ('rf',rf),

                                            ('xgb',xgb)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr', lr),('gnb',gnb),('knn',knn),

                                            ('rf',rf),('svck',svck),('svcl',svcl),('dt',dt),

                                            ('xgb',xgb)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('rf',rf),('svck',svck),('knn',knn),

                                            ('xgb',xgb)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('svck',svck),

                                            ('rf',rf),

                                            ('xgb',xgb)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)