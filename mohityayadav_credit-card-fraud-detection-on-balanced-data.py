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
# Importing libraries for this project 



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import metrics

from sklearn import preprocessing



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer



from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

import time





import warnings

warnings.filterwarnings("ignore")



from imblearn import over_sampling

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import ADASYN

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score,f1_score,classification_report



from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

from collections import Counter # counter takes values returns value_counts dictionary

from sklearn.datasets import make_classification
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
print(df.shape)

print(df.dtypes)

print(df[['Time','Amount','Class']].describe())

print(df.isnull().sum().max())


print(df['Class'].value_counts())

print('\n')

print(df['Class'].value_counts(normalize=True))



sns.countplot(df['Class'])


plt.figure(figsize=(12,8))

plt.subplot(1,2,1)

sns.scatterplot(data=df, x="Class", y="Amount")

plt.subplot(1,2,2)

sns.scatterplot(data=df, x="Class", y="Time")

plt.show()

plt.figure(figsize=(12,8))



sns.lmplot(data=df, x="Amount", y="Time",hue='Class')


sns.boxplot(x='Amount',data=df)


plt.title('Pearson Correlation Matrix')

sns.heatmap(df[['Time', 'Amount','Class']].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="winter",

            linecolor='w',annot=True);
# Removing unwanted column



df.drop(['Time'],axis=1,inplace=True)
# Splitting the data set into train and test data

# using stratify here so that all fraud data is equally stratified.



X=df.drop(['Class'],axis=1)

y=df['Class']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .30,stratify=y,shuffle=True,random_state=100)
# Plot the distribution of variables (to check the skewness)

df.hist(figsize = (25,25))

plt.show()
# Let's treat skewness by using power transformer by making it gausian

# Also scaling the Amount column



pt=PowerTransformer(method='yeo-johnson',standardize=True,copy=False)

X_train = pt.fit_transform(X_train)

X_test = pt.transform(X_test)
print(np.sum(y))

print(np.sum(y_train))

print(np.sum(y_test))
# instantiate the model



logreg=LogisticRegression(random_state=100)

# Hyperparamter Tuning and Cross validation



# #specify number of folds for k-fold CV

# C=np.logspace(-5,8,15)

# penalty=['l1','l2']





# param_grid=dict(C=C,penalty=penalty)

# # parameters to build the model on

# # parameters = {'max_depth': range(2, 20, 5)}







# # fit tree on training data

# logreg_cv = GridSearchCV(logreg, param_grid, 

#                     cv=3, n_jobs=-1,scoring='roc_auc')



# start_time=time.time()

# random_result=logreg_cv.fit(X_train, y_train)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf=LogisticRegression(C=0.006105402296585327,dual=False,penalty='l2',random_state=100)

clf.fit(X_train,y_train)
# Predictions on train data



y_predd=clf.predict(X_train)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_train))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_train , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_train , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_train , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_train , y_predd)))

print(metrics.classification_report(y_train, y_predd))

print(metrics.confusion_matrix(y_train,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_train, y_predd)



auc = metrics.roc_auc_score(y_train, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()





## Predictions on Test data



y_pred=clf.predict(X_test)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()



feature_importance = abs(clf.coef_[0])

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure(figsize=(15,10))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()   

plt.show()
df1=[[81.099,62.209,77.356,54.730,'V14,V4,V12']]

lr=pd.DataFrame(df1,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

lr
lr[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(random_state=100)
# # Let's do hyperparameter tuning 





# # GridSearchCV to find optimal max_depth

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV



# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'max_depth': [10,15],

#     'min_samples_leaf' : [15,25],

#     'min_samples_split': [15,25],

#     'n_estimators': [300,500]

# #    

# }

# # Create a based model

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1,verbose = 1,scoring='roc_auc',return_train_score=True)



# start_time=time.time()

# grid_search.fit(X_train, y_train)



# print('Execution time: ' + str((time.time()-start_time))+'seconds')
# print('We can get roc_auc of',grid_search.best_score_,'using',grid_search.best_params_)

clf = RandomForestClassifier(     max_depth=15,

                                  min_samples_leaf=15, 

                                  min_samples_split=15,

                                  n_estimators=500 ,

                                  n_jobs = -1,

                                  random_state=100

                                                                   

                                  )
clf.fit(X_train, y_train)
# Predictions on train data



y_predd=clf.predict(X_train)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_train))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_train , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_train , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_train , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_train , y_predd)))

print(metrics.classification_report(y_train, y_predd))

print(metrics.confusion_matrix(y_train,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_train, y_predd)



auc = metrics.roc_auc_score(y_train, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()





## Predictions on Test data



y_pred=clf.predict(X_test)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
df2=[[89.094,78.198,85.465,70.946,'V14,V4,V12']]

rf=pd.DataFrame(df2,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

rf

rf[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



dt= DecisionTreeClassifier(random_state=100)
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [5,10],

#     'min_samples_leaf' : [15,25],

#     'min_samples_split': [15,25],

#     'criterion': ["entropy", "gini"]

# }



# n_folds = 3



# # Instantiate the grid search model

# dtree = DecisionTreeClassifier()

# grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_train, y_train)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf = DecisionTreeClassifier(max_depth=5,criterion='entropy', min_samples_leaf= 25, min_samples_split= 15,random_state=100)

clf.fit(X_train, y_train)
# Predictions on train data



y_predd=clf.predict(X_train)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_train))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_train , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_train , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_train , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_train , y_predd)))

print(metrics.classification_report(y_train, y_predd))

print(metrics.confusion_matrix(y_train,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_train, y_predd)



auc = metrics.roc_auc_score(y_train, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()





## Predictions on Test data



y_pred=clf.predict(X_test)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
df3=[[88.656,77.326,86.814,73.649,'V17,V14,V10']]

dt=pd.DataFrame(df3,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

dt

dt[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
!pip install xgboost


from xgboost import XGBClassifier

model = XGBClassifier(random_state=100)
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [6,12],

#     'n_rounds' : [100,200],

#     'gamma'  : [0,5]

# }



# n_folds = 3



# # Instantiate the grid search model

# xg = XGBClassifier()

# grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_train, y_train)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time)))
clf=XGBClassifier(gamma= 5, max_depth= 12, n_rounds= 100,random_state=100)

clf.fit(X_train,y_train)
# Predictions on train data



y_predd=clf.predict(X_train)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_train))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_train , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_train , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_train , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_train , y_predd)))

print(metrics.classification_report(y_train, y_predd))

print(metrics.confusion_matrix(y_train,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_train, y_predd)



auc = metrics.roc_auc_score(y_train, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

## Predictions on Test data



y_pred=clf.predict(X_test)

print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
df4=[[93.895,87.791,88.507,77.027,'V17,V14,V10']]

xg=pd.DataFrame(df4,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

xg
xg[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
from collections import Counter # counter takes values returns value_counts dictionary

from sklearn.datasets import make_classification
print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=100)

X_res, y_res = rus.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))
# #specify number of folds for k-fold CV

# C=np.logspace(-5,8,15)

# dual=[True,False]



# penalty=['l1','l2']





# param_grid=dict(dual=dual,C=C,penalty=penalty)

# # parameters to build the model on

# # parameters = {'max_depth': range(2, 20, 5)}



# # instantiate the model

# logregg1 = LogisticRegression(random_state=100,solver='saga',warm_start=True,fit_intercept=True)





# # fit tree on training data

# logreg_cv = GridSearchCV(logregg1, param_grid, 

#                     cv=3, n_jobs=-1,scoring='roc_auc')



# start_time=time.time()

# random_result=logreg_cv.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf=LogisticRegression(C=0.05179474679231213,dual=False,penalty='l1',random_state=100,solver='saga',warm_start=True,fit_intercept=True)

clf.fit(X_res, y_res)

y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
feature_importance = abs(clf.coef_[0])

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure(figsize=(15,10))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=12)

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()   

plt.show()
data=[[94.186,90.698,92.674,87.162,'V14,V4,V12']]

Lr_rus=pd.DataFrame(data,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

Lr_rus
Lr_rus[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=100)

X_res, y_res = smote.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))

# # specify number of folds for k-fold CV

# C=np.logspace(-5,8,15)



# penalty=['l1','l2']





# param_grid=dict(dual=dual,C=C,penalty=penalty)

# # # parameters to build the model on

# # # parameters = {'max_depth': range(2, 20, 5)}



# # # instantiate the model

# logregg1 = LogisticRegression(random_state=100,solver='saga',dual=False,warm_start=True,fit_intercept=True)





# # # fit tree on training data

# logreg_cv = GridSearchCV(logregg1, param_grid, 

#                     cv=3, n_jobs=-1,scoring='roc_auc')



# start_time=time.time()

# random_result=logreg_cv.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf=LogisticRegression(C=19306.977288832535,dual=False,penalty='l1',random_state=100,solver='saga',warm_start=True,fit_intercept=True)

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
feature_importance = abs(clf.coef_[0])

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure(figsize=(15,10))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=12)

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()   

plt.show()
data1=[[94.912,92.619,93.243,89.189,'V4,V10,V14']]

lr_sm=pd.DataFrame(data1,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features']

                  )

lr_sm
lr_sm[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=100)



X_res, y_res = adasyn.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))
# # specify number of folds for k-fold CV

# C=np.logspace(-5,8,15)

# dual=[True,False]



# penalty=['l1','l2']





# param_grid=dict(dual=dual,C=C,penalty=penalty)

# # # parameters to build the model on

# # # parameters = {'max_depth': range(2, 20, 5)}



# # # instantiate the model

# logregg1 = LogisticRegression(random_state=100,solver='saga',warm_start=True,fit_intercept=True)





# # # fit tree on training data

# logreg_cv = GridSearchCV(logregg1, param_grid, 

#                     cv=3, n_jobs=-1,scoring='roc_auc')



# start_time=time.time()

# random_result=logreg_cv.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf=LogisticRegression(C=0.0007196856730011522,dual=False,penalty='l1',random_state=100,solver='saga',warm_start=True,fit_intercept=True)

clf.fit(X_res, y_res)

y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
feature_importance = abs(clf.coef_[0])

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure(figsize=(15,10))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=12)

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()   

plt.show()
data=[[89.249,87.037,92.392,93.243,'V4,V14,V10']]

Lr_ad=pd.DataFrame(data,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

Lr_ad
Lr_ad[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
from collections import Counter # counter takes values returns value_counts dictionary

from sklearn.datasets import make_classification



print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=100)

X_res, y_res = rus.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))
# # Let's Try hyperparameter tuning



# # GridSearchCV to find optimal max_depth

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import RandomForestClassifier





# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'max_depth': [5,10],

#     'min_samples_leaf' : [5,15,25],

#     'min_samples_split': [5,15, 25],

#     'n_estimators': [100,200]

# #     'max_features': [10,15,20]

# }

# # Create a based model

# rf = RandomForestClassifier(random_state=0)

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1,verbose = 1,scoring='roc_auc',return_train_score=True)



# start_time=time.time()

# grid_search.fit(X_res, y_res)



# print('Execution time: ' + str((time.time()-start_time))+'seconds')

# print('We can get roc_auc of',grid_search.best_score_,'using',grid_search.best_params_)

clf = RandomForestClassifier(     max_depth=10,

                                  min_samples_leaf=5, 

                                  min_samples_split=5,

                                  n_estimators=200 ,

                                   n_jobs = -1,

                                  random_state =100                                  

                                  )

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data3=[[96.948,94.767,92.558,87.838,'V14,V10,V4']]

rf_rus=pd.DataFrame(data3,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

rf_rus

rf_rus[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=100)

X_res, y_res = smote.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))
# # Let's Try hyperparameter tuning



# # GridSearchCV to find optimal max_depth

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV



# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'max_depth': [10,15],

#     'min_samples_leaf' : [25,50],

#     'min_samples_split': [25, 50],

#     'n_estimators': [500,700]

# #     'max_features': [10,15,20]

# }

# # Create a based model

# rf = RandomForestClassifier(random_state=42)

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1,verbose = 1,scoring='roc_auc',return_train_score=True)



# start_time=time.time()

# grid_search.fit(X_res, y_res)



# print('Execution time: ' + str((time.time()-start_time))+'seconds')

# print('We can get roc_auc of',grid_search.best_score_,'using',grid_search.best_params_)
clf = RandomForestClassifier(     max_depth=10,

                                  min_samples_leaf=25, 

                                  min_samples_split=25,

                                  n_estimators=700 ,

                                  oob_score = True, n_jobs = -1,

                                  random_state =100                                  

                                  )

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data4=[[99.176,98.511,92.819,85.811,'V14,V10,V4']]

rf_sm=pd.DataFrame(data4,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

rf_sm
rf_sm[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=100)



X_res, y_res = adasyn.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))
# # Let's Try hyperparameter tuning



# # GridSearchCV to find optimal max_depth

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV



# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'max_depth': [10,15],

#     'min_samples_leaf' : [25,50],

#     'min_samples_split': [25, 50],

#     'n_estimators': [500,700]

# #     'max_features': [10,15,20]

# }

# # Create a based model

# rf = RandomForestClassifier(random_state=100)

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1,verbose = 1,scoring='roc_auc',return_train_score=True)



# start_time=time.time()

# grid_search.fit(X_res, y_res)



# print('Execution time: ' + str((time.time()-start_time))+'seconds')

# print('We can get roc_auc of',grid_search.best_score_,'using',grid_search.best_params_)

clf = RandomForestClassifier(     max_depth=10,

                                  min_samples_leaf=25, 

                                  min_samples_split=25,

                                  n_estimators=700 ,

                                  oob_score = True, n_jobs = -1,

                                  random_state =100                                 

                                  )

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data5=[[99.275,99.631,93.057,87.162,'V4,V14,V17']]

rf_ad=pd.DataFrame(data5,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

rf_ad

rf_ad[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
from collections import Counter # counter takes values returns value_counts dictionary

from sklearn.datasets import make_classification



print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=100)

X_res, y_res = rus.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [6,12],

#     'n_rounds' : [50,100],

#     'gamma': [0,5],

# }



# n_folds = 3



# # Instantiate the grid search model

# xg = XGBClassifier(random_state=100)

# grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time)))
clf=XGBClassifier(gamma= 5, max_depth= 6, n_rounds= 50,random_state=100)

clf.fit(X_res,y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data6=[[97.384,95.640,92.060,87.838,'V14,V10,V4']]

xg_rus=pd.DataFrame(data6,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

xg_rus

xg_rus[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=100)

X_res, y_res = smote.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))

# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [6,12,18],

#     'n_rounds' : [100,200,300]

    

# }



# n_folds = 3



# # Instantiate the grid search model

# xg = XGBClassifier(random_state=100,gamma=5)

# grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time)))
clf=XGBClassifier(gamma= 5, max_depth= 6,learning_rate=0.2,random_state=100, n_rounds= 100,min_child_weight=1,colsample_bytree=0.7)

clf.fit(X_res,y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data7=[[99.988,01.000,92.528,85.135,'V14,V10,V4']]

xg_sm=pd.DataFrame(data6,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

xg_sm
xg_sm[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=100)



X_res, y_res = adasyn.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [6,12],

#     'n_rounds' : [100,300],

# }



# n_folds = 3



# # Instantiate the grid search model

# xg = XGBClassifier(random_state=100,gamma=5)

# grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time)))
clf=XGBClassifier(gamma= 5, max_depth= 12,learning_rate=0.2,random_state=100, n_rounds= 100,min_child_weight=1,colsample_bytree=0.7)

clf.fit(X_res,y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data8=[[99.996,100.000,90.848,81.757,'V4,V14,V10']]

xg_ad=pd.DataFrame(data8,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

xg_ad

xg_ad[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



rus = RandomUnderSampler(random_state=100)

X_res, y_res = rus.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [5,10],

#     'min_samples_leaf' : [5,25,45],

#     'min_samples_split': [5,25,45],

#     'criterion': ["entropy", "gini"]

# }



# n_folds = 3



# # Instantiate the grid search model

# dtree = DecisionTreeClassifier()

# grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf = DecisionTreeClassifier(max_depth=5,criterion='gini', min_samples_leaf= 5, min_samples_split= 45,random_state=100)

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data9=[[95.203,93.314,90.759,87.162,'V14,V4,V7']]

dt_rus=pd.DataFrame(data9,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

dt_rus
dt_rus[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



smote = SMOTE(random_state=100)

X_res, y_res = smote.fit_resample(X_train, y_train)



print('Resampled dataset shape %s' % Counter(y_res))

# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [5,10,15],

#     'min_samples_leaf' : [5,25,50],

#     'min_samples_split': [5,25,50],

#     'criterion': ["entropy", "gini"]

# }



# n_folds = 3



# # Instantiate the grid search model

# dtree = DecisionTreeClassifier()

# grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf = DecisionTreeClassifier(max_depth=15,criterion='entropy', min_samples_leaf= 50, min_samples_split= 25,random_state=100)

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data10=[[99.345,99.545,91.767,84.459,'V14,V4,V12']]

dt_sm=pd.DataFrame(data10,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

dt_sm

dt_sm[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
print('Original dataset shape %s' % Counter(y))



adasyn = ADASYN(random_state=100)



X_res, y_res = adasyn.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))
# # GridSearchCV to find optimal n_estimators

# from sklearn.model_selection import KFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import RandomizedSearchCV

# import time







# param_grid = {

#     'max_depth': [5,10,15],

#     'min_samples_leaf' : [5,25,50],

#     'min_samples_split': [5,25,50],

#     'criterion': ["entropy", "gini"]

# }



# n_folds = 3



# # Instantiate the grid search model

# dtree = DecisionTreeClassifier(random_state=100)

# grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

#                           cv = n_folds, verbose = 1,scoring='roc_auc',return_train_score=True,n_jobs=-1)



# start_time=time.time()

# random_result=grid_search.fit(X_res, y_res)



# print('Best: %f using %s' % (random_result.best_score_,random_result.best_params_))

# print('Execution time: ' + str((time.time()-start_time))+'seconds')
clf = DecisionTreeClassifier(max_depth=5,criterion='gini', min_samples_leaf= 50, min_samples_split= 5,random_state=100)

clf.fit(X_res, y_res)
y_predd=clf.predict(X_res)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_predd , y_res))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_res , y_predd)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_res , y_predd)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_res , y_predd)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_res , y_predd)))

print(metrics.classification_report(y_res, y_predd))

print(metrics.confusion_matrix(y_res,y_predd))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_res, y_predd)



auc = metrics.roc_auc_score(y_res, y_predd)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
y_pred=clf.predict(X_test)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred , y_test))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_test , y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_test , y_pred)))

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))

# plot ROC Curve



plt.figure(figsize=(8,6))



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)



auc = metrics.roc_auc_score(y_test, y_pred)

print("AUC - ",auc,"\n")



plt.plot(fpr,tpr,linewidth=2, label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a credit card fraud detection')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
pd.concat((pd.DataFrame(X.columns,columns=['Variable']),

           pd.DataFrame(clf.feature_importances_,columns=['importance'])),axis=1).sort_values(by='importance',ascending=False)
data11=[[92.762,93.512,90.009,87.838,'V4,V14,V7']]

dt_ad=pd.DataFrame(data11,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'])

dt_ad

dt_ad[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')



final1=[[81.099,62.209,77.356,54.730,'V14,V4,V12'],[89.094,78.198,85.465,70.946,'V14,V4,V12'],[88.656,77.326,86.814,73.649,'V17,V14,V10'],[93.895,87.791,88.507,77.027,'V17,V14,V10']]

data_imbalanced=pd.DataFrame(final1,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'],index =['Logisticreg', 'RandomForest', 'DecisionTree', 'XGBOOST'])

data_imbalanced
plt.figure(figsize=(25,10))



data_imbalanced[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
## Checking all models of logistic regression and choosing the best one 



dataa1=[[81.099,62.209,77.356,54.730,'V14,V4,V12'],[94.186,90.698,92.674,87.162,'V14,V4,V12'],[94.912,92.619,93.243,89.189,'V4,V10,V14'],[89.249,87.037,92.392,93.243,'V4,V14,V10']]

Lr_final=pd.DataFrame(dataa1,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'],index =['imbalanced', 'Randomundersampler', 'SMOTE', 'ADASYN'])

Lr_final
Lr_final[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
## Checking all models of Random Forest and choosing the best one 



dataa2=[[89.094,78.198,85.465,70.946,'V14,V4,V12'],[96.948,94.767,92.558,87.838,'V14,V10,V4'],[99.176,98.511,92.819,85.811,'V14,V10,V4'],[99.275,99.631,93.057,87.162,'V4,V14,V17']]

rf_final=pd.DataFrame(dataa2,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'],index =['imbalanced', 'Randomundersampler', 'SMOTE', 'ADASYN'])

rf_final

rf_final[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
dataa3=[[88.656,77.326,86.814,73.649,'V17,V14,V10'],[95.203,93.314,90.759,87.162,'V14,V4,V7'],[99.345,99.545,91.767,84.459,'V14,V4,V12'],[92.762,93.512,90.009,87.838,'V4,V14,V7']]

dt_final=pd.DataFrame(dataa3,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'],index =['imbalanced', 'Randomundersampler', 'SMOTE', 'ADASYN'])

dt_final
dt_final[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')
dataa4=[[93.895,87.791,88.507,77.027,'V17,V14,V10'],[97.384,95.640,92.060,87.838,'V14,V10,V4'],[99.988,100.000,92.528,85.135,'V14,V10,V4'],[99.996,100.000,90.848,81.757,'V4,V14,V10']]

xg_final=pd.DataFrame(dataa4,columns=['AUC Train','Recall Train','AUC Test','Recall Test','Top 3 Features'],index =['imbalanced', 'Randomundersampler', 'SMOTE', 'ADASYN'])

xg_final

xg_final[['AUC Train','Recall Train','AUC Test','Recall Test']].plot(kind='bar')