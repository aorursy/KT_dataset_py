#Importing The Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Importing The Dataset

train = pd.read_csv('/kaggle/input/titanic/train.csv')

print('Train Data: \n\n',train.head(10))

print('*******************************************************')

print('Train Data Shape: ',train.shape)

print('*******************************************************')

print(train.columns)

print('*******************************************************')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print('Test Data: \n\n',test.head(10))

print('*******************************************************')

print('Test Data Shape: ',test.shape)

print('*******************************************************')

print(test.columns)

print('*******************************************************')
#Drop Unless Columns

train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

print(train.head(10))

print('**************************************************************')

print(test.head(10))

print(train.columns)

print(train.shape)

print('**************************************************************')

print(test.columns)

print(test.shape)
#To know the number of successful people and the number of deaths

print(train['Survived'].value_counts())

train['Survived'].value_counts().plot(kind='bar',alpha=0.7)

plt.title('Survived')
#To know the number of men and the number of women

print(train.Sex.value_counts())

train.Sex.value_counts().plot(kind='bar',)
#To know how many men survived and how many men died

print(train.Survived[train.Sex=='male'].value_counts())

train.Survived[train.Sex=='male'].value_counts().plot(kind='bar',alpha=1)

plt.title('Surviveds for their Sex [ Male ]' )
#To know how many men survived and how many men died

print(train.Survived[train.Sex=='female'].value_counts())

train.Survived[train.Sex=='female'].value_counts().plot(kind='bar',color='green')

plt.title('Surviveds for their Sex [ Female ]')
#To know the division of ship degrees

print(train['Pclass'].value_counts())

train['Pclass'].value_counts().plot(kind='bar')

plt.title('Surviveds for their P Class')
#To know how many people live in every class

for x in [1,2,3]:

  train.Survived[train.Pclass==x].plot(kind='kde')

plt.title('Survived for their P Class')

plt.legend(('1st','2nd','3rd'))
#To know how many men are in each class of ship

print(train.Pclass[train.Sex=='male'].value_counts())

train.Pclass[train.Sex=='male'].value_counts().plot(kind='bar',alpha=1)

plt.title('Pclass for their Sex [ Male ]' )
#To know how many women are in each class of ship

print(train.Pclass[train.Sex=='female'].value_counts())

train.Pclass[train.Sex=='female'].value_counts().plot(kind='bar',color='green')
print(train.Survived[(train.Sex=='male')&(train.Pclass==1)].value_counts())

train.Survived[(train.Sex=='male')&(train.Pclass==1)].value_counts().plot(kind='bar')

plt.title('Rich Man Survived')
print(train.Survived[(train.Sex=='male')&(train.Pclass==3)].value_counts())

train.Survived[(train.Sex=='male')&(train.Pclass==3)].value_counts().plot(kind='bar')

plt.title('Poor Man Survived')
print(train.Survived[(train.Sex=='female')&(train.Pclass==1)].value_counts())

train.Survived[(train.Sex=='female')&(train.Pclass==1)].value_counts().plot(kind='bar',color='green')
print(train.Survived[(train.Sex=='female')&(train.Pclass==3)].value_counts())

train.Survived[(train.Sex=='female')&(train.Pclass==3)].value_counts().plot(kind='bar',color='green')
plt.scatter(train['Survived'],train['Age'],alpha=0.7)

plt.title('Survivors for their ages')

plt.xlabel('Survived')

plt.ylabel('Age')
for x in [1,2,3]:

  train.Age[train.Pclass==x].plot(kind='kde')

plt.title('Age for their P Class')

plt.legend(('1st','2nd','3rd'))
plt.scatter(train.Sex,train.Age)

plt.title('Sex for their ages')

plt.xlabel('Sex')

plt.ylabel('Age')
print(train.Survived[train.SibSp].value_counts())

train.Survived[train.SibSp].value_counts().plot(kind='bar')
print(train.Survived[train.Parch].value_counts())

train.Survived[train.Parch].value_counts().plot(kind='bar')
Survived=train[train['Survived'].isin([1])]

Unsurvived=train[train['Survived'].isin([0])]

print('Survived',Survived)

print('Unsurvived',Unsurvived)
train.var()
train.std()
train.corr()
import seaborn as sns

sns.heatmap(train.corr(),annot=True)
print(train.isnull().sum())

print('***********************************')

print(test.isnull().sum())

import missingno as msno

msno.bar(train)
import missingno as msno

msno.bar(test)
X=train.iloc[:,1:].values

y=train.iloc[:,0:1].values

X_test_original=test.iloc[:,1:].values

print('X: ',X[:10,:])

print('X Shape: ',X.shape)

print('Y: ',y[:10,:])

print('Y_ Shape: ',y.shape)

print('X_test_original: ',X_test_original)

print('X_test_original: ',X_test_original.shape)
from sklearn.impute import SimpleImputer

imputer1=SimpleImputer(missing_values=np.nan,strategy='median')

imputer2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

X[:,2:3]=imputer1.fit_transform(X[:,2:3])

X[:,6:7]=imputer2.fit_transform(X[:,6:7])

X_test_original[:,2:3]=imputer1.fit_transform(X_test_original[:,2:3])

X_test_original[:,5:6]=imputer2.fit_transform(X_test_original[:,5:6])

print(X[:10,:])

print('***********************************')

print(X_test_original[:10,:])

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

dummy_variables = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')

X=dummy_variables.fit_transform(X)

X_test_original=dummy_variables.fit_transform(X_test_original)

le=LabelEncoder()

X[:,4]=le.fit_transform(X[:,4])

X_test_original[:,4]=le.fit_transform(X_test_original[:,4])

print(X[:10,:])

print('*********************')

print(X_test_original[:10,:])
#avoiding  the dummy variables trap

X= X[ : , 1:]
#avoiding  the dummy variables trap

X_test_original= X_test_original[ : , 1:]
print(X[:10,:])

print('******************')

print(X_test_original[:10,:])
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X=sc.fit_transform(X)

X_test_original=sc.fit_transform(X_test_original)

print(X)

print('*************************************')

print(X_test_original)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

LR=LogisticRegression(C=0.01,random_state=22)

CrossValidateValues1 = cross_validate(LR,X,y,cv=10,return_train_score = True)

CrossValidateValues2 = cross_validate(LR,X,y,cv=10,scoring=('r2','neg_mean_squared_error'),return_train_score = True)



# Showing Results

print('Train Score Value : ', CrossValidateValues1['train_score'])

print('Test Score Value : ', CrossValidateValues1['test_score'])

print('Fit Time : ', CrossValidateValues1['fit_time'])

print('Score Time : ', CrossValidateValues1['score_time'])

print('Train MSE Value : ', CrossValidateValues2['train_neg_mean_squared_error'])

print('Test MSE Value : ', CrossValidateValues2['test_neg_mean_squared_error'])

print('Train R2 Value : ', CrossValidateValues2['train_r2'])

print('Test R2 Value : ', CrossValidateValues2['test_r2'])





print('//////////////////////////////////////')





# Showing Results



accuracies = cross_val_score(estimator = LR, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

LR.fit(X,y)

print('Logistic Regression Train Score :',LR.score(X,y))
y_predict_LR=LR.predict(X)

print('Y Predict: ',y_predict_LR)
import seaborn as sns

from sklearn.metrics import confusion_matrix

CM_LR=confusion_matrix(y,y_predict_LR)

print(CM_LR)

sns.heatmap(CM_LR,center=True,annot=True)
from sklearn.metrics import accuracy_score

acc_LR=accuracy_score(y,y_predict_LR)

print(acc_LR)
from sklearn.metrics import classification_report

CR_LR=classification_report(y,y_predict_LR)

print(CR_LR)
from sklearn.metrics import roc_curve,auc

LR_tpr,LR_fpr,threshold=roc_curve(y,y_predict_LR)

LR_auc = auc(LR_tpr, LR_fpr)

print('LR_tpr: ',LR_tpr)

print('LR_fpr: ',LR_fpr)

print('threshold: ',threshold)



#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 8))

plt.plot(LR_tpr, LR_fpr, marker='o', label='Logistic Regression (auc = %0.3f)' % LR_auc)

plt.ylabel('True Positive Rate -->')

plt.xlabel('False Positive Rate -->')



plt.legend()



plt.show()
from sklearn.svm import SVC

from sklearn.model_selection import cross_validate

SVM=SVC(C=0.6,kernel='rbf',random_state=22)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = SVM, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('/////////////////////')
SVM.fit(X,y)

print('Logistic Regression Train Score :',SVM.score(X,y))
y_predict_SVM=SVM.predict(X)

print(y_predict_SVM)
from sklearn.metrics import  confusion_matrix

CM_SVM=confusion_matrix(y,y_predict_SVM)

print(CM_SVM)

sns.heatmap(CM_SVM,center=True,annot=True)
from sklearn.metrics import accuracy_score

acc_svm=accuracy_score(y,y_predict_SVM)

print(acc_svm)
from sklearn.metrics import classification_report

CR_SVM=classification_report(y,y_predict_SVM)

print(CR_SVM)
from sklearn.metrics import roc_curve,auc

svm_tpr,svm_fpr,threshold=roc_curve(y,y_predict_SVM)

svm_auc=auc(svm_tpr,svm_fpr)

print('svm_tpr',svm_tpr)

print('svm_fpr',svm_fpr)

print('threshold',threshold)





#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 5))

plt.plot(svm_tpr, svm_fpr, linestyle=':', label='SVM (auc = %0.3f)' % svm_auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate

knn=KNeighborsClassifier(n_neighbors=15,metric = 'minkowski', p = 1)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = X, y = y, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('/////////////////////')
knn.fit(X,y)

print('KNN Train Score :',knn.score(X,y))
y_predict_knn=knn.predict(X)

print(y_predict_knn)
from sklearn.metrics import confusion_matrix

cm_knn=confusion_matrix(y,y_predict_knn)

print(cm_knn)

sns.heatmap(cm_knn,center=True,annot=True)
from sklearn.metrics import accuracy_score

acc_knn=accuracy_score(y,y_predict_knn)

print(acc_knn)
from sklearn.metrics import classification_report

cr_knn=classification_report(y,y_predict_knn)

print(cr_knn)
from sklearn.metrics import roc_curve,auc

knn_tpr,knn_fpr,threshold=roc_curve(y,y_predict_knn)

print('True Positive Rate',knn_tpr)

print('False Positive Rate',knn_fpr)

print('threshold',threshold)



knn_auc=auc(knn_tpr,knn_fpr)



#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 5))

plt.plot(knn_tpr, knn_fpr, linestyle='--', label='KNN (auc = %0.3f)' %knn_auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



dt=DecisionTreeClassifier(max_depth=6,max_features=4,min_samples_split=6,random_state=22)



parameters = [{'max_depth': [2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],

               'min_samples_split': [2,4,6,8,10,12,14,16,18],

               'max_features' : [2,4,6,8,10,12]

               }]

grid_search = GridSearchCV(estimator = dt,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search.fit(X, y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
dt.fit(X,y)

print(' DecisionTree Train Score :',dt.score(X,y))

print('DecisionTree Classifier Model feature importances are :\n ' , dt.feature_importances_)
y_predict_dt=dt.predict(X)

print(y_predict_dt)
from sklearn.metrics import accuracy_score

acc_dt=accuracy_score(y,y_predict_dt)

print(acc_dt)
from sklearn.metrics import confusion_matrix

cm_dt=confusion_matrix(y,y_predict_dt)

print(cm_dt)

sns.heatmap(cm_dt,center=True,annot=True)
from sklearn.metrics import classification_report

cr_dt=classification_report(y,y_predict_dt)

print(cr_dt)
from sklearn.metrics import roc_curve,auc

dt_tpr,dt_fpr,threshold=roc_curve(y,y_predict_dt)

print('True Positive Rate',dt_tpr)

print('False Positive Rate',dt_fpr)

print('threshold',threshold)



dt_auc=auc(dt_tpr,dt_fpr)



#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 5))

plt.plot(dt_tpr, dt_fpr, linestyle='--', label='DT (auc = %0.3f)' %dt_auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()



from sklearn.ensemble import RandomForestClassifier 

rf=RandomForestClassifier(criterion = 'entropy',max_depth=10,n_estimators=12,min_samples_split=20)

rf.fit(X,y)

print(' RandomForest Train Score :',rf.score(X,y))

print('RandomForest Classifier Model feature importances are :\n ' , rf.feature_importances_)
y_pred_rf=rf.predict(X)

print('Y Pred',y_pred_rf)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y,y_pred_rf)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True,annot=True)

plt.show()
from sklearn.metrics import accuracy_score

acc_rf=accuracy_score(y,y_pred_rf)

print('Accuracy Score',acc_rf)
from sklearn.model_selection import GridSearchCV

parameters = [{'max_depth': [2,4,6,7,8,9,10,11,12,13,14,15,16],

               'min_samples_split': [2,4,6,8,10,12,14,16,18,19,20],

               'n_estimators' : [2,4,6,8,10,12]

               }]

grid_search = GridSearchCV(estimator = rf,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search.fit(X, y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = rf, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('******************************')
from sklearn.metrics import classification_report

cr=classification_report(y,y_pred_rf)

print(cr)
from sklearn.metrics import roc_curve,auc

rf_tpr,rf_fpr,threshold=roc_curve(y,y_pred_rf)

rf_auc=auc(rf_tpr,rf_fpr)

print('rf_tpr Value  : ', rf_tpr)

print('rf_fpr Value  : ', rf_fpr)

print('thresholds Value  : ', threshold)



#Draw ROC Curve && AUC [Area Under The Curve]



plt.figure(figsize=(5, 5), dpi=100)

plt.plot(rf_tpr, rf_fpr, linestyle='-', label='Random Force (auc = %0.3f)' % rf_auc)



plt.xlabel('True Positive Rate -->')

plt.ylabel('False Positive Rate -->')



plt.legend()

plt.show()
from sklearn.naive_bayes import BernoulliNB

NB=BernoulliNB()

NB.fit(X,y)

print('Naive Bayse Train Score',NB.score(X,y))



#Calculating Prediction

y_pred_NB = NB.predict(X)

y_pred_prob = NB.predict_proba(X)

y_pred_prob2=y_pred_prob.astype(int)

print('Y Test' ,y)

print('Predicted Value for BernoulliNBModel is : ' , y_pred_NB)

print('Prediction Probabilities Value for BernoulliNBModel is : \n' , y_pred_prob2)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y,y_pred_NB)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True,annot=True)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = NB, X = X, y = y, cv = 10)

accuracies2 = cross_val_score(estimator = NB, X = X, y = y, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

print('******************************')

print("Accuracy: {:.2f} %".format(accuracies2.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies2.std()*100))
from sklearn.metrics import accuracy_score

acc_nb=accuracy_score(y,y_pred_NB)

print(acc_nb)
from sklearn.metrics import classification_report

cr=classification_report(y,y_pred_NB)

print(cr)
from sklearn.metrics import roc_curve,auc

nb_tpr,nb_fpr,threshold=roc_curve(y,y_pred_NB)

nb_auc=auc(nb_tpr,nb_fpr)

print('nb_tpr Value  : ', nb_tpr)

print('nb_fpr Value  : ', nb_fpr)

print('thresholds Value  : ', threshold)



#Draw ROC Curve && AUC [Area Under The Curve]



plt.figure(figsize=(5, 5), dpi=100)

plt.plot(rf_tpr, rf_fpr, linestyle='-', label='Naive Basye (auc = %0.3f)' % nb_auc)



plt.xlabel('True Positive Rate -->')

plt.ylabel('False Positive Rate -->')



plt.legend()

plt.show()
from sklearn.metrics import roc_curve, auc



LR_tpr,LR_fpr,threshold=roc_curve(y,y_predict_LR)

LR_auc = auc(LR_tpr, LR_fpr)



svm_fpr, svm_tpr, threshold = roc_curve(y, y_predict_SVM)

auc_svm = auc(svm_fpr, svm_tpr)



knn_fpr, knn_tpr, threshold = roc_curve(y, y_predict_knn)

auc_knn = auc(knn_fpr, knn_tpr)



dt_fpr, dt_tpr, threshold = roc_curve(y, y_predict_dt)

auc_dt = auc(dt_fpr, dt_tpr)



rf_fpr, rf_tpr, threshold = roc_curve(y, y_pred_rf)

auc_rf = auc(rf_fpr, rf_tpr)





nb_fpr, nb_tpr, threshold = roc_curve(y, y_pred_NB)

auc_nb = auc(nb_fpr, nb_tpr)



plt.figure(figsize=(9, 5))

plt.plot(LR_tpr, LR_fpr, marker='o', label='Logistic Regression (auc = %0.3f)' % LR_auc)

plt.plot(svm_fpr, svm_tpr, linestyle='--', label='SVM (auc = %0.3f)' % auc_svm)

plt.plot(knn_fpr, knn_tpr, linestyle=':', label='KNN (auc = %0.3f)' % auc_knn)

plt.plot(dt_fpr, dt_tpr, linestyle='-', label='DT (auc = %0.3f)' % auc_dt)

plt.plot(rf_fpr, rf_tpr, linestyle='-', label='RF (auc = %0.3f)' % auc_rf)

plt.plot(nb_fpr, nb_tpr, linestyle='--', label='NB (auc = %0.3f)' % auc_nb)







plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()
models = pd.DataFrame({

    'Model': ['Logistic Regression','Support Vector Machines', 'KNN','Decision Tree', 

              'Random Forest', 'Naive Bayes'],

    'Score': [acc_LR, acc_svm, acc_knn, 

              acc_dt, acc_rf, acc_nb]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions =rf.predict(X_test_original)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

print(output)