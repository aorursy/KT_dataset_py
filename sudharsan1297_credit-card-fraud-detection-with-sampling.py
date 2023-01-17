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
# Importing the Dataset

df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
#Number of rows and features

df.shape
# Checking for the balance of categories in the target variable.

# The feature is highly imbalanced

df['Class'].value_counts()
# Splittig the data into majority and minority based on the target variable. 

from sklearn.utils import resample

df_minor=df[df['Class']==1]

df_major=df[df['Class']==0]



# Upsampling the minority class to the number of rows in the majority class by using the

# sampling with replacement technique.

df_minority_upsampled=resample(df_minor,replace=True,n_samples=len(df_major),random_state=42)



# Concatenating the majority data and the upsampled data

df_sampled=pd.concat([df_major,df_minority_upsampled])

df_sampled['Class'].shape
# Checking for the co-relation of independent features with the target variable 

df_sampled.corr()['Class']
# Extracting features that are highly co-related with the dependent variable as these are the

# features responsible for the variation of values in the dependent variable.

corr_cols=df_sampled.corr()['Class'][((df_sampled.corr()['Class']>0.5) | (df_sampled.corr()['Class']<-0.5))].index
# Apart from the features above, the duration of the transaction and the amount used for the

# transaction also plays a role in predicting the transaction as fraudulent or not.

df1=df_sampled[corr_cols]

col=df_sampled[['Time','Amount']]

df1=pd.concat([df1,col],1)

df1.head()
# Scaling the time and amount variable to bring all the independent variables to the same scale.

from sklearn.preprocessing import StandardScaler

std=StandardScaler().fit_transform(df1[['Time','Amount']])

df1[['Time','Amount']]=std
df1.head()
# Splitting X and y

from sklearn.model_selection import train_test_split

X=df1.drop('Class',1)

y=df1['Class']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

for i in [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]:

    lr=LogisticRegression(C=i).fit(X_train,y_train)

    y_pred_lr=lr.predict(X_test)

    print(accuracy_score(y_test,y_pred_lr))

    
lr=LogisticRegression(C=0.00001).fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)

print(accuracy_score(y_test,y_pred_lr))
# Varying the threshold value of the probability to check for the least misclassified values.

from sklearn.preprocessing import binarize

from sklearn.metrics import f1_score,confusion_matrix

for i in range(1,11):

    y_pred2=lr.predict_proba(X_test)

    bina=binarize(y_pred2,threshold=i/10)[:,1]

    cm2=confusion_matrix(y_test,bina)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(y_test,bina))

    print('accuracy score: ',accuracy_score(y_test,bina))

    print('\n')
from sklearn.metrics import confusion_matrix

print('Confusion matrix for Logistic Regression: ',confusion_matrix(y_test,y_pred_lr))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier().fit(X_train,y_train)

y_pred_dt=dt.predict(X_test)

print(accuracy_score(y_test,y_pred_dt))

print(confusion_matrix(y_test,y_pred_dt))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier().fit(X_train,y_train)

y_pred_rf=rf.predict(X_test)

print('Accuracy score for Random Forest: ',accuracy_score(y_test,y_pred_rf))

print('Confusion matrix for Random Forest: ',confusion_matrix(y_test,y_pred_rf))
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

adb=AdaBoostClassifier().fit(X_train,y_train)

y_pred_adb=adb.predict(X_test)

print("Adaboost Classifier's accuracy: ",accuracy_score(y_test,y_pred_adb))

print("Adaboost Classifier's confusion matrix: ",confusion_matrix(y_test,y_pred_adb))
grb=GradientBoostingClassifier().fit(X_train,y_train)

y_pred_grb=grb.predict(X_test)

print("Gradient boost Classifier's accuracy: ",accuracy_score(y_test,y_pred_grb))

print("Gradient boost Classifier's confusion matrix: ",confusion_matrix(y_test,y_pred_grb))
from sklearn.utils import resample

df_minor=df[df['Class']==1]

df_major=df[df['Class']==0]

# Reducing the number of data in the majority class to the number of observations in the minority class.

df_majority_downsampled=resample(df_major,replace=False,n_samples=len(df_minor),random_state=42)



df_downsampled=pd.concat([df_minor,df_majority_downsampled])
df_downsampled.shape
corr_col=df_downsampled.corr()['Class'][((df_downsampled.corr()['Class']>0.5) | (df_downsampled.corr()['Class']<-0.5))].index

df2=df_downsampled[corr_col]
cols=df_downsampled[['Time','Amount']]

df2=pd.concat([df2,cols],1)

df2.head()
df2.reset_index(drop=True,inplace=True)

df2.head()
df2.shape
std_us=StandardScaler().fit_transform(df2[['Time','Amount']])

df2[['Time','Amount']]=std_us

df2.head()
from sklearn.model_selection import train_test_split

X_us=df2.drop('Class',1)

y_us=df2['Class']



X_train1,X_test1,y_train1,y_test1=train_test_split(X_us,y_us,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

for i in [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]:

    lr_us=LogisticRegression(C=i).fit(X_train1,y_train1)

    y_pred_lr_us=lr_us.predict(X_test1)

    print(accuracy_score(y_test1,y_pred_lr_us))
lr_us=LogisticRegression(C=0.1).fit(X_train1,y_train1)

y_pred_lr_us=lr_us.predict(X_test1)

print(accuracy_score(y_test1,y_pred_lr_us))
for i in range(1,11):

    y_pred2_us=lr_us.predict_proba(X_test1)

    bina_us=binarize(y_pred2_us,threshold=i/10)[:,1]

    cm2=confusion_matrix(y_test1,bina_us)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    print('f1 score: ',f1_score(y_test1,bina_us))

    print('accuracy score: ',accuracy_score(y_test1,bina_us))

    print('\n')
# Taking 0.5 as the threshold as it gives the least misclassified values

y_pred2_us=lr_us.predict_proba(X_test1)

bina_us=binarize(y_pred2_us,threshold=0.5)[:,1]

cm2=confusion_matrix(y_test1,bina_us)

print ('With 0.5 threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

print('f1 score: ',f1_score(y_test1,bina_us))

print('accuracy score: ',accuracy_score(y_test1,bina_us))

print('\n')
from sklearn.tree import DecisionTreeClassifier

dt_us=DecisionTreeClassifier().fit(X_train1,y_train1)

y_pred_dt_us=dt_us.predict(X_test1)

print(accuracy_score(y_test1,y_pred_dt_us))

print(confusion_matrix(y_test1,y_pred_dt_us))
# Grid Search CV for hyperparameter tuning

from sklearn.model_selection import GridSearchCV

dt_us=DecisionTreeClassifier()

param_grid = {



    'criterion': ['gini','entropy'],

    'max_depth': [4,6,8,10],

    'min_samples_split' : [5,10,15,20,25,30],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

}



dt_grid_us=GridSearchCV(estimator=dt_us,param_grid=param_grid,n_jobs=-1,return_train_score=True)



dt_grid_us.fit(X_train1,y_train1)
# Checking for the best parameters of the grid search cv that gives the highest result.

dt_grid_us.best_params_
# The results of the cross validated grid search 

cv_res_df_us=pd.DataFrame(dt_grid_us.cv_results_)
# Plotting the mean test score and mean train score and extracting the point where the test score

# is high and also the difference in the train and test score is minimum.

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.plot(cv_res_df_us['mean_train_score'])

plt.plot(cv_res_df_us['mean_test_score'])

plt.xticks(rotation=90)

plt.show()
cv_res_df_us[cv_res_df_us['mean_test_score']==cv_res_df_us['mean_test_score'].max()]
# Using the parameters that yielded the best results.

dt_us=DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,min_samples_split=15,random_state=135).fit(X_train1,y_train1)

y_pred_dt_us=dt_us.predict(X_test1)

print(accuracy_score(y_test1,y_pred_dt_us))

print(confusion_matrix(y_test1,y_pred_dt_us))
from sklearn.ensemble import RandomForestClassifier

rf_us=RandomForestClassifier().fit(X_train1,y_train1)

y_pred_rf_us=rf.predict(X_test1)

print(accuracy_score(y_test1,y_pred_rf_us))

print(confusion_matrix(y_test1,y_pred_rf_us))
# Hyper parameter tuning using GridSearchCV

rf_us=RandomForestClassifier()

param_grid = {

    'n_estimators': [8,10,20,30],

    'criterion': ['gini','entropy'],

    'max_depth': [4,6,8,10],

    'min_samples_split' : [5,10,15,20,25,30],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

}



rf_grid=GridSearchCV(estimator=rf_us,param_grid=param_grid,n_jobs=-1,return_train_score=True)



rf_grid.fit(X_train1,y_train1)
rf_grid.best_params_
cv_res_df_rf_us=pd.DataFrame(rf_grid.cv_results_)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.plot(cv_res_df_rf_us['mean_train_score'])

plt.plot(cv_res_df_rf_us['mean_test_score'])

plt.xticks(rotation=90)

plt.show()
min(cv_res_df_rf_us['mean_train_score']-cv_res_df_rf_us['mean_test_score'])
cv_res_df_rf_us[(cv_res_df_rf_us['mean_test_score']==cv_res_df_rf_us['mean_test_score'].max())]
rf_us=RandomForestClassifier(**rf_grid.best_params_).fit(X_train1,y_train1)

y_pred_rf_us=rf.predict(X_test1)

print(" Random Forest Classifier's accuracy: ",accuracy_score(y_test1,y_pred_rf_us))

print(" Random Forest Classifier's confusion matrix: ",confusion_matrix(y_test1,y_pred_rf_us))
adbc=AdaBoostClassifier().fit(X_train1,y_train1)

y_pred_adbc=adbc.predict(X_test1)

print("Adaboost Classifier's accuracy: ",accuracy_score(y_test1,y_pred_adbc))

print(" Adaboost Classifier's confusion matrix: ",confusion_matrix(y_test1,y_pred_adbc))
gra_bc=GradientBoostingClassifier().fit(X_train1,y_train1)

y_pred_gra_bc=gra_bc.predict(X_test1)

print(" Gradient boost Classifier's accuracy: ",accuracy_score(y_test1,y_pred_gra_bc))

print(" Gradient boost Classifier's confusion matrix: ",confusion_matrix(y_test1,y_pred_gra_bc))
y_pred_rf=rf.predict(X_test)

print('Accuracy score for Random Forest: ',accuracy_score(y_test,y_pred_rf))

print('Confusion matrix for Random Forest: \n ',confusion_matrix(y_test,y_pred_rf))
sub=pd.DataFrame(y_test,columns=['Actual Class'])

sub['Predicted Class']=y_pred_rf

sub.to_csv('submission.csv')