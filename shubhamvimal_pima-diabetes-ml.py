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
import matplotlib.pyplot as plt

import seaborn as sns

import itertools
diab=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

diab.head()
diab.isnull().sum()
sns.countplot(x='Outcome',data=diab)

plt.show()
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
outcome = diab['Outcome']

data = diab[diab.columns[:8]]

train,test = train_test_split(diab,test_size=0.25,random_state=0,stratify=diab['Outcome'])# stratify the outcome

x_train = train[train.columns[:8]]

x_test = test[test.columns[:8]]

y_train = train['Outcome']

y_test = test['Outcome']
accuracy = []

classifiers = ['Linear SVM','Radial SVM','Logistic Regression','KNN','Decision Tree']

models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]

for i in models:

    model = i

    model.fit(x_train,y_train)

    prediction = model.predict(x_test)

    accuracy.append(metrics.accuracy_score(prediction,y_test))

models_dataframe = pd.DataFrame(accuracy,index=classifiers)   

models_dataframe.columns = ['Accuracy']

models_dataframe
# Correlation Matrix

ax = sns.heatmap(diab[diab.columns[:8]].corr(),annot=True,cmap='inferno')

ax.set_ylim(8.0,0.0)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier 

model= RandomForestClassifier(n_estimators=100,random_state=0)

X=diab[diab.columns[:8]]

Y=diab['Outcome']

model.fit(X,Y)

pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
from sklearn.preprocessing import StandardScaler #Standardisation



# Taking Top 5 from above result

diab2 = diab[['Glucose','BMI','Age','DiabetesPedigreeFunction','Outcome']]

features = diab2[diab2.columns[:4]]

features_standard = StandardScaler().fit_transform(features)



x=pd.DataFrame(features_standard,columns = [['Glucose','BMI','Age','DiabetesPedigreeFunction']])

x['Outcome'] = diab2['Outcome']



train1,test1=train_test_split(x,test_size=0.25,random_state=0,stratify=x.iloc[:,-1])

x1_train = train1[train1.columns[:4]]

x1_test = test1[test1.columns[:4]]

y1_train = train1.iloc[:,-1]

y1_test = test1.iloc[:,-1]
accuracy = []

classifiers = ['Linear SVM','Radial SVM','Logistic Regression','KNN','Decision Tree']

models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]

for i in models:

    model = i

    model.fit(x1_train,y1_train)

    prediction = model.predict(x1_test)

    accuracy.append(metrics.accuracy_score(prediction,y1_test))

new_models_dataframe = pd.DataFrame(accuracy,index=classifiers)   

new_models_dataframe.columns = ['New Accuracy']  
new_models_dataframe=new_models_dataframe.merge(models_dataframe,left_index=True,right_index=True,how='left')

new_models_dataframe['Increase']=new_models_dataframe['New Accuracy']-new_models_dataframe['Accuracy']

new_models_dataframe
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz = []

accuracy1 = []

classifiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']

models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]

for i in models:

    model = i

    cv_result = cross_val_score(model,x[x.columns[:4]],x.iloc[:,-1], cv = kfold,scoring = "accuracy")

    cv_result = cv_result

    xyz.append(cv_result.mean())

    accuracy1.append(cv_result)

new_models_dataframe2 = pd.DataFrame(xyz,index=classifiers)   

new_models_dataframe2.columns = ['CV Mean']    

new_models_dataframe2
from sklearn.ensemble import VotingClassifier #for Voting Classifier
linear_svc = svm.SVC(kernel='linear',C=0.1,gamma=10,probability=True)

radial_svm = svm.SVC(kernel='rbf',C=0.1,gamma=10,probability=True)

lr = LogisticRegression(C=0.1)
ensemble_lin_rbf = VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Radial_svm', radial_svm)],voting='soft', weights=[2,1]).fit(x1_train,y1_train)

print('The accuracy for Linear and Radial SVM is:',ensemble_lin_rbf.score(x1_test,y1_test))
ensemble_lin_lr=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Logistic Regression', lr)], 

                       voting='soft', weights=[2,1]).fit(x1_train,y1_train)

print('The accuracy for Linear SVM and Logistic Regression is:',ensemble_lin_lr.score(x1_test,y1_test))



ensemble_rad_lr=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr)], 

                       voting='soft', weights=[1,2]).fit(x1_train,y1_train)

print('The accuracy for Radial SVM and Logistic Regression is:',ensemble_rad_lr.score(x1_test,y1_test))
ensemble_rad_lr_lin=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr),('Linear_svm',linear_svc)], 

                       voting='soft', weights=[2,1,3]).fit(x1_train,y1_train)

print('The ensembled model with all the 3 classifiers is:',ensemble_rad_lr_lin.score(x1_test,y1_test))