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
#Read the data and remain the columns there
titanic = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic.head()
#Check the proportion of dead people
print('Pclass: ')
A = titanic.groupby(['Pclass'])['Survived'].value_counts(normalize=True)
for i in range(1,4):
    print((i, A[i][0]))
print('Parch: ')
B = titanic.groupby(['Parch'])['Survived'].value_counts(normalize=True)
for i in range(0,7):
    print((i, B[i][0]))
print('SibSp: ')
C = titanic.groupby(['SibSp'])['Survived'].value_counts(normalize=True)
for i in range(0,6):
    print((i, C[i][0]))
print('Embarked: ')
D = titanic.groupby(['Embarked'])['Survived'].value_counts(normalize=True)
for character in ['C','Q','S']:
    print((character, D[character][0]))
print('Age: ')
E = titanic.groupby(['Sex'])['Survived'].value_counts(normalize=True)
for name in ['male','female']:
    print((name, E[name][0]))
# Import all of the libraries
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import itertools
import scipy
from scipy.cluster import hierarchy 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import metrics
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from xgboost import XGBClassifier
titanic['Title'] = titanic['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
for i in range(0,len(titanic)):
    if titanic['Title'][i] != 'Mr' and titanic['Title'][i] != 'Miss' and titanic['Title'][i] != 'Mrs' and titanic['Title'][i] != 'Master':
        titanic['Title'].replace(titanic['Title'][i], 'Others', inplace=True)
    else: continue
titanic['Title'].value_counts()
bins_age = np.linspace(min(titanic["Age"]), max(titanic["Age"]), 6)
group_names_age = ['Age1','Age2','Age3','Age4','Age5']
bins_fare = np.linspace(min(titanic["Fare"]), max(titanic["Fare"]), 6)
group_names_fare = ['Fare1','Fare2','Fare3','Fare4','Fare5']

titanic['Sex'].replace(to_replace=['female','male'],value=[0,1],inplace=True)
titanic['Embarked'].replace(to_replace=['C','Q','S'],value=[0,1,2],inplace=True)
titanic['Age-binned'] = pd.cut(titanic['Age'], bins_age, labels=group_names_age, include_lowest=True )
titanic['Fare-binned'] = pd.cut(titanic['Fare'], bins_fare, labels=group_names_fare, include_lowest=True)

titanic.head()
TA = titanic.groupby(['Title'])['Age'].mean()
for name in ['Mr','Miss','Mrs','Master','Others']:
    for i in range(0,len(titanic)):
        if titanic['Title'][i] == name:
            if titanic['Age'][i] == np.nan:
                titanic['Age'].replace(titanic['Age'][i], TA[name])
            else: continue
        else: continue
titanic['Age'].isna().value_counts()
dummy_age = pd.get_dummies(titanic['Age-binned'])
dummy_fare = pd.get_dummies(titanic['Fare-binned'])
dummy_embarked = pd.get_dummies(titanic['Embarked'])
titanic = pd.concat([titanic, dummy_title, dummy_age, dummy_fare, dummy_embarked], axis=1)
titanic.head()
x_data2 = titanic[['Pclass','Parch','SibSp','Sex','Embarked','Age1','Age2','Age3','Age4','Age5','Fare1','Fare2','Fare3','Fare4','Fare5']]
y_data = titanic['Survived']
x_data = x_data2.fillna(2)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict 
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
print(x_data.shape)
test_titanic = pd.read_csv('/kaggle/input/titanic/test.csv')
#create the title part
test_titanic['Title'] = test_titanic['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
for i in range(0,len(test_titanic)):
    if test_titanic['Title'][i] != 'Mr' and test_titanic['Title'][i] != 'Miss' and test_titanic['Title'][i] != 'Mrs' and test_titanic['Title'][i] != 'Master':
        test_titanic['Title'].replace(test_titanic['Title'][i], 'Others', inplace=True)
    else: continue
test_titanic['Title'].value_counts()

#make the binning
bins_age_test = np.linspace(min(test_titanic["Age"]), max(test_titanic["Age"]), 6)
group_names_age = ['Age1','Age2','Age3','Age4','Age5']
bins_fare_test = np.linspace(min(test_titanic["Fare"]), max(test_titanic["Fare"]), 6)
group_names_fare = ['Fare1','Fare2','Fare3','Fare4','Fare5']

#Settings in the numbers
test_titanic['Sex'].replace(to_replace=['female','male'],value=[0,1],inplace=True)
test_titanic['Embarked'].replace(to_replace=['C','Q','S'],value=[0,1,2],inplace=True)
test_titanic['Age-binned'] = pd.cut(test_titanic['Age'], bins_age_test, labels=group_names_age, include_lowest=True )
test_titanic['Fare-binned'] = pd.cut(test_titanic['Fare'], bins_fare_test, labels=group_names_fare, include_lowest=True)

#Dummies
dummy_title_test = pd.get_dummies(test_titanic['Title'])
dummy_age_test = pd.get_dummies(test_titanic['Age-binned'])
dummy_fare_test = pd.get_dummies(test_titanic['Fare-binned'])
test_titanic = pd.concat([test_titanic, dummy_title_test, dummy_age_test, dummy_fare_test], axis=1)

test_titanic.head()
x_datatest2 = titanic[['Pclass','Parch','SibSp','Sex','Embarked','Age1','Age2','Age3','Age4','Age5','Fare1','Fare2','Fare3','Fare4','Fare5']]
x_datatest = x_datatest2.fillna(2)
x_datatest.isna().value_counts()
SvM = svm.SVC(kernel='rbf')
yhat_svm = cross_val_score(SvM, x_data, y_data, cv=10).mean()
SvM.fit(x_data, y_data)
yhat_svm2 = SvM.predict(x_datatest)
yhat_svm3 = SvM.predict(x_test)
yhat_svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
classification_report(y_test, yhat_svm3)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # it shows the default parameters
drugTree.fit(x_train,y_train)
predTree = drugTree.predict(x_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
predtree2 = drugTree.predict(x_datatest)
predtree2[0]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_hat_linear= regr.predict(x_datatest)
y_hat_linear2 = []
for i in range(0,len(x_datatest)):
    if y_hat_linear[i] < 0.62:
        y_hat_linear2.append(0)
    else: y_hat_linear2.append(1)
y_hat_linear2[0:5]
from sklearn import metrics
from sklearn.ensemble import VotingClassifier 
a1 = LogisticRegression(C=0.01, solver='liblinear')
a2 = svm.SVC(kernel='rbf', probability = True) 
a3 = DecisionTreeClassifier(criterion="entropy", max_depth = 4) 
a4 = KNeighborsClassifier(n_neighbors = 7)
a5 = XGBClassifier()
a6 = RandomForestClassifier()
a7 = GradientBoostingClassifier()
a9 = ensemble.AdaBoostClassifier()
vote_est = [('LR',a1),('SVM',a2),('DecisionTree',a3),('KNN',a4),('XGB',a5),('Forest',a6),('GBC',a7),('ada',a9)]
vot_soft = ensemble.VotingClassifier(estimators = vote_est, voting ='soft') 
vot_soft.fit(x_train,y_train)
y_pred = vot_soft.predict(x_test) 
  
# using accuracy_score 
score = metrics.accuracy_score(y_test, y_pred) 
print("Soft Voting Score % d" % score) 
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
y_predtest = vot_soft.predict(x_datatest)
y_predtest[0]
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
numerical_cols = ['Pclass','Parch','SibSp','female','male','C','Q','S','Kids','Adolescent','Working-Age','Middle-Age','Elderly','Lowest','Low','Medium','High','Highest']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols)])
    
a1 = LogisticRegression(C=0.01, solver='liblinear')
a2 = SVC(kernel='rbf', probability = True) 
a3 = DecisionTreeClassifier(criterion="entropy", max_depth = 4) 
a4 = KNeighborsClassifier(n_neighbors = 4)
pipe=Pipeline(steps=[('preprocessor', preprocessor),('voting',VotingClassifier([
        ('a1', a1), ('a2', a2), ('a3', a3), ('a4',a4)]))
])
Rcross = cross_val_score(pipe, x_data, y_data, cv=15).mean()
Rcross
pipe.fit(x_train,y_train)
yhat_cross = pipe.predict(x_datatest)
yhat_cross[0:5]
# example DataFrame to write to CSV
df_list_cross = []
for i in range(0,418):
    df_list_cross.append([test_titanic['PassengerId'][i],y_predtest[i]])
df_cross = pd.DataFrame(df_list_cross,columns=['PassengerId', 'Survived'])

# set an index to avoid a 'blank' row number column and write to a file named 'submission.csv'
df_cross.set_index('PassengerId').to_csv('submission_crossvote_votingY.csv')
import matplotlib.gridspec as gridspec
a1 = LogisticRegression(C=0.01, solver='liblinear')
a2 = SVC(kernel='rbf', probability = True) 
a3 = DecisionTreeClassifier(criterion="entropy", max_depth = 4) 
a4 = KNeighborsClassifier(n_neighbors = 4)
a5 = XGBClassifier()
a6 = RandomForestClassifier()
a7 = GradientBoostingClassifier()
a8 = NuSVC(probability=True)
a9 = ensemble.AdaBoostClassifier()
sclf = StackingClassifier(classifiers=[a2, a3, a4, a5, a6, a7, a8, a9], 
                          meta_classifier=a1)

label = ['SVC', 'DecisionTree', 'KNN','XGB','Forest','GBC','ExTree','Stacking Classifier']
clf_list = [a2, a3, a4, a5 ,a6, a7,a8,sclf]
    
clf_cv_mean = []
clf_cv_std = []
for clf in clf_list:
        
    scores = cross_val_score(clf, x_data, y_data, cv=4, scoring='accuracy')
    print ("Accuracy: " ,scores.mean())
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(x_data, y_data)
y_Stack = clf.predict(x_datatest)
y_Stack[0:5]