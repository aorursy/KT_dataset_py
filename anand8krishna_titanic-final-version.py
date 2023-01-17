# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# check data for NA values
train_NA = train.isna().sum()
test_NA = test.isna().sum()

pd.concat([train_NA, test_NA], axis=1, sort = False, keys = ['Train NA', 'Test NA'])
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4.5)) 
ax1 = sns.countplot(x = 'Survived', data = train)
labels = (train['Survived'].value_counts())
plt.show()
def plotcountplot(dimension):
    category_survived = sns.catplot(x=dimension,  col="Survived",
                data = train, kind="count",
                height=4, aspect=.7)

    category_survived.set_xticklabels(rotation=0, 
        horizontalalignment='right',
        fontweight='light')
    plt.tight_layout()

plotcountplot('Pclass')
plotcountplot('Sex')
plotcountplot('SibSp')
plotcountplot('Parch')
plotcountplot('Embarked')
train["Fsize"] = train["SibSp"] + train["Parch"]
#---------------------------------------------------
test["Fsize"] = test["SibSp"] + test["Parch"]
train['Title'] = train['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')
#--------------------------------------------------
test['Title'] = test['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')

category_survived = sns.catplot(x='Title',  col="Survived",
                data = train, kind="count",
                height=6, aspect=1)

category_survived.set_xticklabels(rotation=90, 
        horizontalalignment='right',
        fontweight='light')
plt.tight_layout()
train['Embarked'] = train['Embarked'].map({'S': 1,'C' : 2,'Q' : 3})
train['Sex'] = train['Sex'].map({'male': 0,'female' : 1})
train['Title'] = train['Title'].map({'Mr': 1,'Mrs': 2,'Miss': 2,'Master': 2,'Don': 3,'Rev': 3,'Dr': 3,'Mme': 2,'Ms': 2,'Major': 3,'Lady': 2,
                                     'Sir': 3,'Mlle': 2,'Col': 3,'Capt': 3,'the Countess':2,'Jonkheer': 3})
#------------------------------------------------
test['Embarked'] = test['Embarked'].map({'S': 1,'C' : 2,'Q' : 3})
test['Sex'] = test['Sex'].map({'male': 0,'female' : 1})
test['Title'] = test['Title'].map({'Mr': 1,'Mrs': 2,'Miss': 2,'Master': 2,'Don': 3,'Rev': 3,'Dr': 3,'Mme': 2,'Ms': 2,'Major': 3,'Lady': 2,
                                     'Sir': 3,'Mlle': 2,'Col': 3,'Capt': 3,'the Countess': 2,'Jonkheer': 3, 'Dona': 2})
category_survived = sns.catplot(x='Title',  col="Survived",
                data = train, kind="count",
                height=6, aspect=1)

category_survived.set_xticklabels(rotation=90, 
        horizontalalignment='right',
        fontweight='light')
plt.tight_layout()
#plotcountplot('Fsize')
train['Fsize'] = train['Fsize'].map({0:1, 1:2, 2:2, 3:2, 4:3, 5:3, 6:3, 7:3, 10:3})
#------------------------------------
test['Fsize'] = test['Fsize'].map({0:1, 1:2, 2:2, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3 , 9: 3, 10:3})
plotcountplot('Fsize')
plotcountplot('Parch')
plotcountplot('SibSp')
train['Parch'] = train['Parch'].map({0: 1,1: 2,2: 2,3: 2,4: 3,5: 3,6: 3})
train['SibSp'] = train['SibSp'].map({0: 1,1: 2,2: 2,3: 2,4: 3,5: 3,8: 3})
#-------------------------------------
test['Parch'] = test['Parch'].map({0: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3})
test['SibSp'] = test['SibSp'].map({0: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3})
plotcountplot('Parch')
plotcountplot('SibSp')
train[train["Embarked"].isnull()]
train["Embarked"].fillna(2, inplace = True)
train.info()
test[test["Fare"].isnull()]
test["Fare"].fillna(0, inplace = True)
test.info()
train = train.drop(['Name','Cabin','Ticket'], axis=1)
test = test.drop(['Name','Cabin','Ticket'], axis=1)
#===============================================================================================================================================
#---------data-cleaning-is-completed------------------------now-for-data-preparation-for-training-----------------------------------------------
#---------train1---will-contain--age-non-null-values---------test1----will-contain-age-non-null-value-rows--------------------------------------
#---------train2---will-drop-age-col-------------------------test2----will-contain--null-age-rows-----------------------------------------------
#===============================================================================================================================================
#------------train1---will-be-splited-in-70-30-train_test_split---------------------------------------------------------------------------------
#------------train2---will-be-splitted-in-70-30-train_test_split--------------------------------------------------------------------------------
#===============================================================================================================================================
#------------Ensemble-method-will-be-carried-out----for-both-train1-and-train2------------------------------------------------------------------
#------------All-binary-classifier-algorithms-will-be-used-and-then-will-be-ensembled-together--------------------------------------------------
#===============================================================================================================================================
#------------final-ensembled-model-will-predict-values-for-test1-and-test2----------------------------------------------------------------------
train1_with_age = train.dropna()
train2_wo_age = train.drop(['Age'], axis = 1)

print("train1_with_age", train1_with_age.shape)
print("trian2_wo_age", train2_wo_age.shape)

#-----------------------------------------------------------------------------------
test1_with_age = test.dropna()
test2_wo_age = test.drop(['Age'], axis = 1)

print("test1_with_age", test1_with_age.shape)
print("test2_wo_age", test2_wo_age.shape)

#==================================================================================
#===================Dropping=PassengerId=========================================
train1_with_age = train1_with_age.drop(['PassengerId'], axis = 1)
train2_wo_age = train2_wo_age.drop(['PassengerId'], axis = 1)
test1_with_age = test1_with_age.drop(['PassengerId'], axis = 1)
test2_wo_age = test2_wo_age.drop(['PassengerId'], axis = 1)

#--------------------------------------------------------------------------------
#=========Dimensionality=Reduction===============================================

#is not needed as features here are very few - -- around 10----------------------
from sklearn.model_selection import train_test_split





#train1_with_age_train = train1_with_age[['Pclass', 'Sex', 'Age','SibSp','Parch' ,'Fsize','Fare', 'Embarked', 'Title']]
#trian1_with_age_label = train1_with_age[['Survived']]
train2_wo_age_train = train2_wo_age[['Pclass', 'Sex', 'Fsize','SibSp','Parch' , 'Fare', 'Embarked', 'Title']]
train2_wo_age_label = train2_wo_age[['Survived']]

#----------------spliting train2_wo_age in 0.5 one for models, other for stacking-
train2_wo_age_train_indv_models, train2_wo_age_train_stacking, train2_wo_age_label_indv_models, train2_wo_age_label_stacking = train_test_split(train2_wo_age_train, 
                                                                                 train2_wo_age_label, 
                                                                                 test_size=0.01, random_state=42)
#-------------------------------------------------------------------------------
#X_train1, X_test1, y_train1, y_test1 = train_test_split(train1_with_age_train, trian1_with_age_label, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(train2_wo_age_train_indv_models , train2_wo_age_label_indv_models, 
                                                        test_size=0.33, random_state=42)
#X_train2_stacking, X_test2_stacking, y_train2_stacking, y_test2_stacking = train_test_split(train2_wo_age_train_stacking , 
#                                                                                            train2_wo_age_label_stacking, 
#                                                                                            test_size=0.2, random_state=42)
#===========Algorithms to be used==========================
#====1======Complement Naive Bayes=========================
#====2======Support Vector Machine=========================
#====3======K-NEAREST NEIGHBOUR (KNN)======================
#====4======Decision Tree==================================
#====5======Random Forest==================================
#====6======Logistic regression============================
#====7======Boosted Trees==================================
#====8======Stochastic Gradient Descent====================
#====9======AdaBoost=======================================
#====10=====Ensembled Model of all of above================

#========USE====ROC=;=F1=Score=;=confusion=matrix==========
#1=====naive bayes
from sklearn.naive_bayes import ComplementNB
#[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Fare', 'Embarked', 'Title']]
#X_train1_ComplementNB = X_train1
#y_train1_ComplementNB = y_train1
X_train2_ComplementNB = X_train2[['Pclass','Sex']]
y_train2_ComplementNB = y_train2
#X_test1_ComplementNB = X_test1
#y_test1_ComplementNB = y_test1
X_test2_ComplementNB = X_test2[['Pclass','Sex']]
y_test2_ComplementNB = y_test2

#Model1_1 = ComplementNB()
#Model1_1.fit(X_train1_ComplementNB, y_train1_ComplementNB)
#Score1_1 = Model1_1.score(X_test1_ComplementNB, y_test1_ComplementNB)

Model1_2 = ComplementNB()
Model1_2.fit(X_train2_ComplementNB, y_train2_ComplementNB)
Score1_2 = Model1_2.score(X_test2_ComplementNB, y_test2_ComplementNB)

#print(Score1_1)
print(Score1_2)

from sklearn.metrics import f1_score
y_true = y_test2_ComplementNB
y_pred = Model1_2.predict(X_test2_ComplementNB)
f1_2 = f1_score(y_true, y_pred, average='binary')
print(f1_2)
#2=====SVM
from sklearn.svm import SVC

#X_train1_SVC = X_train1
#y_train1_SVC = y_train1
X_train2_SVC = X_train2[['Pclass','Sex','Fsize','Title']] #sibsp parch
y_train2_SVC = y_train2
#X_test1_SVC = X_test1
#y_test1_SVC = y_test1
X_test2_SVC = X_test2[['Pclass','Sex','Fsize','Title']]   #'SibSp','Parch'
y_test2_SVC = y_test2

#Model2_1 = SVC()
#Model2_1.fit(X_train1_SVC, y_train1_SVC)
#Score2_1 = Model2_1.score(X_test1_SVC, y_test1_SVC)

Model2_2 = SVC(C=10)
Model2_2.fit(X_train2_SVC, y_train2_SVC)
Score2_2 = Model2_2.score(X_test2_SVC, y_test2_SVC)

#print(Score2_1)
print(Score2_2)

from sklearn.metrics import f1_score
y_true = y_test2_SVC
y_pred = Model2_2.predict(X_test2_SVC)
f2_2 = f1_score(y_true, y_pred, average='binary')
print(f2_2)
#3=====KNN
from sklearn.neighbors import KNeighborsClassifier

#X_train1_KNeighborsClassifier = X_train1
#y_train1_KNeighborsClassifier = y_train1
X_train2_KNeighborsClassifier = X_train2[['Pclass', 'Sex' ,'SibSp','Parch' ,'Fsize','Embarked','Fare','Title']]
y_train2_KNeighborsClassifier = y_train2
#X_test1_KNeighborsClassifier = X_test1
#y_test1_KNeighborsClassifier = y_test1
X_test2_KNeighborsClassifier = X_test2[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Embarked','Fare', 'Title']]
y_test2_KNeighborsClassifier = y_test2

#data Scaling-==============================================================================
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()

#X_train1_KNeighborsClassifier = scaler1.fit_transform(X_train1_KNeighborsClassifier)
X_train2_KNeighborsClassifier = scaler1.fit_transform(X_train2_KNeighborsClassifier)
#X_test1_KNeighborsClassifier = scaler1.fit_transform(X_test1_KNeighborsClassifier)
X_test2_KNeighborsClassifier = scaler1.fit_transform(X_test2_KNeighborsClassifier)
#========================================================================================
#Model3_1 = KNeighborsClassifier()
#Model3_1.fit(X_train1_KNeighborsClassifier, y_train1_KNeighborsClassifier)
#Score3_1 = Model3_1.score(X_test1_KNeighborsClassifier, y_test1_KNeighborsClassifier)

Model3_2 = KNeighborsClassifier(p=1, n_neighbors=37)
Model3_2.fit(X_train2_KNeighborsClassifier, y_train2_KNeighborsClassifier)
Score3_2 = Model3_2.score(X_test2_KNeighborsClassifier, y_test2_KNeighborsClassifier)

#print(Score3_1)
print(Score3_2)

from sklearn.metrics import f1_score
y_true = y_test2_KNeighborsClassifier
y_pred = Model3_2.predict(X_test2_KNeighborsClassifier)
f3_2 = f1_score(y_true, y_pred, average='binary')
print(f3_2)
#4=====DT
from sklearn.tree import DecisionTreeClassifier
#X_train1_DecisionTreeClassifier = X_train1
#y_train1_DecisionTreeClassifier = y_train1
X_train2_DecisionTreeClassifier = X_train2[['Pclass', 'Sex','Fsize','Title']]
y_train2_DecisionTreeClassifier = y_train2
#X_test1_DecisionTreeClassifier = X_test1
#y_test1_DecisionTreeClassifier = y_test1
X_test2_DecisionTreeClassifier = X_test2[['Pclass', 'Sex','Fsize' ,'Title']]
y_test2_DecisionTreeClassifier = y_test2

#Model4_1 = DecisionTreeClassifier()
#Model4_1.fit(X_train1_DecisionTreeClassifier, y_train1_DecisionTreeClassifier)
#Score4_1 = Model4_1.score(X_test1_DecisionTreeClassifier, y_test1_DecisionTreeClassifier)

Model4_2 = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5)
Model4_2.fit(X_train2_DecisionTreeClassifier, y_train2_DecisionTreeClassifier)
Score4_2 = Model4_2.score(X_test2_DecisionTreeClassifier, y_test2_DecisionTreeClassifier)

#print(Score4_1)
print(Score4_2)

from sklearn.metrics import f1_score
y_true = y_test2_DecisionTreeClassifier
y_pred = Model4_2.predict(X_test2_DecisionTreeClassifier)
f4_2 = f1_score(y_true, y_pred, average='binary')
print(f4_2)
#5=====RF                                                
from sklearn.ensemble import RandomForestClassifier

#X_train1_RandomForestClassifier = X_train1
#y_train1_RandomForestClassifier = y_train1
X_train2_RandomForestClassifier = X_train2[['Pclass', 'Sex' ,'Fsize','Title']]
y_train2_RandomForestClassifier = y_train2
#X_test1_RandomForestClassifier = X_test1
#y_test1_RandomForestClassifier = y_test1
X_test2_RandomForestClassifier = X_test2[['Pclass', 'Sex' ,'Fsize','Title']]
y_test2_RandomForestClassifier = y_test2

#Model5_1 = RandomForestClassifier()
#Model5_1.fit(X_train1_RandomForestClassifier, y_train1_RandomForestClassifier)
#Score5_1 = Model5_1.score(X_test1_RandomForestClassifier, y_test1_RandomForestClassifier)

Model5_2 = RandomForestClassifier(n_estimators=3)
Model5_2.fit(X_train2_RandomForestClassifier, y_train2_RandomForestClassifier)
Score5_2 = Model5_2.score(X_test2_RandomForestClassifier, y_test2_RandomForestClassifier)

#print(Score5_1)
print(Score5_2)

from sklearn.metrics import f1_score
y_true = y_test2_RandomForestClassifier
y_pred = Model5_2.predict(X_test2_RandomForestClassifier)
f5_2 = f1_score(y_true, y_pred, average='binary')
print(f5_2)    #used age here !!
#6=====LogReg
from sklearn.linear_model import LogisticRegression

#X_train1_LogisticRegression = X_train1
#y_train1_LogisticRegression = y_train1
X_train2_LogisticRegression = X_train2[['Pclass', 'Sex','Fsize','Title']]
y_train2_LogisticRegression = y_train2
#X_test1_LogisticRegression = X_test1
#y_test1_LogisticRegression = y_test1
X_test2_LogisticRegression = X_test2[['Pclass', 'Sex','Fsize','Title']]
y_test2_LogisticRegression = y_test2

#Model6_1 = LogisticRegression()
#Model6_1.fit(X_train1_LogisticRegression, y_train1_LogisticRegression)
#Score6_1 = Model6_1.score(X_test1_LogisticRegression, y_test1_LogisticRegression)

Model6_2 = LogisticRegression(C=0.1)
Model6_2.fit(X_train2_LogisticRegression, y_train2_LogisticRegression)
Score6_2 = Model6_2.score(X_test2_LogisticRegression, y_test2_LogisticRegression)

#print(Score6_1)
print(Score6_2)

from sklearn.metrics import f1_score
y_true = y_test2_LogisticRegression
y_pred = Model6_2.predict(X_test2_LogisticRegression)
f6_2 = f1_score(y_true, y_pred, average='binary')
print(f6_2)
#7====Boosted trees
from sklearn.ensemble import GradientBoostingClassifier

#X_train1_GradientBoostingClassifier = X_train1
#y_train1_GradientBoostingClassifier = y_train1
X_train2_GradientBoostingClassifier = X_train2[['Pclass', 'Sex','Fsize','Title']]
y_train2_GradientBoostingClassifier = y_train2
#X_test1_GradientBoostingClassifier = X_test1
#y_test1_GradientBoostingClassifier = y_test1
X_test2_GradientBoostingClassifier = X_test2[['Pclass', 'Sex','Fsize','Title']]
y_test2_GradientBoostingClassifier = y_test2

#Model7_1 = GradientBoostingClassifier()
#Model7_1.fit(X_train1_GradientBoostingClassifier, y_train1_GradientBoostingClassifier)
#Score7_1 = Model7_1.score(X_test1_GradientBoostingClassifier, y_test1_GradientBoostingClassifier)

Model7_2 = GradientBoostingClassifier(n_estimators=8)
Model7_2.fit(X_train2_GradientBoostingClassifier, y_train2_GradientBoostingClassifier)
Score7_2 = Model7_2.score(X_test2_GradientBoostingClassifier, y_test2_GradientBoostingClassifier)

#print(Score7_1)
print(Score7_2)

from sklearn.metrics import f1_score
y_true = y_test2_GradientBoostingClassifier
y_pred = Model7_2.predict(X_test2_GradientBoostingClassifier)
f7_2 = f1_score(y_true, y_pred, average='binary')
print(f7_2)

y_pred7_2 = Model7_2.predict_proba(X_test2_GradientBoostingClassifier)[:, 1]
fpr7_2, tpr7_2, _ = roc_curve(y_test2_GradientBoostingClassifier, y_pred7_2)
roc_auc7_2 = auc(fpr7_2, tpr7_2)
print(roc_auc7_2)
#8======SGD
from sklearn.linear_model import SGDClassifier

#X_train1_SGDClassifier = X_train1
#y_train1_SGDClassifier = y_train1
X_train2_SGDClassifier = X_train2[['Pclass', 'Sex' ,'Fsize','Title']]
y_train2_SGDClassifier = y_train2
#X_test1_SGDClassifier = X_test1
#y_test1_SGDClassifier = y_test1
X_test2_SGDClassifier = X_test2[['Pclass', 'Sex' ,'Fsize','Title']]
y_test2_SGDClassifier = y_test2

#data Scaling-==============================================================================
from sklearn.preprocessing import StandardScaler

scaler2 = StandardScaler()

#X_train1_SGDClassifier = scaler2.fit_transform(X_train1_SGDClassifier)
X_train2_SGDClassifier = scaler2.fit_transform(X_train2_SGDClassifier)
#X_test1_SGDClassifier = scaler2.fit_transform(X_test1_SGDClassifier)
X_test2_SGDClassifier = scaler2.fit_transform(X_test2_SGDClassifier)
#========================================================================================

#Model8_1 = SGDClassifier()
#Model8_1.fit(X_train1_SGDClassifier, y_train1_SGDClassifier)
#Score8_1 = Model8_1.score(X_test1_SGDClassifier, y_test1_SGDClassifier)

Model8_2 = SGDClassifier(random_state=42)
Model8_2.fit(X_train2_SGDClassifier, y_train2_SGDClassifier)
Score8_2 = Model8_2.score(X_test2_SGDClassifier, y_test2_SGDClassifier)

#print(Score8_1)
print(Score8_2)

from sklearn.metrics import f1_score
y_true = y_test2_SGDClassifier
y_pred = Model8_2.predict(X_test2_SGDClassifier)
f8_2 = f1_score(y_true, y_pred, average='binary')
print(f8_2)
#9=====Adaboost
from sklearn.ensemble import AdaBoostClassifier

#X_train1_AdaBoostClassifier = X_train1
#y_train1_AdaBoostClassifier = y_train1
X_train2_AdaBoostClassifier = X_train2[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Title']]
y_train2_AdaBoostClassifier = y_train2
#X_test1_AdaBoostClassifier = X_test1
#y_test1_AdaBoostClassifier = y_test1
X_test2_AdaBoostClassifier = X_test2[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Title']]
y_test2_AdaBoostClassifier = y_test2

#Model9_1 = AdaBoostClassifier()
#Model9_1.fit(X_train1_AdaBoostClassifier, y_train1_AdaBoostClassifier)
#Score9_1 = Model9_1.score(X_test1_AdaBoostClassifier, y_test1_AdaBoostClassifier)

Model9_2 = AdaBoostClassifier()
Model9_2.fit(X_train2_AdaBoostClassifier, y_train2_AdaBoostClassifier)
Score9_2 = Model9_2.score(X_test2_AdaBoostClassifier, y_test2_AdaBoostClassifier)

#print(Score9_1)
print(Score9_2)

from sklearn.metrics import f1_score
y_true = y_test2_AdaBoostClassifier
y_pred = Model9_2.predict(X_test2_AdaBoostClassifier)
f9_2 = f1_score(y_true, y_pred, average='binary')
print(f9_2)
from sklearn.metrics import roc_curve, auc

#------------------------------------------------------------
y_pred1_2 = Model1_2.predict_proba(X_test2_ComplementNB)[:, 1]
fpr1_2, tpr1_2, _ = roc_curve(y_test2_ComplementNB, y_pred1_2)
roc_auc1_2 = auc(fpr1_2, tpr1_2)
#------------------------------------------------------------
#y_pred2_2 = Model2_2.predict_proba(X_test2_SVC)[:, 1]
y_pred2_2 = Model2_2.predict(X_test2_SVC)
fpr2_2, tpr2_2, _ = roc_curve(y_test2_SVC, y_pred2_2)
roc_auc2_2 = auc(fpr2_2, tpr2_2)
#------------------------------------------------------------
y_pred3_2 = Model3_2.predict_proba(X_test2_KNeighborsClassifier)[:, 1]
fpr3_2, tpr3_2, _ = roc_curve(y_test2_KNeighborsClassifier, y_pred3_2)
roc_auc3_2 = auc(fpr3_2, tpr3_2)
#------------------------------------------------------------
y_pred4_2 = Model4_2.predict_proba(X_test2_DecisionTreeClassifier)[:, 1]
fpr4_2, tpr4_2, _ = roc_curve(y_test2_DecisionTreeClassifier, y_pred4_2)
roc_auc4_2 = auc(fpr4_2, tpr4_2)
#------------------------------------------------------------
y_pred5_2 = Model5_2.predict_proba(X_test2_RandomForestClassifier)[:, 1]
fpr5_2, tpr5_2, _ = roc_curve(y_test2_RandomForestClassifier, y_pred5_2)
roc_auc5_2 = auc(fpr5_2, tpr5_2)
#------------------------------------------------------------
y_pred6_2 = Model6_2.predict_proba(X_test2_LogisticRegression)[:, 1]
fpr6_2, tpr6_2, _ = roc_curve(y_test2_LogisticRegression, y_pred6_2)
roc_auc6_2 = auc(fpr6_2, tpr6_2)
#------------------------------------------------------------
y_pred7_2 = Model7_2.predict_proba(X_test2_GradientBoostingClassifier)[:, 1]
fpr7_2, tpr7_2, _ = roc_curve(y_test2_GradientBoostingClassifier, y_pred7_2)
roc_auc7_2 = auc(fpr7_2, tpr7_2)
#------------------------------------------------------------
#y_pred8_2 = Model8_2.predict_proba(X_test2_SGDClassifier)[:, 1]
y_pred8_2 = Model8_2.predict(X_test2_SGDClassifier)
fpr8_2, tpr8_2, _ = roc_curve(y_test2_SGDClassifier, y_pred8_2)
roc_auc8_2 = auc(fpr8_2, tpr8_2)
#------------------------------------------------------------
y_pred9_2 = Model9_2.predict_proba(X_test2_AdaBoostClassifier)[:, 1]
fpr9_2, tpr9_2, _ = roc_curve(y_test2_AdaBoostClassifier, y_pred9_2)
roc_auc9_2 = auc(fpr9_2, tpr9_2)
#------------------------------------------------------------

plt.figure(1, figsize=(11,11))
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr1_2, tpr1_2,label='CNB(area = %0.2f)' % roc_auc1_2)
plt.plot(fpr2_2, tpr2_2, label='SVC(area = %0.2f)' % roc_auc2_2)
plt.plot(fpr3_2, tpr3_2, label='KNN(area = %0.2f)' % roc_auc3_2)
plt.plot(fpr4_2, tpr4_2, label='DecisionTree(area = %0.2f)' % roc_auc4_2)
plt.plot(fpr5_2, tpr5_2, label='RF(area = %0.2f)' % roc_auc5_2)
plt.plot(fpr6_2, tpr6_2, label='LogReg(area = %0.2f)' % roc_auc6_2)
plt.plot(fpr7_2, tpr7_2, label='GradientBoostingClf(area = %0.2f)' % roc_auc7_2)
plt.plot(fpr8_2, tpr8_2, label='SGD(area = %0.2f)' % roc_auc8_2)
plt.plot(fpr9_2, tpr9_2, label='AdaBoost(area = %0.2f)' % roc_auc9_2)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

plt.show()
#10=====Ensemble==============voting==================
from sklearn.ensemble import VotingClassifier

estimators = [('CNB', Model1_2), ('Svm', Model2_2),('KNN', Model3_2), ('DT', Model4_2), ('rf', Model5_2),
              ('LR', Model6_2), ('BT', Model7_2),      #some models are dropped here
              ('ABC', Model9_2)]

              
ensemble1 = VotingClassifier(estimators)
ensemble1.fit(X_train2, y_train2)
Score_ensemble1 = ensemble1.score(X_test2, y_test2)
print(Score_ensemble1)

y_pred_ensemble = ensemble1.predict(X_test2)
fpr_ensemble, tpr_ensemble2, _ = roc_curve(y_test2, y_pred_ensemble)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble2)
print(roc_auc_ensemble)
from sklearn.preprocessing import StandardScaler
scaler3 = StandardScaler()
test2_wo_age_stacking_scaled1 = scaler3.fit_transform(train2_wo_age_train_stacking[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Fare', 
                                                                                    'Embarked', 'Title']])
test2_wo_age_stacking_scaled2 = scaler3.fit_transform(train2_wo_age_train_stacking[['Pclass', 'Sex' ,'Fsize','Title']])

data_stacking = train2_wo_age_train_stacking
data_stacking['feature1'] = Model1_2.predict_proba(train2_wo_age_train_stacking[['Pclass','Sex']])[:,1]
data_stacking['feature2'] = Model2_2.predict(train2_wo_age_train_stacking[['Pclass', 'Sex','Fsize', 'Title']])
data_stacking['feature3'] = Model3_2.predict_proba(test2_wo_age_stacking_scaled1)[:,1]
data_stacking['feature4'] = Model4_2.predict_proba(train2_wo_age_train_stacking[['Pclass', 'Sex','Fsize','Title']])[:,1]
data_stacking['feature5'] = Model5_2.predict_proba(train2_wo_age_train_stacking[['Pclass', 'Sex','Fsize','Title']])[:,1]
data_stacking['feature6'] = Model6_2.predict_proba(train2_wo_age_train_stacking[['Pclass', 'Sex','Fsize','Title']])[:,1]
data_stacking['feature7'] = Model7_2.predict_proba(train2_wo_age_train_stacking[['Pclass', 'Sex','Fsize','Title']])[:,1]
data_stacking['feature8'] = Model8_2.predict(test2_wo_age_stacking_scaled2)
data_stacking['feature9'] = Model9_2.predict_proba(train2_wo_age_train_stacking[['Pclass', 'Sex','SibSp','Parch','Fsize','Title']])[:,1]
#11======ensemble=======stacking=================================
data_stacking = data_stacking[['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']]#,'feature10]]

y_test2 = y_test2
#----------data----preparation------done----till----here--------------------------------------------------

X_train_stacking, X_test_stacking, y_train_stacking, y_test_stacking = train_test_split(data_stacking, train2_wo_age_label_stacking,
                                                                                        test_size=0.4, random_state=42)

#----------------train---test----split----------------done------------------------------------------------

from sklearn.linear_model import LogisticRegression

Stacking_Model = LogisticRegression(C=0.1)
Stacking_Model.fit(X_train_stacking, y_train_stacking)

#-----using---logreg---as---stacking---ensembled----model------------model----fitted-----on---train--set
Score_stacking = Stacking_Model.score(X_test_stacking, y_test_stacking)

print(Score_stacking)

from sklearn.metrics import f1_score
y_true = y_test_stacking
y_pred = Stacking_Model.predict(X_test_stacking)
f_stack = f1_score(y_true, y_pred, average='binary')
print(f_stack)

data_stacking['label'] = train2_wo_age_label_stacking
data_stacking.head(30)



#--------predicting values on original test class using all 10 models-----
from sklearn.preprocessing import StandardScaler
scaler3 = StandardScaler()
test2_wo_age_scaled1 = scaler3.fit_transform(test2_wo_age[['Pclass', 'Sex','SibSp','Parch' ,'Fsize','Fare', 'Embarked', 'Title']])
test2_wo_age_scaled2 = scaler3.fit_transform(test2_wo_age[['Pclass', 'Sex' ,'Fsize','Title']])
#hv to scale accordingly

#_proba(X_test2_ComplementNB)[:, 1] but in 2 and 8

predicted1 = Model1_2.predict_proba(test2_wo_age[['Pclass','Sex']])[:,1]
predicted2 = Model2_2.predict(test2_wo_age[['Pclass', 'Sex','Fsize', 'Title']])
predicted3 = Model3_2.predict_proba(test2_wo_age_scaled1)[:,1]
predicted4 = Model4_2.predict_proba(test2_wo_age[['Pclass', 'Sex','Fsize','Title']])[:,1]
predicted5 = Model5_2.predict_proba(test2_wo_age[['Pclass', 'Sex','Fsize','Title']])[:,1]
predicted6 = Model6_2.predict_proba(test2_wo_age[['Pclass', 'Sex','Fsize','Title']])[:,1]
predicted7 = Model7_2.predict_proba(test2_wo_age[['Pclass', 'Sex','Fsize','Title']])[:,1]
predicted8 = Model8_2.predict(test2_wo_age_scaled2)
predicted9 = Model9_2.predict_proba(test2_wo_age[['Pclass', 'Sex','SibSp','Parch','Fsize','Title']])[:,1]
#predicted10= ensemble1.predict(test2_wo_age)

DATA = test2_wo_age
DATA['feature-1'] = predicted1
DATA['feature-2'] = predicted2
DATA['feature-3'] = predicted3
DATA['feature-4'] = predicted4
DATA['feature-5'] = predicted5
DATA['feature-6'] = predicted6
DATA['feature-7'] = predicted7
DATA['feature-8'] = predicted8
DATA['feature-9'] = predicted9
#DATA['feature-10'] = predicted10

DATA = DATA[['feature-1','feature-2','feature-3','feature-4','feature-5','feature-6','feature-7','feature-8','feature-9']]#,'feature-10]]
finalpredictions = Model7_2.predict(test2_wo_age[['Pclass', 'Sex','Fsize','Title']])      # enter model here-----
submission['Survived'] = finalpredictions
df=pd.DataFrame(submission)
export_csv = df.to_csv('submissionfile.csv')
submission.head(20)
