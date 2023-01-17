%matplotlib inline 

import numpy as np 

import scipy as sp 

import matplotlib as mpl

import matplotlib.cm as cm 

import matplotlib.pyplot as plt

import pandas as pd 

from pandas.tools.plotting import scatter_matrix

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)

import seaborn as sns

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/train.csv")

Test=pd.read_csv("../input/test.csv")
train.tail()
train.info()
Test.info()
train["Survived"].value_counts().plot(kind="bar")

train["Survived"].value_counts()
train["Age"].hist(width=6)
train["Sex"].value_counts().plot(kind="bar")
labels="Cherbourg","Queenstown","Southampton"

sizes=[sum(train["Embarked"]=="C"),sum(train["Embarked"]=="Q"),sum(train["Embarked"]=="S")]

colors=["yellow","aqua","lime"]

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%',startangle=90)

 

plt.axis('equal')

plt.show()
def survival_stacked_bar(variable):

    Died=train[train["Survived"]==0][variable].value_counts()/len(train["Survived"]==0)

    Survived=train[train["Survived"]==1][variable].value_counts()/len(train["Survived"]==1)

    data=pd.DataFrame([Died,Survived])

    data.index=["Did not survived","Survived"]

    data.plot(kind="bar",stacked=True,title="Percentage")

    return data.head()
survival_stacked_bar("Sex")
survival_stacked_bar("Pclass")
survival_stacked_bar("Embarked")
survival_stacked_bar("SibSp")
survival_stacked_bar("Parch")
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
traintestdata=pd.concat([train,Test])

traintestdata.shape
sex_map={"male":1,"female":0}

train["Sex"]=train["Sex"].map(sex_map)

Test["Sex"]=Test["Sex"].map(sex_map)

survival_stacked_bar("Sex")
train.insert(value=train.Name.map(lambda name: name.split(",")[1].split(".")[0].strip()),loc=12,column="Title")

Test.insert(value=Test.Name.map(lambda name: name.split(",")[1].split(".")[0].strip()),loc=11,column="Title")
title_map={"Capt": "Officer",

            "Col": "Officer",

            "Major": "Officer",

            "Jonkheer": "Royalty",

            "Don": "Royalty",

            "Sir" : "Royalty",

            "Dr": "Officer",

            "Rev": "Officer",

            "the Countess":"Royalty",

            "Dona": "Royalty",

            "Mme":  "Mrs",

            "Mlle": "Miss",

            "Ms": "Mrs",

            "Mr" : "Mr",

            "Mrs" : "Mrs",

            "Miss" : "Miss",

            "Master" : "Master",

            "Lady" : "Royalty"}

train["Title"]=train.Title.map(title_map)

Test["Title"]=Test.Title.map(title_map)
for i in train.columns:

    print (i + ": "+str(sum(train[i].isnull()))+" missing values")
for i in Test.columns:

    print (i + ": "+str(sum(Test[i].isnull()))+" missing values")
train_set_1=train.groupby(["Pclass","SibSp"])

train_set_1_median=train_set_1.median()

train_set_1_median
Test_set_1=Test.groupby(["Pclass","SibSp"])

Test_set_1_median=Test_set_1.median()

Test_set_1_median
def fill_age(dataset,dataset_med):

    for x in range(len(dataset)):

        if dataset["Pclass"][x]==1:

            if dataset["SibSp"][x]==0:

                return dataset_med.loc[1,0]["Age"]

            elif dataset["SibSp"][x]==1:

                return dataset_med.loc[1,1]["Age"]

            elif dataset["SibSp"][x]==2:

                return dataset_med.loc[1,2]["Age"]

            elif dataset["SibSp"][x]==3:

                return dataset_med.loc[1,3]["Age"]

        elif dataset["Pclass"][x]==2:

            if dataset["SibSp"][x]==0:

                return dataset_med.loc[2,0]["Age"]

            elif dataset["SibSp"][x]==1:

                return dataset_med.loc[2,1]["Age"]

            elif dataset["SibSp"][x]==2:

                return dataset_med.loc[2,2]["Age"]

            elif dataset["SibSp"][x]==3:

                return dataset_med.loc[2,3]["Age"]

        elif dataset["Pclass"][x]==3:

            if dataset["SibSp"][x]==0:

                return dataset_med.loc[3,0]["Age"]

            elif dataset["SibSp"][x]==1:

                return dataset_med.loc[3,1]["Age"]

            elif dataset["SibSp"][x]==2:

                return dataset_med.loc[3,2]["Age"]

            elif dataset["SibSp"][x]==3:

                return dataset_med.loc[3,3]["Age"]

            elif dataset["SibSp"][x]==4:

                return dataset_med.loc[3,4]["Age"]

            elif dataset["SibSp"][x]==5:

                return dataset_med.loc[3,5]["Age"]

            elif dataset["SibSp"][x]==8:

                return dataset_med.loc[3]["Age"].median()  #I used the median age of Pclass=3 as a replacement as there is no median value for SibSp=8 in training dataset

train["Age"]=train["Age"].fillna(fill_age(train,train_set_1_median))

Test["Age"]=Test["Age"].fillna(fill_age(Test,Test_set_1_median))
traintestdata.Cabin.unique()
train["Cabin"]=train["Cabin"].fillna("U")

Test["Cabin"]=Test["Cabin"].fillna("U")

train["Cabin"]=train["Cabin"].map(lambda x: x[0])

Test["Cabin"]=Test["Cabin"].map(lambda x: x[0])
def new_cabin_features(dataset):

    dataset["Cabin A"]=np.where(dataset["Cabin"]=="A",1,0)

    dataset["Cabin B"]=np.where(dataset["Cabin"]=="B",1,0)

    dataset["Cabin C"]=np.where(dataset["Cabin"]=="C",1,0)

    dataset["Cabin D"]=np.where(dataset["Cabin"]=="D",1,0)

    dataset["Cabin E"]=np.where(dataset["Cabin"]=="E",1,0)

    dataset["Cabin F"]=np.where(dataset["Cabin"]=="F",1,0)

    dataset["Cabin G"]=np.where(dataset["Cabin"]=="G",1,0)

    dataset["Cabin T"]=np.where(dataset["Cabin"]=="T",1,0)  #Cabin U is when the rest of cabins are 0

    
new_cabin_features(train)

new_cabin_features(Test)
train["Embarked"]=train["Embarked"].fillna("S")
def new_embark_features(dataset):

    dataset["Embarked S"]=np.where(dataset["Embarked"]=="S",1,0)

    dataset["Embarked C"]=np.where(dataset["Embarked"]=="C",1,0)  #Embarked on Q is when the rest of embarked are 0
new_embark_features(train)

new_embark_features(Test)
Test["Fare"]=Test["Fare"].fillna(np.mean(Test["Fare"]))
title_map_2={'Mr':1, 

           'Mrs':1, 

           'Miss':1,

           'Master':2,

           'Officer':3,

           'Royalty':4}

train["Title"]=train["Title"].map(title_map_2)

Test["Title"]=Test["Title"].map(title_map_2)
train["FamilySize"]=train["SibSp"]+train["Parch"]+1

Test["FamilySize"]=Test["SibSp"]+Test["Parch"]+1
train.info()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

from sklearn.cross_validation import train_test_split

import scikitplot as skplt

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import xgboost as xgb

from sklearn.metrics import roc_curve, auc
train.drop(["Name","Ticket","PassengerId","Embarked","Cabin"],inplace=True,axis=1)

Test.drop(["Name","Ticket","Embarked","Cabin"],inplace=True,axis=1)

train.tail()
x=train.drop(["Survived"],axis=1)

y=train["Survived"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
def acc_score(model):

    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))
def confusion_matrix_model(model_used):

    cm=confusion_matrix(y_test,model_used.predict(x_test))

    col=["Predicted Dead","Predicted Survived"]

    cm=pd.DataFrame(cm)

    cm.columns=["Predicted Dead","Predicted Survived"]

    cm.index=["Actual Dead","Actual Survived"]

    cm[col]=np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)

    return cm
def importance_of_features(model):

    features = pd.DataFrame()

    features['feature'] = x_train.columns

    features['importance'] = model.feature_importances_

    features.sort_values(by=['importance'], ascending=True, inplace=True)

    features.set_index('feature', inplace=True)

    return features.plot(kind='barh', figsize=(10,10))
def aucscore(model,has_proba=True):

    if has_proba:

        fpr,tpr,thresh=skplt.metrics.roc_curve(y_test,model.predict_proba(x_test)[:,1])

    else:

        fpr,tpr,thresh=skplt.metrics.roc_curve(y_test,model.decision_function(x_test))

    x=fpr

    y=tpr

    auc= skplt.metrics.auc(x,y)

    return auc

def plt_roc_curve(name,model,has_proba=True):

    if has_proba:

        fpr,tpr,thresh=skplt.metrics.roc_curve(y_test,model.predict_proba(x_test)[:,1])

    else:

        fpr,tpr,thresh=skplt.metrics.roc_curve(y_test,model.decision_function(x_test))

    x=fpr

    y=tpr

    auc= skplt.metrics.auc(x,y)

    plt.plot(x,y,label='ROC curve for %s (AUC = %0.2f)' % (name, auc))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim((0,1))

    plt.ylim((0,1))

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="lower right")

    plt.show()
log_reg=LogisticRegression()

log_reg.fit(x_train,y_train)



print("Accuracy: " + str(acc_score(log_reg)))

confusion_matrix_model(log_reg)



#skplt.metrics.plot_confusion_matrix(y_test, log_reg.predict(x_test),normalize=True,figsize=(10,10))
plt_roc_curve("Logistic Regression",log_reg,has_proba=True)
lda = LinearDiscriminantAnalysis()

lda.fit(x_train,y_train)

ldaA = lda.transform(x_train)

print("Accuracy: " + str(acc_score(lda)))

confusion_matrix_model(lda)
plt_roc_curve("LDA",lda,has_proba=True)
qda = QuadraticDiscriminantAnalysis()

qda.fit(x_train,y_train)

print("Accuracy: " + str(acc_score(qda)))

confusion_matrix_model(qda)
plt_roc_curve("QDA",qda,has_proba=True)
SVC_rbf=SVC(kernel="rbf")

SVC_rbf.fit(x_train,y_train)



print("Accuracy: " + str(acc_score(SVC_rbf)))

confusion_matrix_model(SVC_rbf)
plt_roc_curve("RBF SVM",SVC_rbf,has_proba=False)
SVC_lin=SVC(kernel="linear")

SVC_lin.fit(x_train,y_train)



print("Accuracy: " + str(acc_score(SVC_lin)))

confusion_matrix_model(SVC_lin)
plt_roc_curve("Linear SVM",SVC_lin,has_proba=False)
KNN=KNeighborsClassifier(n_neighbors=5)

KNN.fit(x_train,y_train)



print("Accuracy: " + str(acc_score(KNN)))

confusion_matrix_model(KNN)
plt_roc_curve("KNN (5)",KNN,has_proba=True)
Dec_tree=DecisionTreeClassifier(max_depth=4,random_state=5)

Dec_tree.fit(x_train,y_train)



print("Accuracy: " + str(acc_score(Dec_tree)))

confusion_matrix_model(Dec_tree)



#skplt.metrics.plot_confusion_matrix(y_test, Dec_tree.predict(x_test),normalize=True,figsize=(6,6),text_fontsize='small')
plt_roc_curve("Decision Tree",Dec_tree,has_proba=True)
importance_of_features(Dec_tree)
ranfor = RandomForestClassifier(n_estimators=50, max_features='sqrt',max_depth=6,random_state=10)

ranfor = ranfor.fit(x_train,y_train)

print("Accuracy: " + str(acc_score(ranfor)))

confusion_matrix_model(ranfor)

plt_roc_curve("Random Forest",ranfor,has_proba=True)
importance_of_features(ranfor)
xgclass = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01).fit(x_train, y_train)

print("Accuracy: " + str(acc_score(xgclass)))

confusion_matrix_model(xgclass)
plt_roc_curve("XGBoosting",xgclass,has_proba=True)
importance_of_features(xgclass)
Classifiers=["Logistic Regression","Linear Discriminant Analysis","Quadratic Discriminant Analysis","Support Vector Machine (RBF)","Support Vector Machine (Linear)","K-Nearest Neighbours","Decision Tree","Random Forest","XGBoost"]

Acc=[acc_score(x) for x in [log_reg,lda,qda,SVC_rbf,SVC_lin,KNN,Dec_tree,ranfor,xgclass]]

auc_scores_prob=[aucscore(x,has_proba=True) for x in [log_reg,lda,qda,KNN,Dec_tree,ranfor,xgclass]]

auc_scores_noprob=[aucscore(x,has_proba=False) for x in [SVC_rbf,SVC_lin,]]

auc_scores=auc_scores_prob[:3] + auc_scores_noprob + auc_scores_prob[3:]

cols=["Classifier","Accuracy","AUC"]

results = pd.DataFrame(columns=cols)

results["Classifier"]=Classifiers

results["Accuracy"]=Acc

results["AUC"]=auc_scores

results
pred_test=ranfor.predict(Test.drop("PassengerId",axis=1).copy())

submission=pd.DataFrame({"PassengerId": Test["PassengerId"], "Survived": pred_test})

submission.to_csv("submission.csv",index=False)