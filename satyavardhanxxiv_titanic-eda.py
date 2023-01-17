import os
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
dat=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
dat.head()
dat['Sex']=[1 if i =="male" else 0 for i in dat["Sex"]]
dat.isnull().sum()
dat.info()
dat['Embarked'].value_counts()
dat[dat["Embarked"].isnull()]
dat.boxplot(column="Fare",by = "Embarked")
plt.show()
#Since Embarked has Fare closer to median of C
dat["Embarked"]=dat["Embarked"].fillna("C")
dat.shape
plt.style.use("seaborn-whitegrid")
sns.factorplot(x = "Sex",y="Age",data = dat, kind="box")
plt.show()
dat['Age'] = dat['Age'].fillna(dat.groupby('Pclass')['Age'].transform('median'))
dat.groupby('Pclass')['Age'].agg(np.sum)
dat.groupby('Pclass')['Age'].transform('median')
dat["Age"].describe()
dat.head()
import re
dat['Title'] = dat['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())
dat["Title"].unique()
dat['Title'] = dat['Title'].replace(['Capt.', 'Col.','Don.', 'Dr.', 'Major.', 'Rev.', 'Jonkheer.', 'Dona.'], 'Rare.')
dat['Title'] = dat['Title'].replace(['Countess.', 'Lady.', 'Sir.'], 'Royal.')
dat['Title'] = dat['Title'].replace('Mlle.', 'Miss.')
dat['Title'] = dat['Title'].replace('Ms.', 'Miss.')
dat['Title'] = dat['Title'].replace('Mme.', 'Mrs.')
dat.groupby(["Title","Survived"])["Survived"].agg({np.size})
dat["Title"].unique().tolist()
#Title Mapping
title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Royal.": 5, "Rare.": 6}
dat["Title"]=dat["Title"].map(title_mapping)
dat['fam']=dat['SibSp']+dat["Parch"]+1
dat=dat.drop(["SibSp","Parch","PassengerId","Name","Cabin","Ticket"], axis=1)
dat.head()
X=dat.drop(["Survived",], axis=1)
y=dat["Survived"]
X.head()
y.head()
X=pd.get_dummies(X)
X.head()
X.info()
#The following function compares train dataset columns with 'DV' column
def bar_chart(feature):
    survived = dat[dat['Survived']==1][feature].value_counts()
    dead = dat[dat['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
dat['Age'].hist(bins=40,color='salmon')
plt.title("AGE",size=20)
bar_chart('Title')
bar_chart('fam')
bar_chart('Embarked')
import sklearn.model_selection as model_selection
X_train,X_test, y_train, y_test=model_selection.train_test_split(X,y, test_size=0.2, random_state=200)
#Accuracy score without hyperparameter tuning
import sklearn.metrics as metrics
def fit_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_predicted)
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

dt_classifier= tree.DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()

dt_accuracy= fit_evaluate_model(dt_classifier, X_train, y_train, X_test, y_test)
rf_accuracy = fit_evaluate_model(rf_classifier, X_train, y_train, X_test, y_test)
gb_accuracy = fit_evaluate_model(gb_classifier, X_train, y_train, X_test, y_test)
print("Decision Tree : ",dt_accuracy)
print("Random Forest : ",rf_accuracy)
print("GradientBoosting : ",gb_accuracy)
#Now accuracy score with hyperparameter tuning

clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6]})
mod.fit(X_train,y_train)
mod.best_estimator_
mod.best_params_
#Finalizing max_depth as 3
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
clf.fit(X_train,y_train)
# Confusion matrix ( ACTUAL LABLES, PREDICTED LABLES)
metrics.confusion_matrix( y_test, clf.predict(X_test))
mod1=metrics.accuracy_score(y_test, clf.predict(X_test))
mod1
#Random Forest
rf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=400)
rf.fit(X_train,y_train)
rf.oob_score_
#Getting the best n_estimators
for w in range(10,150,10):
    rf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=400)
    rf.fit(X_train,y_train)
    oob=rf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')
#Finalizing n_estimator as 70
rf=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=400)
rf.fit(X_train,y_train)
rf.oob_score_
rf.feature_importances_
imp_feat=pd.Series(rf.feature_importances_,index=X.columns.tolist())
imp_feat.sort_values(ascending=False)
mod2=metrics.accuracy_score(y_test, rf.predict(X_test))
mod2
#Gradient Boosting
gb=GradientBoostingClassifier(n_estimators=80,random_state=400, max_depth=2)
gb.fit(X_train,y_train)
#For n_estimator
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(gb,param_grid={'n_estimators':[20,40,60,80,100,120,140,160,180,200]})
mod.fit(X_train,y_train)
mod.best_estimator_
mod.best_params_
gb=GradientBoostingClassifier(n_estimators=140,random_state=400, max_depth=2)
gb.fit(X_train,y_train)
gb.feature_importances_
feature_imp=pd.Series(gb.feature_importances_,index=X.columns)
feature_imp.sort_values(ascending=False)
#For depth
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6,7,8]})
mod.fit(X_train,y_train)
mod.best_estimator_
#Finalizing max_depth=2 and n_estimators=140
gb=GradientBoostingClassifier(n_estimators=140,random_state=400, max_depth=2)
gb.fit(X_train,y_train)
mod3=metrics.accuracy_score(y_test,clf.predict(X_test))
mod3
print("Decision Tree: ",mod1)
print("Random Forest: ",mod2)
print("Gradient Boosting: ",mod3)
#Hence here, Decision Tree is most accurate for predecting for this dataset