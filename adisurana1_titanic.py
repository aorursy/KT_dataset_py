import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.head()
test_df = pd.read_csv("../input/test.csv")
test_df.head()
train_df.info()
# Convert some columns into categorical data

train_df["Survived"] = train_df["Survived"].astype('category')

train_df["Pclass"] = train_df["Pclass"].astype('category')

test_df["Pclass"] = test_df["Pclass"].astype('category')

train_df.describe()
train_df[train_df['Sex']=='male']['Sex'].count()
train_df[train_df['Sex']=='female']['Sex'].count()
train_df["Sex"].value_counts().plot.pie(figsize=(4, 4),

                                     autopct='%.2f',

                                     title="Percentage of Male and Female passengers",

                                     fontsize = 10)
train_df[train_df['Survived']==1]['Survived'].count()
train_df['Survived'].value_counts().plot.pie(figsize=(4, 4),

                                            autopct='%.2f',

                                            title="Percentage of survivors",

                                            fontsize = 10)
train_df[(train_df['Sex']=='male') & (train_df['Survived']==1)]['Name'].count()
train_df[(train_df['Sex']=='female') & (train_df['Survived']==1)]['Name'].count()
sns.countplot(x="Survived", hue="Sex", data=train_df)
train_df['Pclass'].value_counts().plot.pie(figsize=(4, 4),

                                            autopct='%.2f',

                                            title="Percentage of Pclass passengers",

                                            fontsize = 10)
sns.countplot(x="Survived", hue="Pclass", data=train_df)
print('\033[1m'+"Checking if train_df contains any null value:-"+'\033[0m')

print(train_df.isnull().sum())

print('\n')

print('\033[1m'+"Checking if test_df contains any null value:-"+'\033[0m')

print(test_df.isnull().sum())
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
drop_col=['Sex','Name','Cabin','Ticket','Embarked']

train_df.drop(drop_col, axis=1, inplace=True)

test_df.drop(drop_col, axis=1, inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
from sklearn.model_selection import train_test_split #split the dat in test and train sets

from sklearn.model_selection import cross_val_score #score evaluation with cross validation

from sklearn.model_selection import cross_val_predict #prediction with cross validation

from sklearn.metrics import confusion_matrix #for confusion matrix (metric of succes)

from sklearn.model_selection import KFold #for K-fold cross validation

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
all_features = train_df.drop("Survived",axis=1)

targeted_feature = train_df["Survived"]
X_train,X_test,y_train,y_test = train_test_split(all_features,

                                                 targeted_feature,

                                                 test_size=0.3,random_state=42)
train_X = train_df.drop("Survived", axis=1)

train_Y=train_df["Survived"]

test_X  = test_df

train_X.shape, train_Y.shape, test_X.shape
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

n_estim=range(100,1000,100)

#This is the grid

param_grid = {"n_estimators" :n_estim}





model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_rf.fit(train_X,train_Y)

model_rf.best_score_
print('Accuracy = ', round((model_rf.best_score_)*100,2))
from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(criterion='gini', 

                             min_samples_split=2,min_samples_leaf=1,

                             max_features='auto')



model.fit(X_train,y_train)



prediction_tree=model.predict(X_test)



print('Accuracy =',round(accuracy_score(prediction_tree,y_test)*100,2))



result_tree=cross_val_score(model,all_features,targeted_feature,cv=10,scoring='accuracy')



print('Cross validated score =',round(result_tree.mean()*100,2))



y_pred = cross_val_predict(model,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")



plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 7)



model.fit(X_train,y_train)



prediction_knn=model.predict(X_test)



print('Accuracy =',round(accuracy_score(prediction_knn,y_test)*100,2))



result_knn=cross_val_score(model,all_features,targeted_feature,cv=10,scoring='accuracy')



print('Cross validated score =',round(result_knn.mean()*100,2))



y_pred = cross_val_predict(model,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")



plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.metrics import accuracy_score  #for accuracy_score



model = LogisticRegression()



model.fit(X_train,y_train)



prediction_lr=model.predict(X_test)



print('Accuracy =',round(accuracy_score(prediction_lr,y_test)*100,2))



result_lr=cross_val_score(model,all_features,targeted_feature,cv=10,scoring='accuracy')



print('Cross validated score =',round(result_lr.mean()*100,2))



y_pred = cross_val_predict(model,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")



plt.title('Confusion matrix', y=1.05, size=15)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300,400],

              'learning_rate': [0.1, 0.05, 0.01,0.001],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.2,0.1] 

              }



modelf = GridSearchCV(model,param_grid = param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



modelf.fit(X_train,y_train)





# Best Estimator

modelf.best_estimator_

print('Accuracy = ', round((modelf.best_score_)*100,2))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model= LinearDiscriminantAnalysis()

model.fit(X_train,y_train)

prediction_lda=model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_lda,y_test)*100,2))



result_lda=cross_val_score(model,all_features,targeted_feature,cv=10,scoring='accuracy')



print('The cross validated score',round(result_lda.mean()*100,2))



y_pred = cross_val_predict(model,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")



plt.title('Confusion matrix', y=1.05, size=15)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score 

model_rf = RandomForestClassifier(criterion='gini', n_estimators=100,

                             min_samples_split=2,min_samples_leaf=1,

                             max_features='auto',oob_score=True,

                             random_state=1,n_jobs=-1)



model_rf.fit(X_train,y_train)



prediction_rm= model_rf.predict(X_test)



print('Accuracy =',round(accuracy_score(prediction_rm,y_test)*100,2))



result_rm=cross_val_score(model_rf,all_features,targeted_feature,cv=10,scoring='accuracy')



print('Cross validated score =',round(result_rm.mean()*100,2))



y_pred = cross_val_predict(model_rf,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")



plt.title('Confusion matrix', y=1.05, size=15)
from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier()

model.fit(X_train,y_train)

prediction_gbc=model.predict(X_test)



print('Accuracy',round(accuracy_score(prediction_gbc,y_test)*100,2))



result_gbc=cross_val_score(model,all_features,targeted_feature,cv=10,scoring='accuracy')



print('The cross validated score',round(result_gbc.mean()*100,2))



y_pred = cross_val_predict(model,all_features,targeted_feature,cv=10)



sns.heatmap(confusion_matrix(targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="cool")

plt.title('Confusion matrix', y=1.05, size=15)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(test_X)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(acc_random_forest)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_rf})
submission.head()
submission.to_csv('submission.csv', index=False)