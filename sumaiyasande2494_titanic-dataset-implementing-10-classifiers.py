import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.info()

train.shape

test.shape
#check whether the data has missing values 

train.isnull().values.any()

#print how many missing values data has

train.isnull().values.sum()
#We can plot the missing values with following library

import missingno as msno

msno.matrix(train)
#drop the two rows in which there are two missing values in the variable `Embarked'

train['Embarked'].isnull()

train[train['Embarked'].isna()]

train_dropped = train.drop([61,829])

train_dropped.shape
#Drop the variable 'cabin' since it does not give any meaningful information (just for simplification)

train_dropped['Cabin']
train_dropped1 = train_dropped.drop('Cabin',axis=1)

train_dropped1.shape
train_dropped1['Age'].fillna(train_dropped1['Age'].mean(),inplace=True)
train_dropped1.isnull().values.any()
test.info()
test['Age'].fillna(test['Age'].mean(),inplace=True)
test = test.drop('Cabin',axis=1)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.info()
train = train_dropped1
train.shape
test.shape
train.head()
attributes = ['Survived','Sex','Age','SibSp','Parch','Fare','Embarked']

train[attributes].hist(figsize=(10,10))
train[['Survived','Sex']].value_counts()
pd.crosstab(train.Survived,train.Sex)
import seaborn as sns

#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)
sns.barplot(x="Pclass", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)
sns.barplot(x="Embarked", y="Survived", data=train)
sns.barplot(x="SibSp", y="Survived", data=train)
age_count = pd.cut(train.Age,bins=[0,2,17,50,80],labels=['Toddler/Baby','Child','Adult','Elderly'])

train.insert(10,'Age_Group',age_count)

sns.barplot(x="Age_Group", y="Survived", data=train)
train = train.drop('Age_Group',axis=1)
#check whether the Age_Group variable is removed from dataset.

train.info()
#Converting Categorical variables into numeric

train['Sex'] = train['Sex'].map({'male':1, 'female':0})

test['Sex'] = test['Sex'].map({'male':1, 'female':0})

train['Embarked'] = train['Embarked'].map({'Q':2, 'S':1, 'C':0})

test['Embarked'] = test['Embarked'].map({'Q':2, 'S':1, 'C':0})
X_train = train.drop(["Name", "Survived", "PassengerId","Ticket"], axis=1)

Y_train = train["Survived"]

X_test  = test.drop(['Name',"PassengerId","Ticket"], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_train.head()
X_test.head()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

import xgboost as xgb

!pip install pygam

from pygam import LogisticGAM

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
# Logistic Regression

logreg = LogisticRegression(max_iter=200)

logreg.fit(X_train,Y_train)

logreg_Y_pred=logreg.predict(X_test)

logreg_probs = logreg.predict_proba(X_train)[:,1] #predicted probabilities on training data

logreg_accuracy = logreg.score(X_train, Y_train) #accuracy on training data

logreg_accuracy
#Ridge Classifier

#Method 1:

ridge = LogisticRegression(penalty="l2",max_iter=200)

ridge.fit(X_train,Y_train)

ridge_Y_pred=ridge.predict(X_test)

ridge_probs = ridge.predict_proba(X_train)[:,1] #predicted probabilities on training data

ridge_accuracy = ridge.score(X_train, Y_train) #accuracy on training data

ridge_accuracy



#Method 2:

sgd = SGDClassifier(penalty="l2",random_state=7)

sgd.fit(X_train,Y_train)

sgd_Y_pred=sgd.predict(X_test)

sgd_accuracy = sgd.score(X_train, Y_train)

sgd_accuracy
ridge_accuracy
# Logistic GAM

gam = LogisticGAM()

gam.fit(X_train,Y_train)

gam_Y_pred=gam.predict(X_test)

gam_Y_pred = gam_Y_pred*1

gam_probs = gam.predict_proba(X_train) #predicted probabilities on training data

gam_accuracy = gam.accuracy(X_train, Y_train) #accuracy on training data

gam_accuracy
# Support Vector Machine

svm = SVC(probability=True)

svm.fit(X_train, Y_train)

svm_Y_pred = svm.predict(X_test)

svm_probs = svm.predict_proba(X_train)[:,1] #predicted probabilities on training data

svm_accuracy = svm.score(X_train, Y_train) #accuracy on training data

svm_accuracy
# k-nearest neighbor

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, Y_train)

knn_Y_pred = knn.predict(X_test)

knn_probs = knn.predict_proba(X_train)[:,1] #predicted probabilities on training data

knn_accuracy = knn.score(X_train,Y_train) #accuracy on training data

knn_accuracy
# Gaussian Naive Bayes

nb = GaussianNB()

nb.fit(X_train, Y_train)

nb_Y_pred = nb.predict(X_test)

nb_probs = nb.predict_proba(X_train)[:,1] #predicted probabilities on training data

nb_accuracy = nb.score(X_train, Y_train) #accuracy on training data

nb_accuracy
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

decision_tree_Y_pred = decision_tree.predict(X_test)

decision_tree_probs = decision_tree.predict_proba(X_train)[:,1] #predicted probabilities on training data

decision_tree_accuracy = decision_tree.score(X_train, Y_train) #accuracy on training data

decision_tree_accuracy
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest_Y_pred = random_forest.predict(X_test)

random_forest_probs = random_forest.predict_proba(X_train)[:,1] #predicted probabilities on training data

random_forest.score(X_train, Y_train)

random_forest_accuracy = random_forest.score(X_train, Y_train) #accuracy on training data

random_forest_accuracy
#XGBOOST

dtrain = xgb.DMatrix(data=X_train,label=Y_train)

dtest = xgb.DMatrix(X_test)

param = {'max_depth':11, 'eta':0.9, 'objective':'binary:logistic' }

num_round = 2

model = xgb.train(param, dtrain, num_round)

# make prediction for test data

preds = model.predict(dtest)

xgboost_Y_pred = [round(value) for value in preds]

# make predictions for training data

preds = model.predict(dtrain)

Y_pred = [round(value) for value in preds]

xgboost_accuracy = accuracy_score(Y_train,Y_pred) #accuracy on training data

xgboost_probs = preds #predicted probabilities on training data

xgboost_accuracy

#MLP

mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1,max_iter=1000)

mlp.fit(X_train, Y_train)

mlp_Y_pred = mlp.predict(X_test)

mlp_probs = mlp.predict_proba(X_train)[:,1] #predicted probabilities on training data

mlp_accuracy = mlp.score(X_train, Y_train) #accuracy on training data

mlp_accuracy
all_predicted = pd.DataFrame({"Logistic": logreg_Y_pred,"Ridge":ridge_Y_pred,"GAM": gam_Y_pred,"SVM": svm_Y_pred,"KNN": knn_Y_pred,"Naive_Bayes":nb_Y_pred,"Tree":decision_tree_Y_pred,"RF": random_forest_Y_pred,"XGBOOST": xgboost_Y_pred,"MLP":mlp_Y_pred})



all_predicted
# This creates an object final_pred which gives the final predicted class from all classifiers using majority votes method.



final_pred = list(range(418))

for i in range(418):

    if sum(all_predicted.loc[i])< 5:

        final_pred[i] = 0

    else : 

        final_pred[i] = 1
# submission file for a single classifier 

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": random_forest_Y_pred})

submission.to_csv('submission.csv', index=False)



# submission file for ensemble of all classifiers

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_pred})

submission.to_csv('submission.csv', index=False)
from sklearn.metrics import roc_auc_score, roc_curve



classifiers = ["Logistic","Ridge","GAM","SVM","KNN","NB","Decision Tree","RF","xgboost","MLP"]



predicted_probs = pd.DataFrame({"Logistic": logreg_probs,"Ridge":ridge_probs,"GAM": gam_probs,"SVM": svm_probs,"KNN": knn_probs,"Naive_Bayes":nb_probs,"Tree":decision_tree_probs,"RF": random_forest_probs,"XGBOOST": xgboost_probs,"MLP":mlp_probs})



predicted_probs
#Draw ROC Curves

plt.figure(figsize=(10,7))

plt.plot([0,1], [0,1], linestyle='--')

for i in range(len(classifiers)):

    fpr , tpr, thresholds = roc_curve(Y_train, predicted_probs.iloc[:,i])

    plt.plot(fpr, tpr, label= classifiers[i])



plt.legend()

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.title('Receiver Operating Characteristic')

plt.show()   
#Obtain AUC values for all the cassifiers for the training data

auc = list(range(len(classifiers)))

for i in range(len(classifiers)):

    auc[i] = roc_auc_score(Y_train, predicted_probs.iloc[:,i])

  

pd.DataFrame({"Classifier":classifiers,"AUC":auc})
accuracy = [logreg_accuracy,ridge_accuracy,gam_accuracy,svm_accuracy,knn_accuracy,nb_accuracy,decision_tree_accuracy,random_forest_accuracy,xgboost_accuracy,mlp_accuracy]



pd.DataFrame({"Classifier":classifiers,"Accuracy":accuracy})