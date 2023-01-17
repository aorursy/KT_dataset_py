import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

# from sklearn.cross_validation import train_test_split

import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
data =  pd.read_csv("../input/adult.csv")
data.head()
data.info()
data.isnull().sum()
# select all categorical variables

df_categorical = data.select_dtypes(include=['object'])



# checking whether any other columns contain a "?"

df_categorical.apply(lambda x: x=="?", axis=0).sum()
data[data['workclass'] == '?' ].count()
data[data['occupation'] == '?' ].count()
data[data['native.country'] == '?' ].count()
(1836/32561)/100
data.count()
data = data[data["workclass"] != "?" ]
data = data[data["occupation"] != "?" ]
data = data[data["native.country"] != "?" ]
data.count()
data.head()
data["income"].unique()
data["income"] = data["income"].map({'<=50K' : 0, '>50K': 1})

data.head()
data["income"].unique()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



catogorical_data = data.select_dtypes(include =['object'])
catogorical_data.head()
catogorical_data = catogorical_data.apply(le.fit_transform)
catogorical_data.head()
data = data.drop(catogorical_data.columns, axis=1)

data = pd.concat([data, catogorical_data], axis=1)

data.head()
data.info()
data['income'] = data['income'].astype('category')

x=data.drop('income',axis=1)

y=data['income']

#Train & Test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state= 476)
tree = DecisionTreeClassifier()

model_tree = tree.fit(x_train,y_train)

model_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
model_tree = tree.fit(x_train,y_train)

pred_tree = tree.predict(x_test)

a1 = accuracy_score(y_test,pred_tree)

print("The Accuracy of Desicion Tree is ", a1)
confusion_matrix(y_test,pred_tree)
print(classification_report(y_test, pred_tree))
rf = RandomForestClassifier()

model_rf = rf.fit(x_train,y_train)

pred_rf = rf.predict(x_test)

a2 = accuracy_score(y_test, pred_rf)

print("The Accuracy of Random Forest is ", a2)
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()



model_lg = lg.fit(x_train,y_train)

pred_lg = lg.predict(x_test)

a3 = accuracy_score(y_test, pred_lg)

print("The Accuracy of logistic regression is ", a3)
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier()
model_knn =knn.fit(x_train,y_train) 

pred_knn = knn.predict(x_test)

a4 = accuracy_score(y_test, pred_knn)

print("The Accuracy of KNN is ", a4)
rf_param = {

    "n_estimators": [25,50,100],

    "criterion" : ["gini"],

    "max_depth" : [3,4,5,6],

    "max_features" : ["auto","sqrt","log2"],

    "random_state" : [123]

}
GridSearchCV(rf, rf_param, cv = 5)
grid =GridSearchCV(rf, rf_param, cv = 5)
grid.fit(x_train,y_train).best_params_
rf1 = RandomForestClassifier(criterion = 'gini',

    max_depth = 6,

    max_features = 'auto',

    n_estimators = 100,

    random_state = 123)

model_rf1 = rf1.fit(x_train,y_train)

pred_rf1 = rf1.predict(x_test)

accuracy_score(y_test, pred_rf1)
cross_val_score(tree,x_train,y_train,scoring= "accuracy", cv=10)
cross_val_score(tree,x,y,scoring= "accuracy", cv=5).mean()
cross_val_score(rf,x_train,y_train,scoring= "accuracy", cv=5).mean()
cross_val_score(lg,x_train,y_train,scoring= "accuracy", cv=5).mean()
cross_val_score(knn,x_train,y_train,scoring= "accuracy", cv=5).mean()
from sklearn.ensemble import VotingClassifier
model_vote = VotingClassifier(estimators=[('logistic Regression', lg), ('random forrest', rf), ('knn neighbors', knn),(' decision tree', tree)], voting='soft')

model_vote = model_vote.fit(x_train, y_train)
vote_pred = model_vote.predict(x_test)
a5 =  accuracy_score(y_test, vote_pred)

print("The Accuracy of voting classifier is ", a5)
print(classification_report(y_test, vote_pred))
from sklearn.ensemble import BaggingClassifier
bagg = BaggingClassifier(base_estimator=rf1,n_estimators=15)
model_bagg =bagg.fit(x_train,y_train) 

pred_bagg = bagg.predict(x_test)
a6 = accuracy_score(y_test, pred_bagg)

print("The Accuracy of BAAGING is ", a6)
confusion_matrix(y_test,pred_bagg)
print(classification_report(y_test, pred_bagg))
from sklearn.ensemble import AdaBoostClassifier
Adaboost = AdaBoostClassifier(base_estimator=rf1, n_estimators=15)
model_boost =Adaboost.fit(x_train,y_train) 

pred_boost = Adaboost.predict(x_test)
a7 = accuracy_score(y_test, pred_boost)

print("The Accuracy of BOOSTING is ", a7)
confusion_matrix(y_test,pred_boost)
print(classification_report(y_test, pred_boost))