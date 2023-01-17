import numpy as np 

import pandas as pd 

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
Data = pd.read_csv("../input/heart-risk/Heart_risk.csv")

Data.head()
Data.tail()
Data.dtypes
Data.describe()
Data['Class'].value_counts()
Data.isnull().sum()
import seaborn as sns

sns.pairplot(Data,diag_kind='kde')

import matplotlib.pyplot as plt

plt.show()
Data.shape
X=Data.drop(['Class'],axis=1)

Y=Data[['Class']]
# Spliting the dataset into train and test 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, 

                                  min_samples_leaf=5) 
clf_gini.fit(X_train, Y_train)
y_pred = clf_gini.predict(X_test) 

print("Predicted values:") 

print(y_pred)# GiniIndex

print(confusion_matrix(y_test, y_pred))
# GiniIndex

print(confusion_matrix(Y_test, y_pred))
accuracy_score(Y_test,y_pred)
clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100,max_depth = 3,

                                     min_samples_leaf = 5) 
clf_entropy.fit(X_train, Y_train) 
y_pred_2 = clf_entropy.predict(X_test) 

print("Predicted values:") 

print(y_pred_2)
print(confusion_matrix(Y_test, y_pred_2))
accuracy_score(Y_test,y_pred_2)
from sklearn.ensemble import BaggingClassifier
mdl=DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=mdl, n_estimators=100, random_state=100)
model.fit(X_train, Y_train)
y_pred3 = model.predict(X_test) 

print("Predicted values:") 

print(y_pred3)
print(confusion_matrix(Y_test, y_pred3))
accuracy_score(Y_test,y_pred3)
from sklearn import model_selection

import warnings

warnings.filterwarnings('ignore')
kfold = model_selection.KFold(n_splits=10, random_state=100)
results = model_selection.cross_val_score(model, X_test, Y_test, cv=kfold)

print(results.mean())
from sklearn.ensemble import AdaBoostClassifier
kfold = model_selection.KFold(n_splits=10, random_state=100)
model1 = AdaBoostClassifier(base_estimator=mdl,n_estimators=10, random_state=100)
model1.fit(X_train, Y_train)
y_pred4 = model1.predict(X_test) 

print("Predicted values:") 

print(y_pred4)
accuracy_score(Y_test,y_pred4)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier
LR=LogisticRegression()

NB=GaussianNB()

KNN=KNeighborsClassifier(n_neighbors=12, weights='distance')
LR_bag=BaggingClassifier(base_estimator=LR,n_estimators=25,random_state=100)

NB_bag=BaggingClassifier(base_estimator=NB,n_estimators=24,random_state=100)

KNN_bag=BaggingClassifier(base_estimator=KNN,n_estimators=80,random_state=100)

#Boosting models

LR_boost=AdaBoostClassifier(base_estimator=LR,n_estimators=170)

NB_boost=AdaBoostClassifier(base_estimator=NB,n_estimators=350)

GB_boost=GradientBoostingClassifier(n_estimators=100)

#Stacked Model

stacked = VotingClassifier(estimators = [('Bagged_LR',LR_bag), 

                                         ('GBoost', GB_boost)],voting='soft')
LR_bag.fit(X_train, Y_train)

y_pred5 =LR_bag.predict(X_test) 

print("Predicted values:") 

print(y_pred5)

accuracy_score(Y_test,y_pred5)
LR_boost.fit(X_train, Y_train)

y_pred6=LR_boost.predict(X_test) 

print("Predicted values:") 

print(y_pred6)

accuracy_score(Y_test,y_pred6)
NB_bag.fit(X_train, Y_train)

y_pred7=NB_bag.predict(X_test) 

print("Predicted values:") 

print(y_pred7)

accuracy_score(Y_test,y_pred7)
NB_boost.fit(X_train, Y_train)

y_pred9 =NB_boost.predict(X_test) 

print("Predicted values:") 

print(y_pred9)

accuracy_score(Y_test,y_pred9)
GB_boost.fit(X_train, Y_train)

y_pred8 =GB_boost.predict(X_test) 

print("Predicted values:") 

print(y_pred8)

accuracy_score(Y_test,y_pred8)
stacked.fit(X_train, Y_train)

y_pred10 =stacked.predict(X_test) 

print("Predicted values:") 

print(y_pred10)

accuracy_score(Y_test,y_pred10)
#Based on the model building decision tree classifier has highest accuracy (0.84) among gradient

#boost,naive's bayes,logistic regressors.