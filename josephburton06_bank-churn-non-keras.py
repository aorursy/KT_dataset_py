import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import tree



import jb_helper_functions_prep

from jb_helper_functions_prep import create_enc
df = pd.read_csv('Churn_Modelling.csv')

df.head(3)
df = create_enc(df, ['Geography', 'Gender'])

df = df[['CreditScore', 'Geography_enc',

       'Gender_enc', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',

       'IsActiveMember', 'EstimatedSalary', 'Exited']]
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['Exited']])
print('Train shape: ' + str(train.shape))

print('Test shape: ' + str(test.shape))
y_train = train[['Exited']]

y_test = test[['Exited']]
X_train = train[['IsActiveMember', 'Balance', 'CreditScore', 'Tenure', 'Age']]

X_test = test[['IsActiveMember', 'Balance', 'CreditScore', 'Tenure', 'Age']]
log_reg = LogisticRegression(random_state=123, solver='saga').fit(X_train, y_train)
y_pred = log_reg.predict(X_train)
print('Accuracy of Logistic Regression classifier on training set: {:.6f}'

     .format(log_reg.score(X_train, y_train)))
print(classification_report(y_train, y_pred))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=123).fit(X_train, y_train)
y_pred = clf.predict(X_train)

y_pred_proba = clf.predict_proba(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.6f}'

     .format(clf.score(X_train, y_train)))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
knn = KNeighborsClassifier(n_neighbors=4, weights='uniform')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_train)

print('Accuracy of KNN classifier on training set: {:.6f}'

     .format(knn.score(X_train, y_train)))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
rf = RandomForestClassifier(bootstrap=True, 

                            class_weight=None, 

                            criterion='gini',

                            min_samples_leaf=3,

                            n_estimators=100,

                            max_depth=4, 

                            random_state=123)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_train)

print('Accuracy of random forest classifier on training set: {:.6f}'

     .format(rf.score(X_train, y_train)))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
print('Accuracy of Decision Tree classifier on test set: {:.6f}'

     .format(clf.score(X_test, y_test)))
import graphviz



from graphviz import Graph

from sklearn.tree import export_graphviz



dot_data = export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 



graph.render('churn_decision_tree', view=True)