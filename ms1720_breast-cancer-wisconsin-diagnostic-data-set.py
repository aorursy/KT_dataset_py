import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import statsmodels.api as sm
import seaborn as sns
bc = pd.read_csv('../input/data.csv')
bc.head()
bc.describe()
bc.info()
dum = pd.get_dummies(bc.diagnosis)
bc = pd.concat([bc, dum], axis = 1)
bc = bc.drop('diagnosis', axis = 1)
bc = bc.drop('B', axis = 1)
bc = bc.drop('Unnamed: 32', axis = 1)
bc.head()
bc.drop(['id'], axis = 1).hist(figsize = (14,14))
plt.show()
plt.figure(figsize = (12,10))
sns.heatmap(bc.corr())
plt.show()
def num_densityplot():
    for n in range(1, 31):
        plt.subplot(9, 4, n)
        bc.iloc[:, n].plot.kde()
        plt.xlabel(bc.iloc[:, n].name)
        
plt.figure(figsize = (25, 60))
num_densityplot()
plt.show()
bc2 = bc[['radius_mean','perimeter_mean','area_mean','concavity_mean',
         'concave points_mean','radius_worst','perimeter_worst','area_worst',
         'concave points_worst']]
a = pd.plotting.scatter_matrix(bc2, figsize = (15, 10))
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.show()
sns.pairplot(bc, x_vars = bc2.columns[0: 4], y_vars = ['M'], kind = 'reg')
plt.yticks([0.0, 1.0],['Benign', 'Malignant'])
sns.pairplot(bc, x_vars = bc2.columns[4: ], y_vars = ['M'], kind = 'reg')
plt.yticks([0.0, 1.0],['Benign', 'Malignant'])
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier

from yellowbrick.classifier import ConfusionMatrix
X = bc.drop(['id','M'], axis = 1).values
y = bc.M
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)
def model_fit(x):
    x.fit(X_train, y_train)
    y_pred = x.predict(X_test)
    model_fit.accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy Score',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    model_cm = ConfusionMatrix(
    x, classes = ['Malignat', 'Benign'],
    label_encoder = {1 : 'Malignat', 0 : 'Benign'})
    model_cm.fit(X_train, y_train)
    model_cm.score(X_test, y_test)
    
    model_cm.poof() 
list = []
for i in range(1,10): 
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    list.append(accuracy_score(y_test, y_pred))
   
for n in range(0, len(list)):
    if list[n] == max(list):
        i = n+1
model_fit(KNeighborsClassifier(n_neighbors = i))
KNN = model_fit.accuracy
from sklearn.linear_model import LogisticRegression
model_fit(LogisticRegression())
Logistic = model_fit.accuracy
from sklearn.naive_bayes import GaussianNB
model_fit(GaussianNB())
Gaussian = model_fit.accuracy
from sklearn import tree
model_fit(tree.DecisionTreeClassifier())
Tree = model_fit.accuracy
from sklearn.ensemble import RandomForestClassifier
model_fit(RandomForestClassifier(n_estimators = 100, max_depth =10, random_state = 1))
RandomForest = model_fit.accuracy
list=[]
ival = range(1, 100)
jval = range(1,100)
for i,j in zip(ival, jval): 
    clfr = RandomForestClassifier(n_estimators = i, max_depth = j, random_state = 1)
    clfr.fit(X_train, y_train)
    y_pred = clfr.predict(X_test)
    
    list.append((accuracy_score(y_test, y_pred)))
list = pd.DataFrame(list)
list[list == list.max()].dropna().head()
from xgboost import XGBClassifier
model_fit(XGBClassifier())
XGBClf = model_fit.accuracy
scores_list_1 = ['KNN','Logistic','Gaussian','Tree','RandomForest','XGBClassifier']
scores_1 = [KNN, Logistic, Gaussian, Tree, RandomForest, XGBClf]
score_df_classification = pd.DataFrame([scores_list_1, scores_1]).T
score_df_classification.index = score_df_classification[0]
del score_df_classification[0]
score_df_classification