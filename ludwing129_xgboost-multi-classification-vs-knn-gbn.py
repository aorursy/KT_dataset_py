import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import  accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



import xgboost as xgb

from xgboost.sklearn import XGBClassifier



from sklearn.preprocessing import StandardScaler, LabelBinarizer

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import recall_score

def printreport(exp, pred):

    print(classification_report(exp, pred))

    print("recall score")

    print(recall_score(exp,pred,average = 'macro'))
gr = pd.read_csv('../input/greedata/featuren.csv')
sns.set(style="whitegrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

ax = sns.violinplot(x="label", y="v2", data=gr, inner=None)

ax = sns.swarmplot(x="label", y="v2", data=gr,color="c", edgecolor="black")
for column in gr.columns[0:-1]:

    for spec in gr["label"].unique():

        selected_spec = gr[gr["label"] == spec]

        selected_column = selected_spec[column]

        

        std = selected_column.std()

        avg = selected_column.mean()

        

        three_sigma_plus = avg + (3 * std)

        three_sigma_minus =  avg - (3 * std)

        

        outliers = selected_column[((selected_spec[column] > three_sigma_plus) | (selected_spec[column] < three_sigma_minus))].index

        gr.drop(outliers, inplace=True)

        print(column, spec, outliers)
x = gr.iloc[:,0:3].values 

y = gr.label.values

print(x)

print(y)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0, stratify=y)
# 7. XGboost Classification

import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import plot_importance



xgb_cls = xgb.XGBClassifier(objective="multi:softmax", num_class=5)

xgb_cls.fit(x_train, y_train)

y_pred = xgb_cls.predict(x_test)



print('XGboost')

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

cm = confusion_matrix(y_test, y_pred)

printreport(y_test, y_pred)







#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()



plot_importance(xgb_cls)
kfold = KFold(n_splits=5, shuffle=True)

results = cross_val_score(xgb_cls, x, y, cv=kfold)

print(results)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# 4. Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(x_train, y_train)



y_pred = gnb.predict(x_test)



print('GNB')

cm = confusion_matrix(y_test,y_pred)

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

printreport(y_test, y_pred)



#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(x_train,y_train)



y_pred = knn.predict(x_test)



print('KNN')

cm = confusion_matrix(y_test,y_pred)

s = accuracy_score(y_test, y_pred)

print('accury')

print(s)

printreport(y_test, y_pred)



#print(cm)

import seaborn as sns

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()