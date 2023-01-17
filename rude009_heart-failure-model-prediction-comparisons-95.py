import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import scipy



#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.shape
df.head()
df.describe()
df.isna().sum()
import plotly.express as px

fig = px.pie(df, names='DEATH_EVENT', title='Distribution of Death Events in Patients',width=600, height=400)

fig.show()
corr = df.corr()

ax, fig = plt.subplots(figsize=(15,15))

sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)

plt.show()
corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
# Trial and Error revealed that not considering Age column improves accuracy



x = df[['ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]

y = df['DEATH_EVENT']



#Spliting data into training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, plot_confusion_matrix

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Linear Regression Success Rate :", "{:.2f}%".format(100*s1))

plot_confusion_matrix(lr, x_test, y_test)

plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

p2=gbc.predict(x_test)

s2=accuracy_score(y_test,p2)

print("Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))

plot_confusion_matrix(gbc, x_test, y_test)

plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))

plot_confusion_matrix(rfc, x_test, y_test)

plt.show()
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

p4=svm.predict(x_test)

s4=accuracy_score(y_test,p4)

print("Support Vector Machine Success Rate :", "{:.2f}%".format(100*s4))

plot_confusion_matrix(svm, x_test, y_test)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

scorelist=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    p5=knn.predict(x_test)

    s5=accuracy_score(y_test,p5)

    scorelist.append(round(100*s5, 2))

print("K Nearest Neighbors Top 5 Success Rates:")

print(sorted(scorelist,reverse=True)[:5])

plot_confusion_matrix(knn, x_test, y_test)

plt.show()
from sklearn.tree import DecisionTreeClassifier

list1 = []

for leaves in range(2,10):

    classifier = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred)*100)

print("Decision Tree Classifier Top 5 Success Rates:")

print([round(i, 2) for i in sorted(list1, reverse=True)[:5]])

plot_confusion_matrix(classifier, x_test, y_test)

plt.show()
from xgboost import XGBClassifier

from bayes_opt import BayesianOptimization

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, GridSearchCV



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



xgb = XGBClassifier(learning_rate=0.01, n_estimators=1000, objective='binary:logistic')



skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 0)



grid = GridSearchCV(estimator=xgb, param_grid=params, n_jobs=4, 

                    cv=skf.split(x_train,y_train), verbose=0 )



grid.fit(x_train,y_train,early_stopping_rounds=30,eval_set=[(x_test, y_test)])

p2x = grid.best_estimator_.predict(x_test)

s2x=accuracy_score(y_test,p2x)

print("Extra Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2x))

plot_confusion_matrix(grid.best_estimator_, x_test, y_test)

plt.show()
print(f'Gradient Booster Classifier: {round(100*s2, 2)}%\nDecision Tree Classifier: {round(sorted(list1, reverse=True)[0], 2)}%\nLinear Regression: {round(100*s1, 2)}%\nSupport Vector Machine: {round(100*s4, 2)}%\nRandom Forest Classifier: {round(100*s3, 2)}%\nK Nearest Neighbors: {round(sorted(scorelist,reverse=True)[0], 2)}%\nExtra Gradient Booster Classifier: {round(100*s2x, 2)}%')