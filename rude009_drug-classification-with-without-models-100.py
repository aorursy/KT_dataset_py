#Imports



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import warnings



#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/drug-classification/drug200.csv')
df.head()
fig = px.pie(df,names='Drug', title='Drug Distribution',width=600, height=400)

fig.show()
df['Drug'].replace('DrugY', 'drugY', inplace=True)
#Copying df to avoid manipulating original data

df2 = df.copy()



df2['Sex'].replace({'M', 'F'},{1, 0}, inplace=True)

df2['BP'].replace({'HIGH', 'LOW', 'NORMAL'},{1, -1, 0}, inplace=True)

df2['Cholesterol'].replace({'HIGH', 'NORMAL'},{1, 0}, inplace=True)
x = df2.drop(['Drug'], axis=1)

y = df2['Drug']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression

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

print(sorted(scorelist)[:-6:-1])

plot_confusion_matrix(knn, x_test, y_test)

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



grid.fit(x_train,y_train,early_stopping_rounds=20,eval_set=[(x_test, y_test)])

p2x = grid.best_estimator_.predict(x_test)

s2x=accuracy_score(y_test,p2x)

plot_confusion_matrix(grid.best_estimator_, x_test, y_test)

plt.show()
print("Extra Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2x))
target = df['Drug']

x = df.drop('Drug', axis=1)



pred = []

for index, row in x.iterrows():

    if row['Na_to_K'] > 15:

        pred.append('drugY')

    elif row['BP']=='HIGH' and row['Age'] <= 50:

        pred.append('drugA')

    elif row['BP']=='HIGH' and row['Age'] >50:

        pred.append('drugB')

    elif row['BP']=='LOW' and row['Cholesterol']=='HIGH':

        pred.append('drugC')

    else:

        pred.append('drugX')
print(accuracy_score(target, pred)*100,'%', sep='')