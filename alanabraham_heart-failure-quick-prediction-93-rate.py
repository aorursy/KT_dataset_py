import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isna().sum()
corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
small_df=df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
x = small_df

y = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Linear Regression Success Rate :", "{:.2f}%".format(100*s1))
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

p2=gbc.predict(x_test)

s2=accuracy_score(y_test,p2)

print("Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))
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



grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', n_jobs=4, 

                    cv=skf.split(x_train,y_train), verbose=0 )



grid.fit(x_train,y_train,early_stopping_rounds=20,eval_set=[(x_test, y_test)])

#print('\n Best estimator:')

#print(grid.best_estimator_)

#print('\n Best parameters:')

#print(grid.best_params_)

p2x = grid.best_estimator_.predict(x_test)

s2x=accuracy_score(y_test,p2x)
print("Extra Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2x))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

p4=svm.predict(x_test)

s4=accuracy_score(y_test,p4)

print("Support Vector Machine Success Rate :", "{:.2f}%".format(100*s4))
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