import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.columns
df.Class.value_counts(normalize = True)*100
df.Time.value_counts().plot.hist(bins =20)
for i in ['Time']:

    sns.distplot(df[[i]].value_counts())

    plt.show()
X = df.drop(['Class'],axis = 1)

y = df.Class
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 2,stratify = y)
xgb = XGBClassifier()

cross_val_score(xgb,X_train,y_train,cv = 5,scoring = 'accuracy').mean()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

accuracy_score(y_test,y_pred)*100
confusion_matrix(y_test,y_pred)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

print(tn, fp, fn, tp)
precession = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

accuracy = (tp+tn)/(tp+tn+fp+fn)*100
F1score = 2*(precession*recall)/(precession+recall)
print(f'accuracy is {accuracy}\nPrecession is {precession}\nrecall is {recall}\nF1 score is {F1score}')