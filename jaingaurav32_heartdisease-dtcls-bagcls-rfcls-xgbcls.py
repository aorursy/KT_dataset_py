# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart_data.shape
heart_data.info()
heart_data.head()
heart_data.describe()
heart_data.hist(figsize=(10,10));
X=heart_data.copy()

del X['target']

y=heart_data.target

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
y.value_counts()
baseline_accuracy=165/(165+138)

baseline_accuracy
from sklearn.tree import DecisionTreeClassifier

hd_model=DecisionTreeClassifier(max_depth=4)
hd_model.fit(X_train,y_train)
from sklearn.tree import export_graphviz

from IPython.display import Image

export_graphviz(hd_model, out_file='clf_tree.dot',filled=False, rounded=True,special_characters=True,feature_names=X_train.columns)

! dot -Tpng clf_tree.dot -o clf_tree.png

Image("clf_tree.png")
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix, classification_report,roc_auc_score

hd_model.score(X_train,y_train)
y_pred_train=hd_model.predict(X_train)

y_pred_train
confusion_matrix(y_train,y_pred_train)
y_pred_test=hd_model.predict(X_test)

y_pred_test
confusion_matrix(y_test,y_pred_test)
accuracy_score(y_test,y_pred_test)
sens=41/(41+9)

sens
spec=28/(28+13)

spec
pred_prob=hd_model.predict_proba(X_test)

pred_prob[:,1]
fpr,tpr,t=roc_curve(y_test,pred_prob[:,1],pos_label=1)

print(fpr)

print(tpr)

print(t)
t[0]=1

print(t)
import matplotlib.pyplot as plt

plt.scatter(fpr,tpr,c=t)

plt.colorbar(ticks=np.arange(0,1,0.1))

plt.xlabel('fpr')

plt.ylabel('tpr')
from matplotlib.collections import LineCollection

import matplotlib as mpl

l1 = []

for i in range(len(fpr)-1):

    l1.append([(fpr[i],tpr[i]),(fpr[i+1],tpr[i+1])])

print(l1)



lc = LineCollection(l1, cmap='viridis')

fig, ax = plt.subplots()

line=ax.add_collection(lc)

lc.set_array(t[1:])



plt.axis([-0.1,1,0,1])

plt.colorbar(line, ticks=np.arange(0,1,0.1))

plt.show()
auc(fpr,tpr)
parameters={'max_depth':[1,2,3,4,5,6,7,8,9]} #hyperparameter tuning
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(hd_model,parameters,cv=10,scoring='accuracy')
grid.fit(X_train,y_train)
grid.best_params_
grid.best_score_
model_prune=DecisionTreeClassifier(max_depth=3,random_state=42)
model_prune.fit(X_train,y_train)
model_prune.score(X_train,y_train)
model_prune.feature_importances_
data=pd.Series(data=model_prune.feature_importances_,index=X.columns)

data.sort_values(ascending=True,inplace=True)

data.plot.barh()
y_prune=model_prune.predict(X_test)

y_prune
accuracy_score(y_test,y_prune)
confusion_matrix(y_test,y_prune)
sens_prune=40/(40+10)

sens_prune
spec_prune=28/(28+13)

spec_prune
print(classification_report(y_test,y_prune))
from sklearn.ensemble import BaggingClassifier

model_bag=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100,random_state=42)
model_bag.fit(X_train,y_train)
model_bag.score(X_train,y_train)
y_pred_bag=model_bag.predict(X_test)

y_pred_bag
accuracy_score(y_test,y_pred_bag)
confusion_matrix(y_test,y_pred_bag)
from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier(n_estimators=200,max_depth=5,max_features=12,oob_score=True,verbose=1,random_state=50)
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train)
model_rf.oob_score_
model_rf.feature_importances_
data_rf=pd.Series(data=model_rf.feature_importances_,index=X_train.columns)

data_rf.sort_values(ascending=True,inplace=True)

data_rf.plot.barh()
y_pred_rf=model_rf.predict(X_test)

y_pred_rf
accuracy_score(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_rf)
parameters_rf={'max_depth':np.arange(1,6),'max_features':np.arange(1,10)}
tune_model=GridSearchCV(model_rf,parameters_rf,cv=5,scoring='accuracy')
tune_model.fit(X_train,y_train)
tune_model.best_score_
tune_model.best_params_
y_pred_tune_rf=tune_model.predict(X_test)

y_pred_tune_rf
accuracy_score(y_test,y_pred_tune_rf)
confusion_matrix(y_test,y_pred_tune_rf)
#!pip install xgboost
from xgboost import XGBClassifier
model_xgb=XGBClassifier(objective='binary:logistic',n_estimators=500,random_state=80)
model_xgb.fit(X_train,y_train)
model_xgb.score(X_train,y_train)
y_pred_xgb=model_xgb.predict(X_test)
accuracy_score(y_test,y_pred_xgb)
confusion_matrix(y_test,y_pred_xgb)
parameters_xgb={'max_depth':np.arange(2,6),'learning_rate':[0.1,0.01,0.001]}
tune_xgb=GridSearchCV(model_xgb,parameters_xgb)
tune_xgb.fit(X_train,y_train)
tune_xgb.best_score_
tune_xgb.best_params_
y_pred_xgb_cv=tune_xgb.predict(X_test)
confusion_matrix(y_test,y_pred_xgb_cv)
accuracy_score(y_test,y_pred_xgb_cv)