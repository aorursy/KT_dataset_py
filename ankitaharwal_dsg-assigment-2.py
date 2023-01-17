import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.metrics as score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

sns.set()
data=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")

data.head()
data.describe()
plt.matshow(data.corr(),cmap="Greys")

print("no multicollinearity")

plt.show()

print("singerID is damn usefull")

print(data.artistID.value_counts().head())

print("\nsongname is not much usefull")

print(data.songtitle.value_counts().head())
data=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")

y=data.Top10

data=data.drop(["Top10"],axis=1)

artistID=pd.get_dummies(pd.concat((data.artistID,data_test.artistID),axis=0))

songtitle=pd.get_dummies(pd.concat((data.songtitle,data_test.songtitle),axis=0))

del data["artistID"],data["artistname"],data["songID"],data["songtitle"]

del data_test["artistID"],data_test["artistname"],data_test["songID"],data_test["songtitle"]

data=pd.concat((data,artistID.iloc[0:4999]),axis=1)

data_test=pd.concat((data_test,artistID.iloc[4999:]),axis=1)

#data=pd.concat((data,songtitle.iloc[0:4999]),axis=1)

#data_test=pd.concat((data_test,songtitle.iloc[4999:]),axis=1)

print("If you want to use songtitle uncomment")
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

data_whole=pd.DataFrame(scaler.fit_transform(pd.concat((data,data_test),axis=0)),columns=data.columns)

data=data_whole.iloc[0:4999]

data_test=data_whole.iloc[4999:]
from sklearn.metrics import roc_auc_score,make_scorer

from sklearn.model_selection import StratifiedKFold,train_test_split

def auc_area(y,y_pred):

    return roc_auc_score(y_score=y_pred,y_true=y)

scorer=make_scorer(score_func=auc_area,greater_is_better=True,needs_proba=True)

cv=StratifiedKFold(n_splits=10,random_state=42)
lr=LogisticRegression(max_iter=500,multi_class="ovr",class_weight={0:1,1:1},solver="liblinear",penalty="l2",)

C=[0.015,0.1,1]

grid=[{"C":C}]

gs=GridSearchCV(estimator=lr,param_grid=grid,scoring=scorer,cv=cv)

gs.fit(data,y)
y_pred_proba=gs.best_estimator_.predict_proba(data)

print("AUC area on training data:",score.roc_auc_score(y,y_pred_proba.T[1]))

print("AUC area on testing data with cross validation:",gs.best_score_)

fpr,tpr,threshold=score.roc_curve(y,y_pred_proba.T[1])

plt.figure(figsize=(3,3))

plt.plot(fpr,tpr)

plt.plot([0,1],[0,1])

plt.xlabel("FPR or 1-specivity")

plt.ylabel("TPR or sensitivity")

plt.show()

gs_results=pd.DataFrame(dict(gs.cv_results_))

plt.plot(gs_results.mean_train_score,label="train")

plt.plot(gs_results.mean_test_score,label="test")

plt.xlabel("C value")

plt.xticks(list(range(len(C))),C)

plt.ylabel("AUC score")

plt.legend()

print(gs.best_estimator_)
gs=gs.best_estimator_

gs.fit(data,y)

y_pred_proba=gs.predict_proba(data_test)

y_pred_df=pd.DataFrame({})

y_pred_df["songID"]=pd.read_csv("../input/test.csv").songID

y_pred_df["Top10"]=y_pred_proba.T[1]

y_pred_df.to_csv("submission_best.csv",index=False)