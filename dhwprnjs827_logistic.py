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
#read csv
data=pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
#data information
print(data.info())
print(data.head())
# is data balance?
# yes  50vs50
lose=data[data['blueWins']==0]
win=data[data['blueWins']==1]
print(win.shape[0]/(win.shape[0]+lose.shape[0]))
#My goal is to determine if the user can win based on the model I provided during the game.
#In other words, only the information that can be obtained in the game is selected.
# and gameid is not important in model 
data=data.drop('gameId',axis=1)
y=data['blueWins']
X=data.drop(['blueWins','blueEliteMonsters','blueHeralds','blueTotalGold','blueTotalExperience',
         'blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','blueTotalMinionsKilled',
         'blueGoldPerMin','redWardsPlaced', 'redWardsDestroyed','redFirstBlood',
         'redKills', 'redDeaths', 'redAssists',
       'redEliteMonsters','redHeralds','redTotalGold', 'redTotalExperience','redTotalJungleMinionsKilled', 'redGoldDiff',
       'redExperienceDiff','redGoldPerMin'],axis=1)
X['CSPerdiff']=X['blueCSPerMin']-X['redCSPerMin']
X['avgleveldiff']=X['blueAvgLevel']-X['redAvgLevel']
X=X.drop(['blueCSPerMin','redCSPerMin','blueAvgLevel','redAvgLevel'],axis=1)
column=X.columns
print(X)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,make_scorer#정확도,민감도등
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=\
    train_test_split(X,y,
    test_size=0.4,
    train_size=0.6,
    random_state=12354,
    shuffle=True)

stdc=StandardScaler()
X_train=stdc.fit_transform(X_train)
X_test=stdc.transform(X_test)
#make model
logic=LogisticRegression(penalty='elasticnet',solver='saga',n_jobs=-1,l1_ratio=0.4)
logic.fit(X_train,y_train)
y_pred=logic.predict(X_test)
confmat=pd.DataFrame(confusion_matrix(y_test,y_pred),
index=['True[0]','True[1]'],
columns=['Predict[0]','predict[1]'])
print(confmat)
print(classification_report(y_test,y_pred)) 
#accuarcy  71%
#checking overfitting
from sklearn.model_selection import KFold
fold=KFold(n_splits=10)
train_sizes,train_scores,test_scores=\
    learning_curve(estimator=logic,#수정
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=-1,
    cv=fold)
import matplotlib.pyplot as plt


train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)

test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,
color='blue',marker='o',
markersize=5,label='training accuracy')

plt.fill_between(train_sizes,
train_mean+train_std,
train_mean-train_std,
alpha=0.5,color='blue')

plt.plot(train_sizes,test_mean,
color='green',linestyle='--',
marker='s',markersize=5,
label='validation accuracy')
plt.fill_between(train_sizes,
test_mean+test_std,
test_mean-test_std,
alpha=0.15,color='green')

plt.grid()
plt.xlabel('number of trainning samples')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.ylim([0,1.03])
plt.tight_layout()
plt.show()

# I was able to determine that there was no problem with overfit.
#roc_auc_score
fpr,tpr,thresholds=roc_curve(y_test,logic.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,'--',label='logic')
plt.plot([0,1],[0,1],'-',label='50%')
plt.plot([fpr],[tpr],'r-',ms=10)
plt.show()
cross_stfold=cross_validate(estimator=logic,X=X_train,y=y_train,cv=fold,n_jobs=-1,
                          scoring=['accuracy','roc_auc'])

print(cross_stfold['test_accuracy'].mean())
#72%
print(cross_stfold['test_accuracy'].std())
# 2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

logicpipe=make_pipeline(LogisticRegression(penalty='elasticnet',solver='saga'
                                           ,l1_ratio=0.5))
print(logicpipe.get_params().keys())
param_l1_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
param_penalty=['elasticnet']
param_solver=['saga']
param_gid=[{'logisticregression__l1_ratio':param_l1_ratio,
           'logisticregression__solver':param_solver}]
gs=GridSearchCV(estimator=logicpipe,param_grid=param_gid,scoring='accuracy',cv=fold,n_jobs=-1)
gs.fit(X_train,y_train)
print(gs.cv_results_)
print(gs.best_score_.mean())
# i find logic best params 
print(gs.best_params_)
new=gs.best_estimator_
y_pred_gs=new.predict(X_test)
confmat1=pd.DataFrame(confusion_matrix(y_test,y_pred_gs),
index=['True[0]','True[1]'],
columns=['Predict[0]','predict[1]'])
print(confmat1)
print('Classification Report')
print(classification_report(y_test,y_pred_gs))
train_sizes,train_scores,test_scores=\
    learning_curve(estimator=new,#수정
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=-1,
    cv=fold)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)

test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,
color='blue',marker='o',
markersize=5,label='training accuracy')

plt.fill_between(train_sizes,
train_mean+train_std,
train_mean-train_std,
alpha=0.5,color='blue')

plt.plot(train_sizes,test_mean,
color='green',linestyle='--',
marker='s',markersize=5,
label='validation accuracy')
plt.fill_between(train_sizes,
test_mean+test_std,
test_mean-test_std,
alpha=0.15,color='green')

plt.grid()
plt.xlabel('number of trainning samples')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.ylim([0,1.03])#수정  y값의 범위
plt.tight_layout()

plt.show()
#Let's see what features in the best model 
#have the most probabilistic effect
new_logic=LogisticRegression(l1_ratio=0.7, penalty='elasticnet',
                                    solver='saga',n_jobs=-1)
new_logic.fit(X_train,y_train)
print("weight {}".format(new_logic.coef_))
print("max weight {}".format(new_logic.coef_.max()))
# max weight location is 3
frame=pd.DataFrame(data=X_train,columns=column)
print(frame)
#I found out that what's in the third column now has the most impact.
#But from a statistical perspective,
#I wondered if I could get the same result.

import statsmodels.api as sm

xts=StandardScaler()
X=xts.fit_transform(X)
logit_mod=sm.Logit(y,X)
result=logit_mod.fit()
print(result.summary())
print(np.exp(result.params))
#Just like the model, from the odds concept,
#we could see that the killer had the most impact on winning the game.
