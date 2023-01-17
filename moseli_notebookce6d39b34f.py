# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "."]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import re

import math

import seaborn as sb

from matplotlib import pyplot as plt



from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split as tts

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.preprocessing import MinMaxScaler



from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.neural_network import MLPClassifier as MLPC

from sklearn.neighbors import KNeighborsClassifier as kNC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier as DTC

import xgboost as XGB
filename='../input/train.csv'

try:

    train=pd.read_csv(filename)

    print("Dataset %s successfully loaded"%filename)

except Exception as k:

    print(k)

    raise



data=train



data.shape

data.describe()


features=[k for k in data]

feat_stats={}



for k in features: 

    feat_stats[k]=list([data[k].min(),data[k].max(),data[k].mean(),sum(data[k])])

print(feat_stats)
del data['Soil_Type7'], data['Soil_Type15']
corr_mat=data.corr()

sb.heatmap(corr_mat,vmax=.9,square=True)
y=data['Cover_Type']

x=data

del x['Cover_Type'],x['Id']
scaler=MinMaxScaler()

x_scaled=scaler.fit_transform(x)
pca=PCA(n_components=20)

pca.get_params()

x_pca=pca.fit_transform(x_scaled)
x_train,x_test,y_train,y_test=tts(x_pca,y,test_size=0.35,random_state=0)
ANN=MLPC(warm_start=True,hidden_layer_sizes=100)

ranfor=RFC(n_jobs=-1,random_state=0,verbose=1)

tree=DTC(random_state=0)

xgboost=XGB.XGBClassifier(objective="multi:softmax")

knn=kNC(n_jobs=-1,weights='distance')

svm=SVC(verbose=True,random_state=0)
algorithms={}

algorithms['Random Forrest']=ranfor

algorithms['Neural Network']=ANN

algorithms['Decision Tree']=tree

algorithms['K Neighbors']=knn

algorithms['Xgboost']=xgboost

algorithms['Support Vector Machine']=svm



cv = ShuffleSplit(n_splits=1,test_size=0.35, random_state=0)
#ANN

ANN.get_params()

param_grid=dict(learning_rate=['adaptive'])

grid=GridSearchCV(algorithms['Neural Network'],param_grid=param_grid,

                  cv=cv,n_jobs=-1,verbose=1)

grid.fit(x_pca,y)

best_ann=grid.best_estimator_

best_param_ann=grid.best_params_

best_score_ann=grid.best_score_

print(best_score_ann)

print(best_param_ann)
#SVM

svm.get_params()

param_grid=dict(degree=[2,3])

grid=GridSearchCV(algorithms['Support Vector Machine'],param_grid=param_grid,

                  cv=cv,n_jobs=-1,verbose=1)

grid.fit(x_pca,y)

best_svm=grid.best_estimator_

best_param_svm=grid.best_params_

best_score_svm=grid.best_score_

print(best_score_svm)

print(best_param_svm)

    #KNN

print(knn.get_params())
param_grid=dict(n_neighbors=[6],leaf_size=[25,30])

grid=GridSearchCV(algorithms['K Neighbors'],param_grid=param_grid,

                  cv=cv,n_jobs=-1,verbose=1)

grid.fit(x_pca,y)

best_knn=grid.best_estimator_

best_param_knn=grid.best_params_

best_score_knn=grid.best_score_

print(best_score_knn)

print(best_param_knn)
   #xgboost

print(xgboost.get_params())
param_grid=dict(max_depth=[9],silent=[False],n_estimators=[110],learning_rate=[0.27])

grid=GridSearchCV(algorithms['Xgboost'],param_grid=param_grid,

                  cv=cv,n_jobs=-1,verbose=1)

grid.fit(x_pca,y)

best_xgb=grid.best_estimator_

best_param_xgb=grid.best_params_

best_score_xgb=grid.best_score_

print(best_score_xgb)

print(best_param_xgb)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    return plt



plot_cv = StratifiedShuffleSplit(n_splits=2,test_size=0.35, random_state=1) 



plot_learning_curve(best_xgb,'Gradient boosting',

                    x_pca,y,ylim=(0.7,1.07),cv=plot_cv)
xgb_pred=grid.predict(x_test)

print('Prediction Acuracy on test: %s'%accuracy_score(y_test,ann_pred))
filename='../input/test.csv'

try:

    test=pd.read_csv(filename)

    print("Dataset %s successfully loaded"%filename)

except Exception as k:

    print(k)

    raise
test.describe()
del test['Soil_Type7'],test['Soil_Type15']
kk=pd.DataFrame()

kk['ID']=test['Id']
del test['Id']
test_scaled=scaler.transform(test)

test_scaled_pca=pca.transform(test_scaled)
kk['Cover_Type']=grid.predict(test_scaled_pca)
kk.head()
kk.to_csv('submission1.csv',index=False)

print(check_output(["ls", "."]).decode("utf8"))