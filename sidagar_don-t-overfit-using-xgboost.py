# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xg

import seaborn as sb

import plotly.express as px

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier
df_train = pd.read_csv('/kaggle/input/dont-overfit-ii/train.csv')
df_train.describe()
corr = df_train.drop('id',axis =1).corr()



'''fig = px.imshow(img = corr,color_continuous_scale = [[0,'white'],[0.33,'yellow'],

                                                     [0.66,'red'],[1.0,'black']],

                labels = {'0':'target'},

                height = 1100,width = 1100,color_continuous_midpoint = 0,

                title = 'Correlation matrix for all fields.')

fig.show()'''
df_train_x = df_train.drop(['id','target'],axis=1)

df_train_y = df_train[['target']]

df_train_x.describe()
df_train_y.describe()
models = ['Gaussian Classifier','Bernaulli Classifier','Random Forest Classifier',

          'XGBoost Classifier']
def custom_scorer(y_true,y_pred):

    return roc_auc_score(y_true,y_pred)

scorer = make_scorer(custom_scorer,greater_is_better = True)
gauss = GaussianProcessClassifier()

gauss_dic = {'warm_start':[True,False],'multi_class':["one_vs_rest","one_vs_one"],

             'n_restarts_optimizer':[0,1,2],'max_iter_predict': [20,50,100]}



bernoulli = BernoulliNB()

bernoulli_dic = {'alpha' : [0,1,2,4,8]}



forest = RandomForestClassifier()

forest_dic = {'n_estimators': [10,20,30,40],'criterion':['geni','entropy'],

              'max_depth':[2,4,6,8,10]}



xgb = xg.XGBClassifier()

xgb_dic = {'max_depth':[2,4,6,8,10,12],"eta": [0.01,0.03,0.05],'gamma' : [0,0.00001,0.0001,0.001]}
best_params = []

best_score = []

gauss_clf = GridSearchCV(gauss,gauss_dic,cv=9,scoring = scorer,n_jobs =-1)

gauss_clf.fit(df_train_x,df_train_y.to_numpy().ravel())

best_params.append(gauss_clf.best_params_)

best_score.append(gauss_clf.best_score_)



bernoulli_clf = GridSearchCV(bernoulli,bernoulli_dic,cv=9,scoring = scorer,n_jobs =-1)

bernoulli_clf.fit(df_train_x,df_train_y.to_numpy().ravel())

best_params.append(bernoulli_clf.best_params_)

best_score.append(bernoulli_clf.best_score_)



forest_clf = GridSearchCV(forest,forest_dic,cv=9,scoring = scorer,n_jobs =-1)

forest_clf.fit(df_train_x,df_train_y.to_numpy().ravel())

best_params.append(forest_clf.best_params_)

best_score.append(forest_clf.best_score_)



xgb_clf = GridSearchCV(xgb,xgb_dic,cv=9,scoring = scorer,n_jobs =-1)

xgb_clf.fit(df_train_x,df_train_y.to_numpy().ravel())

best_params.append(xgb_clf.best_params_)

best_score.append(xgb_clf.best_score_)
for i in range (4):

    print ('The best  parameters for the model ' + models[i] + 'are : '+ str(best_params[i]) )

    print ('The score obtained using the model ' + models[i] + 'is : '+ str(best_score[i]) )
df_test = pd.read_csv('/kaggle/input/dont-overfit-ii/test.csv')

df_test.describe()
output= df_test[['id']]

df_test = df_test.drop('id',axis = 1)
output['xgb'] = xgb_clf.predict(df_test)

output['target'] = output['xgb']
output[['id','target']].to_csv('Submission.csv',index = False)