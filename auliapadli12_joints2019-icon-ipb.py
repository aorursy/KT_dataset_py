import pandas as pd

import numpy as np

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report

from xgboost import plot_tree

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import matplotlib.pylab as plt

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test_data.csv")

test_id=test['id']

test=test.drop("id",axis=1)

train=train.drop("id",axis=1)



train.describe()
perempuan=train[train['gender']=='putri'].copy()

laki=train[train['gender']=='putra'].copy()

campur=train[train['gender']=='campur'].copy()



for column in ['fac_1','fac_2','fac_3','fac_4','fac_5','fac_6','fac_7','fac_8']:

  perempuan[column].fillna(perempuan[column].mode().iloc[0], inplace = True)

  laki[column].fillna(laki[column].mode().iloc[0], inplace=True)

  campur[column].fillna(campur[column].mode().iloc[0], inplace=True)



perempuan.iloc[:,8:15] = perempuan.iloc[:,8:15].fillna(perempuan.iloc[:,8:15].median())

laki.iloc[:,8:15] = laki.iloc[:,8:15].fillna(laki.iloc[:,8:15].median())

campur.iloc[:,8:15] = campur.iloc[:,8:15].fillna(campur.iloc[:,8:15].median())



train_baru = pd.concat([perempuan,laki,campur],axis=0)

train_baru.describe()
train_baru.describe()
for column in [x for x in train_baru.columns if x not in ['gender','size']]:

    train_baru[column]=train_baru[column].astype('int64')

train_baru.dtypes
le=LabelEncoder()

le.fit(train_baru['gender'])

label=le.transform(train_baru['gender'])

label2=le.inverse_transform(label)

label
predictors = [x for x in train_baru.columns if x not in ['gender']]

predictors
def modelfit(alg, train, predictors,labels,cv_folds=5, early_stopping_rounds=50):

    

    xgb_param = alg.get_xgb_params()

    xgtrain = xgb.DMatrix(train[predictors].values, label=labels)

    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], metrics='merror',nfold=cv_folds,early_stopping_rounds=early_stopping_rounds)

    alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(train[predictors], train['gender'],eval_metric='merror')

        

    #Predict training set:

    train_predictions = alg.predict(train[predictors])

    train_predprob = alg.predict_proba(train[predictors])[:,1]

        

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(train['gender'].values, train_predictions))

    

    feat_imp = pd.Series(alg.feature_importances_,predictors).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
xgb1 = XGBClassifier(

 learning_rate =0.05,

 n_estimators=1000,

 num_class=3,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'multi:softmax',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb1, train_baru, predictors,label)
y_pred = xgb1.predict(test)

y_pred
from collections import Counter

print(Counter(y_pred))
submit = pd.concat([test_id,pd.Series(y_pred)], axis=1)

submit.columns = ["id", "gender"]

submit.to_csv(index=False)