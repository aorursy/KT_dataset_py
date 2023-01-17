# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import numpy as np

import xgboost as xgb

from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')
for column in train_dataset:

    if len(train_dataset[column].unique())==1 :

        print (column)

        del train_dataset[column]

        del test_dataset[column]

    elif column.__contains__('cat') and len(train_dataset[train_dataset[column]==1])<9 and len(train_dataset[train_dataset[column]==1])> 0:

        print (column)

        del train_dataset[column]

        del test_dataset[column]
from sklearn.base import TransformerMixin



class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)
dfi = DataFrameImputer()

train_imputer = dfi.fit_transform(train_dataset)

test_imputer = dfi.transform(test_dataset)
from sklearn import preprocessing 





for column in train_imputer:

    if column.__contains__('cat') and train_imputer[column].dtype == np.dtype('O'):

        print (column)

        Labelencoder = preprocessing.LabelEncoder()

        a = Labelencoder.fit(train_imputer[column].append(test_imputer[column], ignore_index=True)) 

        a = Labelencoder.transform(test_imputer[column])

        test_imputer[column] = a

        a = Labelencoder.transform(train_imputer[column])

        train_imputer[column] = a


del train_imputer['transaction_id']

del test_imputer['transaction_id']

train_imputer.info()
target = train_dataset['target']

del train_imputer['target']
from imblearn.over_sampling import SMOTE, ADASYN

x1, y1 = SMOTE().fit_sample(train_imputer, target)

def modelfit(xgbmodel, trainx, trainy, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = xgbmodel.get_xgb_params()

        xgtrain = xgb.DMatrix(trainx, label=trainy)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgbmodel.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=50)

        print (cvresult)

        xgbmodel.set_params(n_estimators=cvresult.shape[0])

        

    #Fit the algorithm on the data

    xgbmodel.fit(trainx, trainy,eval_metric='auc')



                        
xgb1 = xgb.XGBClassifier(

 learning_rate =0.01,

 n_estimators=1000,

 max_depth=7,

 min_child_weight=1,

 gamma=0,

 subsample=0.5,

 colsample_bytree=0.6,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=3,

 seed=123)
modelfit(xgb1,train_imputer.values,target,useTrainCV=False)
from sklearn.ensemble import IsolationForest

isf = IsolationForest(n_estimators=500, max_features = 6)

isf.fit(train_imputer.values,target)
ypred = xgb1.predict_proba(test_imputer.values)[:,1]

ypred1 = isf.predict(test_imputer.values)
ypred1[ypred1==1]= 0

ypred1[ypred1==-1]= 1
ypred1[:5]
pred_df = pd.DataFrame({'transaction_id':test_dataset['transaction_id'],'target':ypred}, columns=["transaction_id" , "target"]).to_csv("a.csv" , index=False)

pred_df = pd.DataFrame({'transaction_id':test_dataset['transaction_id'],'target':ypred1}, columns=["transaction_id" , "target"]).to_csv("b.csv" , index=False)

pred_df = pd.DataFrame({'transaction_id':test_dataset['transaction_id'],'target':ypred *0.5 + ypred1*0.5}, columns=["transaction_id" , "target"]).to_csv("c.csv" , index=False)