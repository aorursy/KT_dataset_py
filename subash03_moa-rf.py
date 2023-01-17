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
scored=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

noscored=pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train=pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

submit=pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

test=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
X=train.drop('sig_id',axis=1)

Y=scored.drop('sig_id',axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss

from category_encoders.target_encoder import TargetEncoder

from sklearn.decomposition import PCA
X1=pd.concat([X,Y],axis=1)

corr1=X1.corr()

a=pd.DataFrame(corr1[Y.columns].T.abs().max()>0.4)

aa=train[a[a[0]==False].index.tolist()].drop('cp_time',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=42)

pca = PCA(n_components=100)

X_train_pca=pd.DataFrame(pca.fit_transform(X_train[aa.columns]),index=X_train.index)

X_train=pd.concat([X_train.drop(aa.columns,axis=1),X_train_pca],axis=1)

X_test=pd.concat([X_test.drop(aa.columns,axis=1),pd.DataFrame(pca.transform(X_test[aa.columns]),index=X_test.index)],axis=1)

test2=pd.concat([test.drop(aa.columns,axis=1),pd.DataFrame(pca.transform(test[aa.columns]),index=test.index)],axis=1)
params1 = {'bootstrap': True,

 'max_depth': 10,

 'max_features': 2,

 'min_samples_leaf': 1,

 'min_samples_split': 2,

 'n_estimators': 25}

params2 = {'bootstrap': False,

 'max_depth': 80,

 'max_features': 3,

 'min_samples_leaf': 2,

 'min_samples_split': 2,

 'n_estimators': 50}
def pred_output(X_train, X_test, Y_train, Y_test, test):

    try :

        test=test.drop('sig_id',axis=1)

        cet=TargetEncoder(cols=['cp_type','cp_time','cp_dose'])

        X_train=cet.fit_transform(X_train,Y_train).drop(['cp_type','cp_time','cp_dose'],axis=1)

        X_test=cet.transform(X_test).drop(['cp_type','cp_time','cp_dose'],axis=1)

        test=cet.transform(test).drop(['cp_type','cp_time','cp_dose'],axis=1)

        if Y_train.sum()>300:

            params=params2

        else : 

            params=params1

        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split']

                                     , min_samples_leaf=params['min_samples_leaf'], bootstrap = params['bootstrap'], max_features=params['max_features'])

        clf.fit(X_train, Y_train)

        Y_submit = clf.predict_proba(test)[:,1]

        return(Y_submit,log_loss(Y_test,clf.predict_proba(X_test)[:,1]))

    except (ValueError, IndexError):

        return(np.zeros(test.shape[0]),np.nan)
score=[]

j=0

for i in scored.drop('sig_id',axis=1).columns:

    print(i)

    j=j+1

    print(j)

    output, m= pred_output(X_train, X_test, Y_train[i], Y_test[i], test2)

    score.append(m)

    submit[i]=output
submit.to_csv('submission.csv', index=False)
train[train.sig_id=='id_000644bb2']
X
sum(scored.sum()==0)
sum(scored.sum()==0)
params1['bootstrap']