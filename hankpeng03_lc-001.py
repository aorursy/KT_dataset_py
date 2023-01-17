# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.metrics import r2_score

from sklearn.ensemble import  BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn import svm

from sklearn.model_selection import  cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import  train_test_split

from sklearn.tree import DecisionTreeClassifier
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

index=list(df_train.columns)

train_data=df_train[index[2:]]

train_label=df_train[index[1]]

del df_test['id']

test_data=df_test
#summarize the missing data 

train_data.isnull().sum()

#bulid BaggingClassifier model

base_estimator_bag=DecisionTreeClassifier(criterion='gini',splitter='best',max_features=None,

                                     max_depth=None,min_samples_split=2,min_samples_leaf=1,

                                      min_impurity_split=1e-7)

clf_bag=BaggingClassifier(n_estimators=50,base_estimator=base_estimator_bag,max_samples=1.0,max_features=0.7,bootstrap=True,

                          bootstrap_features=False,oob_score=True,n_jobs=-1,random_state=22)

#build AdaBoostingClassifier model

base_estimator_ada=svm.SVC(kernel='rbf',gamma='auto',shrinking=True)

clf_ada=AdaBoostClassifier(base_estimator=base_estimator_ada,n_estimators=50,algorithm='SAMME',

                           random_state=44,learning_rate=0.1)

#build GradientBoostingClassifier

clf_GBCT=GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=50,

                                    max_depth=4,subsample=0.7,max_features=0.7,random_state=55)

eclf=VotingClassifier([('bag',clf_bag),('ada',clf_ada),('GBCT',clf_GBCT)],voting='hard')

#a=cross_val_score(estimator=eclf,X=train_data,y=train_label,cv=5,n_jobs=-1)
eclf.fit(X=train_data,y=train_label)
#the prediction

pred=eclf.predict(test_data)