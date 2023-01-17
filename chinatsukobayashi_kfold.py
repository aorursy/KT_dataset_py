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
gender_submissison=pd.read_csv("../input/titanic/gender_submission.csv")

train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
print(train.shape,test.shape)
#import pandas_profiling as pdp



#pdp.ProfileReport(train)
train=pd.get_dummies(train,columns=["Sex","Embarked"])

test=pd.get_dummies(test,columns=["Sex","Embarked"])

train.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)

test.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)
#train["Cabin1st"] = train["Cabin"].str[:1]

#train.loc[:,"Cabin1st"]=train.loc[:,"Cabin1st"].fillna("U")

#test["Cabin1st"] = test["Cabin"].str[:1]

#test.loc[:,"Cabin1st"]=test.loc[:,"Cabin1st"].fillna("U")



#train=pd.get_dummies(train,columns=["Cabin1st"])

#test=pd.get_dummies(test,columns=["Cabin1st"])



train.drop(["Cabin"],axis=1,inplace=True)

test.drop(["Cabin"],axis=1,inplace=True)

x_train=train.drop(["Survived"],axis=1)

y_train=train["Survived"]



from sklearn.model_selection import train_test_split

train_x,valid_x,train_y,valid_y=train_test_split(x_train,y_train,test_size=0.33,random_state=0)
train_x.head()
import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold



kf=KFold(n_splits=5)



score_list=[]

models=[]



for fold_, (train_index,valid_index) in enumerate(kf.split(x_train,y_train)):

    train_x=x_train.iloc[train_index]

    valid_x=x_train.iloc[valid_index]

    train_y=y_train[train_index]

    valid_y=y_train[valid_index]

    

    print(f'fold{fold_+1} start')

    

    gbm=lgb.LGBMClassifier(objective="binary")

    gbm.fit(train_x,train_y,eval_set=[(valid_x,valid_y)],early_stopping_rounds=20,verbose=-1)

    

    oof = gbm.predict(valid_x,num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y,oof)*100,2))

    models.append(gbm)

    print(f'fold{fold_+1}end\n')

    

print(score_list,"平均score",np.mean(score_list))
test_pred=np.zeros((len(test),5))



for fold_, gbm in enumerate(models):

    pred_=gbm.predict(test,num_iteraion=gbm.best_iteration_)

    test_pred[:,fold_]=pred_

    

pred=(np.mean(test_pred,axis=1)>0.5).astype(int)
