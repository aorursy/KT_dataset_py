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
import pandas as pd
train=pd.read_csv(r"../input/riiid-test-answer-prediction/train.csv",nrows=2*(10**4))
train
train.isna().sum()
mean=train.prior_question_elapsed_time.mean()
mean
import math

Mean=math.floor(mean)

Mean
train.prior_question_elapsed_time=train.prior_question_elapsed_time.fillna(Mean)
train
train.prior_question_had_explanation.value_counts()
train.prior_question_had_explanation=train.prior_question_had_explanation.fillna(True)
train
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
train['prior_question_had_explanation']=lb.fit_transform(train['prior_question_had_explanation'])
train
train.prior_question_had_explanation.value_counts()
Train=train[train.answered_correctly!= -1]
x=Train.drop(['row_id','user_answer','answered_correctly'],axis='columns')
y=Train.answered_correctly
from sklearn.model_selection import train_test_split

x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.1,random_state=42)
import lightgbm as lgb

params={

    'objective' : 'binary',

    'max_bin' : 26445,

    'learning_rate' : 0.1,

    'num_leaves' : 4095,

    'max_depth' : 12,

    'feature_fraction': 0.25,

    'boosting' : 'gbdt',

    'lambda_l1': 0.5,

    'bagging_fraction': 0.6,

    'bagging_freq': 423

    

}
lgb_train = lgb.Dataset(x_Train,y_Train)

lgb_test = lgb.Dataset(x_Test, y_Test, reference=lgb_train)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_test],

    verbose_eval=10,

    num_boost_round=10000,

    early_stopping_rounds =10

)
from sklearn.metrics import roc_auc_score

y_pred=model.predict(x_Test)

y_true=np.array(y_Test)

roc_auc_score(y_true,y_pred)
import riiideducation

env=riiideducation.make_env()

iter_test=env.iter_test()

for (test_df,sample_prediction_df) in iter_test:

    test_df['prior_question_elapsed_time']= test_df.prior_question_elapsed_time.fillna(25475)

    test_df['prior_question_had_explanation']= test_df.prior_question_had_explanation.fillna(True)

    test_df['prior_question_had_explanation']= lb.fit_transform(test_df['prior_question_had_explanation'])

    test_df['answered_correctly']= model.predict(test_df[['timestamp','user_id','content_id','content_type_id','task_container_id','prior_question_elapsed_time','prior_question_had_explanation']])

    env.predict(test_df.loc[test_df['content_type_id']==0,['row_id','answered_correctly']])