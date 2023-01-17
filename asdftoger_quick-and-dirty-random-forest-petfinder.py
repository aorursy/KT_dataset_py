# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train/train.csv')
df.head()
df.info()
df_obj=df.dtypes[df.dtypes=='object'].index
df_num=df.drop(columns=df_obj)
pd.isnull(df_num).mean()
for i in df_num.columns:
    print('Feature: "{}", # unique: {}. '.format(i,df_num.loc[:,i].nunique()),df_num.loc[:,i].unique())
from sklearn.model_selection import train_test_split
RANDOM_SEED=142
np.random.seed(RANDOM_SEED)
def xy_split(df,y_cols):
    y=df.loc[:,y_cols]
    x=df.drop(columns=y_cols)
    return x,y
x,y=xy_split(df_num,'AdoptionSpeed')
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.1,random_state=RANDOM_SEED)
from sklearn.ensemble import RandomForestClassifier
m=RandomForestClassifier(n_estimators=100,max_features='log2')
%time m.fit(x_train,y_train)
m.score(x_valid,y_valid)

df_test=pd.read_csv('../input/test/test.csv')
df_test.head()
df_test.info()
x_test=df_test.loc[:,x_train.columns]
x_test.info()
y_pred=m.predict(x_test)
sub=df_test.loc[:,['PetID']].copy()
sub.loc[:,'AdoptionSpeed']=y_pred
sub.to_csv('submission.csv',index=False)
sub.loc[:,'AdoptionSpeed']=2
sub.to_csv('submission_g2.csv',index=False)