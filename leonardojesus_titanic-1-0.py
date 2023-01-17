# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.



df_train = pd.read_csv('../input/train.csv', index_col='PassengerId')

df_test = pd.read_csv('../input/test.csv', index_col='PassengerId')



# Any results you write to the current directory are saved as output.
#Preparing the tables



target = df_train['Survived']  #target variable

df_train = df_train.drop('Survived', axis=1)

df_train['training_set'] = True

df_test['training_set'] = False

#dropping irrelevant columns

df_train = df_train.drop('Name', axis=1)

df_train = df_train.drop('Ticket', axis=1)

df_test = df_test.drop('Name', axis=1)

df_test = df_test.drop('Ticket', axis=1)
#Filling in the missing values and converting categorical features to numerical ones



df_full = pd.concat([df_train, df_test])

df_full = df_full.interpolate()   

df_full = pd.get_dummies(df_full)   
#Separating tables again



df_train = df_full[df_full['training_set']==True]

df_train = df_train.drop('training_set', axis=1)



df_test = df_full[df_full['training_set']==False]

df_test = df_test.drop('training_set', axis=1)
#Training



rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

rf.fit(df_train, target)
#Results



preds = rf.predict(df_test)

my_submission = pd.DataFrame({'PassengerId': df_test.index, 'Survived': preds})

#my_submission.to_csv('../input/submission.csv', index=False)
my_submission.to_csv('submission.csv', index=False)