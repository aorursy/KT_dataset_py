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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head(10)
df_train.columns
df_test.columns
df = pd.concat([df_train.drop(['Survived'],axis=1),df_test])
df
df['Cabin'].isnull().values.any()

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Survived',data=df_train)
print('% of Not Survived : ', df_train['Survived'].value_counts()[0]/len(df_train)*100)
print('% of Survived : ', df_train['Survived'].value_counts()[1]/len(df_train)*100)
df_test['Survived']=0
df_test[['PassengerId','Survived']].to_csv('Null_Accuracy_Submission.csv',index=False)

