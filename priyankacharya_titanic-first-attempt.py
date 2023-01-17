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
# Read the training data\n

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.info()
# Identify how many passgengers of what gender survived

df_train.pivot_table(index='Sex',columns='Survived',values='PassengerId',aggfunc='count').merge(df_train.pivot_table(index='Sex',columns='Survived',values='PassengerId',aggfunc='count').divide([314,577],axis=0)*100,

                                                                                                   left_on='Sex',right_on='Sex',suffixes=('_count','_per'))
# Use these percentages to randomly assign Survival flag for each gender

df_train_female = df_train.query('Sex=="female"')

df_train_male = df_train.query('Sex=="male"')

df_train['survived_p'] = np.zeros((891,1))

df_train_final = df_train.copy()

female_index = df_train_female.sample(233).index

male_index = df_train_male.sample(109).index

df_train_final.loc[female_index,'survived_p']=1

df_train_final.loc[male_index,'survived_p']=1

df_train_final.pivot_table(index='Sex',columns='survived_p',values='PassengerId',aggfunc='count')

np.unique(df_train_final.survived_p==df_train_final.Survived,return_counts=True)

print("The accuracy score is:",587*100/891)
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.info()

df_test.pivot_table(index='Sex',values='PassengerId',aggfunc='count')

print(152*0.74,266*0.19)
df_test_female = df_test.query('Sex=="female"')

df_test_male = df_test.query('Sex=="male"')

df_test['survived_p'] = np.zeros((418,1))

df_test_final = df_test.copy()

female_index = df_test_female.sample(112).index

male_index = df_test_male.sample(51).index

df_test_final.loc[female_index,'survived_p']=1

df_test_final.loc[male_index,'survived_p']=1

df_test_final.pivot_table(index='Sex',columns='survived_p',values='PassengerId',aggfunc='count')

df_test_final[['PassengerId','survived_p']].to_csv("my_submission_1.csv",index=False)