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
people_df = pd.read_csv('../input/people.csv')

act_train_df = pd.read_csv('../input/act_train.csv')



print(people_df.head())

print(act_train_df.head())
#nos = act_train_df.groupby('outcome').count()

#print(nos)

#print(act_train_df['outcome'].value_counts())

train_type1_df = act_train_df[act_train_df['activity_category']!='type 1']

#print(train_type1_df.head())

#print(train_type1_df.shape[0])



print(train_type1_df['outcome'].value_counts())

print(act_train_df['outcome'].value_counts())

out_char10 = pd.crosstab(index=train_type1_df["outcome"], columns=train_type1_df["char_10"])

print(out_char10)
'''train_transform = train_type1_df[["outcome","char_10"]]

print(train_transform.head())

train_transform = pd.get_dummies(train_transform,columns=["char_10"])

print(train_transform.head())'''

from scipy.stats import chi2_contingency

chi2_contingency(out_char10)
