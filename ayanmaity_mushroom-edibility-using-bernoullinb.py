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
df = pd.read_csv('../input/mushrooms.csv')

df.head()
df.shape
train_df = df[:7000]

test_df = df[7000:]
edible = train_df['class']=='e'

poison = train_df['class']=='p'

target = train_df['class']

target.value_counts(normalize=1)
del train_df['class']

cols = list(train_df)
for f in cols :

    for elem in df[f].unique():

        train_df[f+'_'+str(elem)] = (train_df[f]==elem)

    ##train_df = train_df.drop([f],inplace=True,axis=1)

for f in cols:

    del train_df[f]

train_df.head()
from sklearn.naive_bayes import BernoulliNB

clf_ber = BernoulliNB()

train_x = train_df.as_matrix()

clf_ber.fit(train_x,target)

clf_ber.score(train_x,target)
test_y = test_df['class']

del test_df['class']

for f in cols :

    for elem in df[f].unique():

        test_df[f+'_'+str(elem)] = (test_df[f]==elem)

for f in cols:

    del test_df[f]
test_x = test_df.as_matrix()

clf_ber.score(test_x,test_y)