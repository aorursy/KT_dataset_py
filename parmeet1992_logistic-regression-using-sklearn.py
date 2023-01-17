# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test"))

#!unzip ../input/test/

# Any results you write to the current directory are saved as output.

#!pip install pyspark
df = pd.read_csv('../input/train/train.csv')

df_test = pd.read_csv('../input/test/test.csv')
cols = [a for a,b in zip(df.columns,df.dtypes) if b=='int64']

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(df[cols[:-1]],df[cols[-1]])

predictions = lr.predict(df[cols[:-1]])
df_new = pd.DataFrame()

df_new['PetID'] = df['PetID']

df_new['AdoptionSpeed'] = predictions
df_new.to_csv('submission.csv',index=False)
df_new.head()