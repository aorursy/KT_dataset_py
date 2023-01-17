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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# read test file
test=pd.read_csv('/kaggle/input/adulttest/adult.test',header=None)
test.head()
#Take a look at the data
df.describe()
#all Columns
columns=df.columns
np.sum(df['workclass']=='?')

#get all the columns where '?' missing value is present and store them to missing_val_column
missing_val_column=[]
for col in columns:
    if np.sum(df[col]=='?')>0:
        missing_val_column.append(col)

missing_val_column

# so in ['workclass', 'occupation', 'native.country'] we got missingf values
# creating a dataframe with missing values
missing_df=df[['workclass', 'occupation', 'native.country']]
missing_df.head()
df['workclass'].value_counts()

# there are 1836 number of missing values

df.groupby(['workclass'])['income'].value_counts()

# as most missing value  earns less than <=50k and same for Private we replace missing value '?' with 'Private'
# replace '?' with 'Private'
df['workclass'].replace('?','Private',inplace=True)
df['workclass'].value_counts()
# Now the missing values of occupation
df['occupation'].value_counts()

# There are 1843 number of missing value

df.groupby(['occupation'])['income'].value_counts()

# it gives not much insights as most of them kind of similar
# Try occuption against education.num

df.groupby(['occupation'])['education.num'].mean()
missing_indices=df[df['occupation']=='?'].index.tolist()
# In this index there are missing values in occupation
missing_indices
# as these missing values are close to every other occuption then i will
# divide these missing values to other occuption ['Adm-clerical','Craft-repair','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Other-service']
replace_category=['Adm-clerical','Craft-repair','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Other-service']
division=int(df.shape[0]/len(replace_category))
division
for i in range(len(replace_category)):
    df.loc[:int((i+1)*division),'occupation'].replace('?',replace_category[i],inplace=True)
    

# df.loc[:int(division),'occupation'].replace('?',replace_category[0],inplace=True)
df['occupation'].value_counts()
df
#Now handle native.country missing value
df['native.country'].value_counts()

# There are 583 missing values
df.groupby(['native.country'])['income'].value_counts()

# we will replace the missing value by 'United-States'
df.loc[df['native.country']=='?','native.country']="United-States"
df['native.country'].value_counts()

df.describe()
# We need to normalize the data
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
