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
#Importing some useful libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import csv
#Reading in the training data 
filename = "../input/titanic/train.csv"
df_train = pd.read_csv(filename)
df_train.head(10)
df_train.info()
df_train.describe()
df_train.describe(include=np.object)
is_ticket =  df_train['Ticket']=="CA. 2343"
df_train_ticket = df_train[is_ticket]
print(df_train_ticket)

df_train_no_embarked = df_train[df_train['Embarked'].isnull()]
print(df_train_no_embarked)
is_pclass_1 =  df_train['Pclass']==1
df_train_pclass_1 = df_train[is_pclass_1]
print(df_train_pclass_1)
df_train_pclass_1['Embarked'].describe()
df_train['Embarked'] = df_train['Embarked'].replace(np.NaN,'S')
df_train_no_embarked = df_train[df_train['Embarked'].isnull()]
print(df_train_no_embarked)

df_train.groupby('Ticket').filter(lambda g: len(g) > 1).groupby(['Ticket', 'Cabin']).head(1)

is_ticket =  df_train['Ticket']=="19950"
df_train_ticket = df_train[is_ticket]
print(df_train_ticket)
