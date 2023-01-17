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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# pd.read_csv(name,header, index_col)
#header - 0 wskazuje że zerowy wiersze będzie nam wskazywał nazwy kolumn

train_df = pd.read_csv('../input/train.csv', header = 0, index_col = 0)
test_df = pd.read_csv('../input/test.csv', header = 0, index_col = 0)
# concatenate two dataframes

full = pd.concat([train_df , test_df], sort=True)
full.info
full.head()
# Pierwszych 10 elementów
full[:10] 
# Ostatnich 5 elementów
full[-5:] 
# [od : do : co ile]
SURV = 890
full[SURV:SURV+11:3] # Like in regular Python you can get to the Item by Index
#filter data by columns
#full[ (full['Age'] > 8.0) & (full['Age'] <= 10.0 ) ] 

# filter and sort by Age
full[ (full['Age'] > 8.0) & (full['Age'] <= 10.0 ) ].sort_values('Age')
#filter data by columns

#full[(full['Cabin'].str.contains('B2',na=False)) ] 

full[(full['Cabin'].str.contains('B2',na=False)) & full['Sex'].str.contains('f')] 


# is important to find null rows.
# .isnull()
full.isnull()

full.isnull().sum()  # sum of null rows fo each column
# null values matrix

import missingno as msno
msno.matrix(full)
# group_by

#train_df.groupby(['Pclass','Sex'])['Survived'].sum() 
#train_df.groupby(['Sex','Pclass'])['Survived'].sum() 
train_df.groupby(['Sex','Embarked'])['Survived'].sum() 
