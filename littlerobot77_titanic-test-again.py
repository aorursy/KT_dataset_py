# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")





# Any results you write to the current directory are saved as output.
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = ((df_train.isnull().sum()/(df_train.count()+df_train.isnull().sum()))*100).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent Missing'])

missing_data.head(20)

#77% of Cabin data is missing, so we might as well drop it

#it won't have a big enough impact on the outcome and trying to 

# guess the other values isn't worthwhile

df_train.drop("Cabin",axis=1,inplace=True)
#fill missing age values with the average age

averageAge = df_train['Age'].sum()/df_train['Age'].count()

df_train['Age'].fillna(averageAge,inplace=True)



print(df_train['Age'])





df_train[df_train.dtypes[(df_train.dtypes=="float64")|(df_train.dtypes=="int64")]

                        .index.values].hist(figsize=[11,11])