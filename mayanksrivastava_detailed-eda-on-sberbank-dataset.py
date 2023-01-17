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
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import mode



# Set ipython's max row display

pd.set_option('display.max_row', 10000)

#Setting to print all the values in array

np.set_printoptions(threshold=np.nan)

# Set iPython's max column width to 50

pd.set_option('display.max_columns', 500)
#Import  Dataset for EDA

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')



train['price_doc'].head()
#Identify the columns with missing Values

total = train.isnull().sum().sort_values(ascending=False)

total.columns = ['column_name', 'missing_count']

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.loc[missing_data['Total']!= 0]
train['price_doc'].describe()
np.log(train['price_doc']).skew()
train['price_doc'].kurtosis()
sns.distplot(train['price_doc'], color = 'g', bins = 100)
sns.distplot(np.log(train['price_doc']), color = 'g', bins = 100, kde = 'True')
train.plot.scatter(x = 'full_sq', y = 'price_doc')
train.plot.scatter(x = 'life_sq', y = 'price_doc')