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
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')



all_data = pd.concat([train,test], sort=False)
print(train.shape)

pd.set_option('display.max_columns', 100) # display max 100 columns 

train.head()

print(test.shape)

test.head()
all_data.info()
all_data.dtypes.sort_values()
# All data type int

all_data.select_dtypes(include='int').head()
# All data type float

all_data.select_dtypes(include='float').head()
# All data type object

all_data.select_dtypes(include='object').head()
# All columns with columns with one null or more

missingPercent = (all_data.isnull().sum()[all_data.isnull().sum() > 0]/all_data.shape[0]*100).round(decimals=2) # missing count to %

missingPercentSorted = missingPercent.sort_values(ascending=False)

missingPercentSorted
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(25,8))

plt.title('Number of missing rows')

missing_count = pd.DataFrame(missingPercent, columns=['percent']).sort_values(by=['percent'],ascending=False).head(20).reset_index()

missing_count.columns = ['features','percent']

sns.set_style("whitegrid")

sns.barplot(x='features',y='percent', data = missing_count)
# The features with most missing values

missingPercentSorted[missingPercentSorted>10]