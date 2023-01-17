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
#for playing data

import numpy as np

import pandas as pd



# for visualizations

import seaborn as sns

import matplotlib.pyplot as plt



# for missing value visualizations

import missingno as msno
data = pd.read_csv('/kaggle/input/google-job-skills/job_skills.csv')

data.head()
data.shape
data.info()
# df.isna() shows NaN value True or False

# df.isna().sum() shows NaN value with column name



print("Missing value: ",data.isnull().sum().sum())
# missing value visualizations -1



# All columns are 1250 but one of them 1235 and two of them 1236

msno.bar(data);
# missing value visualizations we can see white areas are missing and right corners ping 4 number

msno.matrix(data);
cat_col = data['Category'].value_counts()

cat_col
cat_col.plot(kind='barh', figsize=(20,10))

plt.title("What is the most popular job department at Google?", fontsize=20)

plt.show()
Loc_col = data['Location'].value_counts()

Loc_col
Loc_col.head(10).plot(kind='barh', color='C4' , figsize=(20,10))

plt.title("What is the most 10 location at Google?", fontsize=20)

plt.show()
sns.catplot(x="Company",data=data, kind="count")

plt.title('Names of companies', fontsize = 20)

plt.show()
title_col = data['Title'].value_counts()

title_col
title_col.head(10).plot(kind='barh', figsize=(20,10), color='C1')

plt.title("What is the most popular 20 job Title at Google?", fontsize=20)

plt.show()