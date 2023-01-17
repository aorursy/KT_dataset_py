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
import numpy as np # linear algebra

import pandas as pd # data processing

import os

import matplotlib.pyplot as plt

import seaborn as sns
# load the csv

df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
# lets get the glimpse of the data by fetching first 5 records

df.head()
# lets see the the range of suicide data by years available to us using box plot

sns.set(style='whitegrid')

plt.figure(figsize=(15,5))

ax = sns.boxplot(df.year, palette='YlGnBu',)

ax.set(xlim=[1990, 2020])

plt.show()
# lets change the age range years to age group for better understanding

df.age.unique()
age_groups = ['child', 'youth', 'young adult', 'early adult', 'adult', 'senior']

age_ranges = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']





 # range will be 6 as we have 6 groups

for i in range(6): 

    df["age"]=df["age"].apply(lambda x: str(x).replace(age_ranges[i],age_groups[i]) if age_ranges[i] in str(x) else str(x))
df.head()
df.shape # this will tell us how may rows and column we have in total (rows, columns)
df.isnull().any() # this will show True if any column has null values
df.isnull().sum() # as HDI for year shows true lets sum up the null values and see how many in total null values do we have
# As we can see 70% of the values are null in *HDI for year* column so we better drop it from our dataframe 

# along wih *country-year* as this column just has Country name The year concatenated 

df=df.drop(['HDI for year','country-year'],axis=1) 
###Let's check for country

sns.set_style("whitegrid")

alpha = 1

plt.figure(figsize=(12,30))

sns.countplot(y='country', data=df, alpha=alpha, linewidth=2, palette="YlGnBu")

plt.title('Suicide counts by country')

plt.show()
# to see the gender distribution

plt.figure(figsize=(12,7))

sex = sns.countplot(x='sex',data = df, palette="YlGnBu")
# as we can see above gender is evenly distributed so lets see how gender is distributed by age group 

plt.figure(figsize=(12,7))

sex = sns.countplot(y='sex', hue="age", data = df, palette="YlGnBu")
# to see the Correlation between each feature

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12,7))

cor = sns.heatmap(df.corr(),mask=mask, annot = True, cmap='YlGnBu')
# lets see suicides commited by age group and gender

plt.figure(figsize=(16,7))

sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = df, palette="YlGnBu")
# Lets see the suicide numbers for people of different age groups for every year



age_groups = ['child', 'youth', 'young adult', 'early adult', 'adult', 'senior']

plt.figure(figsize=(16,7))

for age in age_groups:

    data = df.loc[df.age==age,:]

    sns.lineplot(x='year', y='suicides_no', data=data)



plt.legend(age_groups)