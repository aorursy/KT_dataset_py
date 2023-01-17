# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set(color_codes=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/museum-directory/museums.csv', low_memory=False)
# We'll look at the head in order to understand labels of the data. 

df.head()
# Let's see for the general information about the data. 

#It seems, there are a lot of missing values in columns such as Institution Name column.



df.info()
df.describe()
#Let's look at the statistical values from a differnet perspective. 

df.describe(include='O').transpose()
#Let's check the check the shape of the dataframe.

df.shape
#Let's get the column names' first



df.columns
#Now, let's drop the columns.



df = df.drop(['Museum ID','Legal Name', 'Alternate Name', 'Street Address (Physical Location)', 'City (Physical Location)',

       'State (Physical Location)', 'Phone Number', 'Employer ID Number'], axis=1)



# And check the head again. 



df.head()
# Let' check the shape of dataframe again. 



df.shape
# Let's remove the number of rows before removing the duplicates.



df = df.drop_duplicates()
# Let's count the number of rows after removing the duplicates.



df.count()
# The next thing is to identify the number of missing values. 

#According to our dataframe, columns such as Institution Name or 

#Zip Code (Physical Location) have the most missing values.



df.isnull().sum()
# Let's find the columns, which have more than 50% missing values.



most_missing_cols = set(df.columns[df.isnull().mean() > 0.50])



most_missing_cols
# In this case let's drop the "Institution Name" and Zip Code (Physical Location) columns, since we believe that 

#ommision of these columns will not much differnce to our analysis. 



df = df.drop(['Institution Name', 'Zip Code (Physical Location)'], axis=1)
# Let's check the head of our dataframe again to make sure that the last two columns have been dropped.



df.head()
# The next step is to drop the missing values. 



df = df.dropna()

df.count()
# Let's check the columns with 0 missing values



no_nulls = set(df.columns[df.isnull().mean()==0])

no_nulls
# When we plot the Revenue column with 0s included, the mean is close to zero and almost invisible. 

sns.boxplot(x=df['Revenue'], showfliers=False);
# Now, we are going to filter out the rows that have 0 values and assign the result to a new variable.

no_zeros = df[df['Revenue']!=0]
# Let's look at the spread of the revenue with 0s filtered out. As you can observe the mean, 

#even though slightly, has moved to the right. 

sns.boxplot(x=no_zeros['Revenue'] , showfliers=False);
#In terms of normal distribution, we have a right skewed histogram. 

fig, ax = plt.subplots()

ax.hist(x=no_zeros['Revenue'], bins =2)

ax.set(title = 'Normal distribution of museum revenues');
# Let's count the number of museum types available in the dataframe.



df['Museum Type'].value_counts().nlargest(20).plot(figsize=(10,5), kind = 'bar')

plt.title("Number of museum types in the United States")

plt.ylabel("Number of museums")

plt.xlabel("Types of museums");
#Let's see what type of museums attract the most visitors in terms of revenue.



type_rev = df.groupby(['Museum Type']).agg({'Revenue':'sum'})

type_rev = type_rev.sort_values(by='Revenue', ascending=False)

type_rev.plot(kind='bar', figsize=(10,5));
# How about the states that earn the majority of Revenues?

state_rev = df.groupby(['State (Administrative Location)']).agg({'Revenue':'sum'})

state_rev = state_rev.sort_values(by='Revenue', ascending=False)

state_rev[:20].plot(kind='bar', figsize=(10,5))

plt.ylabel('Revenue (million $)');
# Let's see the top grossing museums in terms of their city locations. 

city_rev = df.groupby(['City (Administrative Location)']).agg({'Revenue':'sum'})

city_rev = city_rev.sort_values(by='Revenue', ascending=False)

city_rev[:20].plot(kind='bar', figsize=(10,5))

plt.ylabel('Revenue (million $)');
# In this section we are going to create a column based on revenue 

#column in order to rank the museums in terms of their revenues. 



cols = ['Revenue']



no_zeros['Rank'] = no_zeros.sort_values(cols, ascending=False).groupby(cols, sort=False).ngroup()+1

no_zeros.head()
no_zeros.sort_values('Rank')
#Here we can see that the Museums, that are sorted by their revenues. When we compare with the Rank column

#we see different museum names. It's probably due to the rank method.



museum_rev = df[['Museum Name','Revenue']].sort_values(by='Revenue', ascending=False)

museum_rev.head()