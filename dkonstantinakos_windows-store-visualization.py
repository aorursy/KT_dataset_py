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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/windows-store/msft.csv')
df.head()
df.info()
df.isnull()
df = df.drop([5321])
df.tail()
df_free_apps = df[df['Price'] == 'Free']

df_no_free_apps = df[df['Price'] != 'Free']
len(df_free_apps)
len(df_no_free_apps)
df_free_apps
df_no_free_apps
df_no_free_apps = df_no_free_apps.reset_index()

df_no_free_apps
for i in range(len(df_no_free_apps)):

    df_no_free_apps['Price'][i] = df_no_free_apps['Price'][i].replace(',', '')
for i in range(len(df_no_free_apps)):

    df_no_free_apps['Price'][i] = df_no_free_apps['Price'][i][2:6]
df_no_free_apps['Price'] = df_no_free_apps['Price'].astype(float)
df_no_free_apps.info()
df_no_free_apps = df_no_free_apps.drop(['index'], axis  = 1)

df_no_free_apps
df_no_free_apps['Date'] = pd.to_datetime(df_no_free_apps['Date'])

df_free_apps['Date'] = pd.to_datetime(df_free_apps['Date'])
df_no_free_apps.info()
df_free_apps.info()
df_no_free_apps['Month'] = df_no_free_apps['Date'].dt.month
df_free_apps['Month'] = df_free_apps['Date'].dt.month
df_no_free_apps['Year'] = df_no_free_apps['Date'].dt.year

df_free_apps['Year'] = df_free_apps['Date'].dt.year
df_free_apps.info()
df_no_free_apps.info()
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)

sns.countplot(x = 'Rating', data = df_free_apps)

plt.title('Rating of free apps')





plt.subplot(1, 2, 2)

sns.countplot(x = 'Rating', data = df_no_free_apps)

plt.title('Rating of non-free apps')
plt.figure(figsize = (12, 6))

sns.distplot(df_no_free_apps['Price'], kde = False, bins = 50)

plt.title('Distribution of prices for non-free apps')
plt.figure(figsize = (60, 6))

plt.subplot(1, 2, 1)

sns.countplot(x = 'Category', data = df_free_apps)

plt.title('Categories of free apps')



plt.figure(figsize = (20, 6))

plt.subplot(1, 2, 2)

sns.countplot(x = 'Category', data = df_no_free_apps)

plt.title('Categories of non-free apps')
category_no_free_apps = df_no_free_apps.groupby(['Category']).mean().drop(['Month', 'Year'], axis = 1)

category_free_apps = df_free_apps.groupby(['Category']).mean().drop(['Month', 'Year'], axis = 1)
category_free_apps
category_no_free_apps
category_free_apps['Rating'] = category_free_apps['Rating'].round(2)

category_free_apps['No of people Rated'] = category_free_apps['No of people Rated'].round()



category_no_free_apps['Rating'] = category_no_free_apps['Rating'].round(2)

category_no_free_apps['No of people Rated'] = category_no_free_apps['No of people Rated'].round()

category_no_free_apps['Price'] = category_no_free_apps['Price'].round()
category_free_apps
category_no_free_apps
plt.figure(figsize = (20, 6))

plt.subplot(1, 3, 1)

category_no_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of non-free apps per category')





plt.subplot(1, 3, 2)

category_no_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people rated non-free apps per category')





plt.subplot(1, 3, 3)

category_no_free_apps['Price'].plot(kind = 'bar')

plt.title('Price of non-free apps per category')
plt.figure(figsize = (40, 6))

plt.subplot(1, 2, 1)

category_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of free apps per category')



plt.figure(figsize = (40, 6))

plt.subplot(1, 2, 2)

category_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people rated free apps per category')
plt.figure(figsize = (20,6))

plt.subplot(1, 2, 1)

sns.countplot(df_no_free_apps['Month'])

plt.title('Number of non-free apps released per month')



plt.subplot(1, 2, 2)

sns.countplot(df_no_free_apps['Year'])

plt.title('Number of non-free apps released per year')
plt.figure(figsize = (20,6))

plt.subplot(1, 2, 1)

sns.countplot(df_free_apps['Month'])

plt.title('Number of free apps released per month')



plt.subplot(1, 2, 2)

sns.countplot(df_free_apps['Year'])

plt.title('Number of free apps released per year')
month_free_apps = df_free_apps.groupby(['Month']).mean().drop(['Year'], axis = 1)

year_free_apps = df_free_apps.groupby(['Year']).mean().drop(['Month'], axis = 1)



month_no_free_apps = df_no_free_apps.groupby(['Month']).mean().drop(['Year'], axis = 1)

year_no_free_apps = df_no_free_apps.groupby(['Year']).mean().drop(['Month'], axis = 1)
month_free_apps['Rating'] = month_free_apps['Rating'].round(2)

month_free_apps['No of people Rated'] = month_free_apps['No of people Rated'].round()



year_free_apps['Rating'] = year_free_apps['Rating'].round(2)

year_free_apps['No of people Rated'] = year_free_apps['No of people Rated'].round()



month_no_free_apps['Rating'] = month_no_free_apps['Rating'].round(2)

month_no_free_apps['No of people Rated'] = month_no_free_apps['No of people Rated'].round()

month_no_free_apps['Price'] = month_no_free_apps['Price'].round()



year_no_free_apps['Rating'] = year_no_free_apps['Rating'].round(2)

year_no_free_apps['No of people Rated'] = year_no_free_apps['No of people Rated'].round()

year_no_free_apps['Price'] = year_no_free_apps['Price'].round()
plt.figure(figsize = (16, 6))

plt.subplot(1, 2, 1)

month_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of free apps released per month')



plt.subplot(1, 2, 2)

month_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people that rated free apps released per month')
plt.figure(figsize = (16, 6))

plt.subplot(1, 2, 1)

year_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of free apps released per year')



plt.subplot(1, 2, 2)

year_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people that rated free apps released per year')
plt.figure(figsize = (22, 6))

plt.subplot(1, 3, 1)

month_no_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of non-free apps released per month')



plt.subplot(1, 3, 2)

month_no_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people that rated non-free apps released per month')



plt.subplot(1, 3, 3)

month_no_free_apps['Price'].plot(kind = 'bar')

plt.title('Price of non-free apps released per month')
plt.figure(figsize = (22, 6))

plt.subplot(1, 3, 1)

year_no_free_apps['Rating'].plot(kind = 'bar')

plt.title('Rating of non-free apps released per year')



plt.subplot(1, 3, 2)

year_no_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people that rated non-free apps released per year')



plt.subplot(1, 3, 3)

year_no_free_apps['Price'].plot(kind = 'bar')

plt.title('Price of non-free apps released per year')
plt.figure(figsize = (16,10))

corr = df_free_apps.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask, annot = True, cmap = 'viridis')

plt.title('Heatmap of correlation between variables for free apps')
plt.figure(figsize = (16,10))

corr = df_no_free_apps.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask, annot = True, cmap = 'viridis')

plt.title('Heatmap of correlation between variables for non-free apps')
plt.figure(figsize = (12,6))

sns.boxplot(x = 'Rating', y = 'No of people Rated', data = df_no_free_apps)

plt.title('Distribution of number of people that rated apps and their ratings')
plt.figure(figsize = (12,6))

sns.boxplot(x = 'Rating', y = 'Price', data = df_no_free_apps)

plt.title('Distribution of prices of  apps and their ratings')
plt.figure(figsize = (12,6))

sns.boxplot(x = 'Rating', y = 'No of people Rated', data = df_free_apps)

plt.title('Distribution of number of people that rated apps and their ratings (free apps)')
rating_no_free_apps = df_no_free_apps.groupby(['Rating']).mean().drop(['Month', 'Year'], axis = 1)

rating_no_free_apps['No of people Rated'] = rating_no_free_apps['No of people Rated'].round()

rating_no_free_apps['Price'] = rating_no_free_apps['Price'].round()

rating_no_free_apps
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)

rating_no_free_apps['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people rated for non-free apps per rating level')



plt.subplot(1, 2, 2)

rating_no_free_apps['Price'].plot(kind = 'bar')

plt.title('Price non-free apps per rating level')
new_df_no_free_apps = df_no_free_apps[df_no_free_apps['Price'] <= 1000.0]
len(new_df_no_free_apps)
rating_new_df = new_df_no_free_apps.groupby(['Rating']).mean().drop(['Month', 'Year'], axis = 1)

rating_new_df['No of people Rated'] = rating_new_df['No of people Rated'].round()

rating_new_df['Price'] = rating_new_df['Price'].round()
plt.figure(figsize = (26, 6))

plt.subplot(1, 2, 1)

rating_new_df['No of people Rated'].plot(kind = 'bar')

plt.title('Number of people rated for non-free apps per rating level - Removed outliers with price over 1000')



plt.subplot(1, 2, 2)

rating_new_df['Price'].plot(kind = 'bar')

plt.title('Price non-free apps per rating level - Removed outliers with price over 1000')