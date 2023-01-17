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
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/windows-store/msft.csv')
df.head()
df.tail()
df.info()
df.isnull().sum()
df.dropna(inplace = True)

df['Category'].unique()
df_gby = df.groupby('Category')   # Creating Groupby object



categories = df_gby['Name'].count().sort_values(ascending = False).index

values = df_gby['Name'].count().sort_values(ascending = False).values



with plt.style.context('fivethirtyeight'):

    plt.barh(categories[::-1], values[::-1])

    plt.xlabel('Number of Apps')

    plt.title('Apps on Diffrent Catrgories', fontweight = 'bold')

    plt.show()

free_apps = (df['Price'] == 'Free').sum()

per_free_apps = ((free_apps)/(5321))*100   # percentage of free apps



paid_apps = df[~(df['Price'] == 'Free')].shape[0]

per_paid_apps = ((paid_apps)/(5321))*100 # percentage of paid apps 

# OR (per_paid_apps = 100 - per_free_apps) 
plt.pie([per_free_apps, per_paid_apps], labels = ['Free_Apps', 'Paid_Apps'],shadow = True,

        autopct='%1.1f%%', pctdistance=0.5, textprops = {'fontsize' : 15})
df['No of people Rated'].describe()
df1 = df[(df['No of people Rated'] > 500)]

Ratings = df1['Rating']

with plt.style.context('ggplot'):

    plt.hist(Ratings, bins = [1.0, 2.0, 3.0, 4.0, 5.0], edgecolor = 'white')

    plt.ylabel('Number of Apps', fontweight = 'bold')

    plt.xlabel('Ratings', fontweight = 'bold')

    plt.xticks(fontweight = 'bold')

    plt.yticks(fontweight = 'bold')

    plt.text(1, 1600, 'Minimum 500 Reviews')

    plt.title('Apps vs Rating', fontweight = 'bold')

    plt.show()
def remove_rupee_sign(x):

    x = x.split()              # it gives a list ['₹', '250.00']

    try:

        return int(float(x[1]))  # taking out the second element from the list and converting to int

    except:

        return int(float(x[1].replace(',', '')))    # 1,624.00 => used to handle this type of data

    
Paid_apps_df = df.loc[5163:].copy()    # creating separate dataframe for paid apps

Paid_apps_df['Price in ₹'] = Paid_apps_df['Price'].map(remove_rupee_sign)     # replacing old column with new one



Paid_apps_df.set_index('Name', inplace = True)



Name_Top_10 = Paid_apps_df['Price in ₹'].nlargest(10).index    # Name of top 10 expensive apps

Price_Top_10 = Paid_apps_df['Price in ₹'].nlargest(10).values    # Price of top 10 expensive apps



plt.barh(Name_Top_10[::-1], Price_Top_10[::-1], color = 'gray')

plt.title('Top 10 Expensive apps', pad = 20, fontsize = 20, fontfamily = 'Rockwell Extra Bold')

plt.xlabel('Price of Apps', fontweight = 'bold')

plt.yticks(fontweight = 'bold')

plt.xticks(fontweight = 'bold')

plt.grid(True)

year = pd.DatetimeIndex(df['Date']).year.value_counts().index

apps = pd.DatetimeIndex(df['Date']).year.value_counts().values



with plt.style.context('seaborn'):

    plt.bar(year, apps)

    plt.xticks(range(2010, 2021))

    plt.title('Apps Added each Year', fontweight = 'bold', pad = 10)

    plt.ylabel('Number of apps added')

    plt.show()