# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df.head()
print('There are ',len(df.App.unique()),'unique apps')
df.loc[10472, :] = df.loc[10472, :].shift(1) 

df.iloc[10472,0] = df.iloc[10472,1]

df.iloc[10472,1] = 'LIFESTYLE'



df.Installs = df.loc[:, 'Installs'].map(lambda x: x.rstrip('+'))

df.Installs = df['Installs'].replace(',','', regex=True)

df.Installs = df['Installs'].astype(int)



df.Price = df.loc[:, 'Price'].map(lambda x: x.lstrip('$'))

df.Price = df['Price'].astype(float)



df.Size = df['Size'].str.replace('M','')

df.Size = pd.to_numeric(df['Size'], errors= 'coerce')



df.Type = df['Type'].str.replace('0','Free')



#print(df['Rating'].info())

df.Rating = df.Rating.astype(float)



total_downloads = df.Installs.sum()

print('Total number of downloads of all applications: ' + str(total_downloads) + '+')
df['Total_money'] = df['Installs'] * df['Price']

print('Total money spent for all paid applications: $' + str(df.Total_money.sum()))
print('Percentages of number of installs by category')

data_category = df.groupby('Category').sum()['Installs'].sort_values(ascending= False)

#(100. * data_category / data_category.sum()).round(0)

labels = data_category.index.values

fig, ax = plt.subplots(figsize = (15,8))

sns.barplot(data_category, labels, ax= ax)
downloads = df.groupby('Category').sum().Installs.sort_values(ascending= False)

labels = downloads.index.values

downloads.plot.pie( y = 'Category', subplots = True, autopct='%.1f%%', 

 startangle=90, shadow=False, labels=labels, legend = False, fontsize=15, figsize=(15,15))
df_install_percent = df.groupby(['Category'])['Installs'].sum()

for category in df_install_percent.index:

    df_ratio_app_by_cat = pd.DataFrame(columns = ['App', 'Installs'])

    df_ratio_app_by_cat = df[df['Category'] == category][['App', 'Installs']]

    df_ratio_app_by_cat['Install_ratio'] = df_ratio_app_by_cat['Installs'] / df_ratio_app_by_cat['Installs'].sum() * 100

    print('Percentages of number of Installs of apps in category ' + str(category))

    print(df_ratio_app_by_cat[['App','Install_ratio']])
for category in df_install_percent.index:

    df_ratio_app_by_cat = pd.DataFrame(columns = ['App', 'Installs'])

    df_ratio_app_by_cat = df[df['Category'] == category][['App', 'Installs']]

    df_ratio_app_by_cat['Install_ratio'] = df_ratio_app_by_cat['Installs'] / df_ratio_app_by_cat['Installs'].sum() * 100

    largest5 = df_ratio_app_by_cat.nlargest(5, "Install_ratio") 

    print(category)

    print(largest5)

    print("-"*80)
print("Average rating is " + str(df["Rating"].mean()))
df['Rating'].plot.hist(bins=20, alpha = 0.8)

print('Average ratings of apps by categories')

df_cat_mean_rating = df.groupby(['Category'])['Rating'].mean()

print(df_cat_mean_rating)

print('-'*40)

print('Top-5 high-rated categories')

print(df_cat_mean_rating.nlargest(5))

print('-'*40)

print('Top-5 low-rated categories')

print(df_cat_mean_rating.nsmallest(5))