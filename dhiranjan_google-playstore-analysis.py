# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
%matplotlib inline
sns.set()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.head()
df.columns
df.shape
df.info()
df.isnull().sum()
df.columns = [x.lower() for x in df.columns]
df.columns
df[['rating','reviews','size','installs','price','current ver', 'android ver']]
rat = df['rating'].fillna(0).astype(float)
df['rating'] = rat.apply(lambda x : df['rating'].mean() if x == 0 else x)
df[df['reviews'].str.contains('[A-Z]', flags=re.I)]
result  = df.iloc[10472].values.tolist()
result.insert(1, 'Lifestyle')
result.pop()
df.iloc[10472] = result
df.iloc[10472]
df['reviews'] = df['reviews'].astype(int)
df['size'] = df['size'].apply(lambda x : '10.0' if x == 'Varies with device' else x.replace('M', ""))
df['size'] = df['size'].apply(lambda x : x.replace('K', ""))
df['size'] = df['size'].apply(lambda x : x.replace('k', ""))
df['size'] = df['size'].astype(float)
df['installs'].unique()
df['installs'] = df['installs'].apply(lambda x : x.replace('+', ''))
df['installs'] = df['installs'].apply(lambda x : x.replace(',', ''))
df['installs'] = df['installs'].astype(int)
df['installs'] = df.installs / 1000000 # Normalizing in term of Million.
df['price'] = df['price'].apply(lambda x : x.replace('$', ''))
df['price'] = df['price'].astype(float)
df['price'].unique()
apps = df.app
dublicate_apps = []
unique_apps = []

for each in apps:
    if each in unique_apps:
        dublicate_apps.append(each)
    else:
        unique_apps.append(each)
        
print('Total Dublicate apps : ', len(dublicate_apps))
print('Total unique apps : ', len(unique_apps))
reviews_max = {}

Google_clean = []
Google_already_added = []

for index, value in df.iterrows():
    result = value.tolist()
    app_name = result[0]
    app_reviews = result[3]
    
    if app_name in reviews_max and reviews_max[app_name] < app_reviews:
            reviews_max[app_name] = app_reviews
    elif app_name not in reviews_max:
            reviews_max[app_name] = app_reviews

    if reviews_max[app_name] == app_reviews and (app_name not in Google_already_added):
        Google_clean.append(value)
        Google_already_added.append(app_name)
df = pd.DataFrame(Google_clean, index = range(len(Google_clean))) # Creating new dataset.
print('Total columns: ', len(df))
def check_character(string):
    non_english_character = 0
    
    for character in string:
        if ord(character) > 127:
            non_english_character += 1
    
    if non_english_character > 3:
        return False
    else:
        return True

df['app_origin'] = df['app'].apply(lambda x : 'English' if check_character(x) else 'Foreign')

df.app_origin.value_counts()
df.loc[df['app_origin']=='Foreign'].head()
cats = df.category.value_counts()
cats
g = sns.catplot(x='category', kind='count', data=df, order=df.category.value_counts().index)
g.fig.set_figwidth(16)
g.fig.set_figheight(4)
plt.xticks(rotation=90)
fig = plt.figure(figsize=(16,4))
sns.distplot(df.rating, bins=30)
fig = plt.figure(figsize=(16,4))
sns.boxplot(x='category', y='rating', data=df)
plt.xticks(rotation=90)
df.groupby('category')['rating'].mean().sort_values(ascending=False).head()
app_type = df.type.value_counts()
print('Total' ,app_type.index[0], ":" , app_type.values[0])
print('Total' ,app_type.index[1], ":" , app_type.values[1])
Free_apps = df[(df['type'] == 'Free') & (df['installs']>100)][['app','category','installs']].sort_values(by='installs', ascending=False)
Free_apps # Free apps install more than 100 Millions and their category.
Free_apps['category'].value_counts()
Paid_apps = df[(df['type'] == 'Paid')][['app','category','installs']].sort_values(by='installs', ascending=False)
Paid_apps
Paid_apps['category'].value_counts()
# Game Category

df.groupby('category')['installs'].sum().sort_values(ascending=False)

df[(df['category'] == 'GAME') & (df['installs'] > 100)][['app','installs']] 
# ommuniction Category

df[(df['category'] == 'COMMUNICATION') & (df['installs'] > 100)][['app','installs']].sort_values(by='installs', ascending=False) 
# Tools Category.

df[(df['category'] == 'TOOLS') & (df['installs'] > 100)][['app','installs']]