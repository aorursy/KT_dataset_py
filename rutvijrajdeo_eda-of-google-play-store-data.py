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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
dataset_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
dataset_df.info()
dataset_df.rename(columns={
    "Content Rating":"Content_Rating",
    "Last Updated":"Last_Updated",
    "Current Ver":"Current_Ver",
    "Android Ver":"Android_Ver"},inplace=True)
### Converting Reviews column to numeric

dataset_df['Reviews'] = pd.to_numeric(dataset_df.Reviews,errors='coerce')
### Converting Price column to numeric

dataset_df['Price'] = pd.to_numeric(dataset_df.Price.str.replace('$',''),errors='coerce')
### Converting Installs column to numeric

dataset_df['Installs'] = pd.to_numeric(dataset_df.Installs.str.replace(',','').str.replace('+',''), errors='coerce')
### Also converting Last updated column into datetime object

dataset_df['Last_Updated'] = pd.to_datetime(dataset_df['Last_Updated'], errors = 'coerce')
dataset_df.info()
dataset_df['Last_Updated_Year'] = dataset_df['Last_Updated'].dt.year
dataset_df['Last_Updated_Month'] = dataset_df['Last_Updated'].dt.month
dataset_df['Last_Updated_Day'] = dataset_df['Last_Updated'].dt.day
dataset_df.Last_Updated_Year.unique()
dataset_df.head()
dataset_df.tail()
dataset_df.isna().sum()
dataset_df.describe()
top_categories = dataset_df.Category.value_counts().head(10)
top_categories
plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.title('Top 10 Categories')
sns.barplot(top_categories.index, top_categories);
types = dataset_df.Type.value_counts()
types
plt.figure(figsize=(12,6))
plt.title('Types')
plt.pie(types, labels=types.index, autopct='%1.1f%%', startangle=180);
content_rating = dataset_df.Content_Rating.value_counts()
content_rating
sns.barplot(content_rating.index, content_rating)
plt.xticks(rotation=75);
plt.title('Age group the app is targeted at')
plt.ylabel(None);
plt.figure(figsize=(12, 6))
plt.title('Overall user rating of the app')
plt.xlabel('Ratings')
plt.ylabel('Number of users')

plt.hist(dataset_df.Rating, bins=np.arange(1,5,0.5), color='green');
genres = dataset_df.Genres.value_counts().head(10)
genres
sns.barplot(genres, genres.index)

plt.title('Top 10 Genres')
plt.ylabel(None);
plt.xlabel('Count');
top_category_installs = dataset_df.groupby('Category')[['Installs']].sum().sort_values('Installs', ascending=False).head(10)
top_category_installs
sns.barplot(top_category_installs.index, top_category_installs.Installs)

plt.title('Top 10 Categories with largest number of installs')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('Category');
top_app_installs = dataset_df.groupby('App')[['Installs']].sum().sort_values('Installs', ascending=False).head(10)
                                              
top_app_installs
                   
sns.barplot(top_app_installs.index, top_app_installs.Installs)

plt.title('Top 10 Apps with largest number of installs')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('Category');
dataset_df['Size']
dataset_df['Size1'] = dataset_df['Size'].str.replace('M','e+6').str.replace('k','e+3').str.replace('+','').str.replace(',','').str.replace('Varies with device','0').astype('float')
large_size_apps = dataset_df[dataset_df['Size1']==dataset_df['Size1'].max()]
                                              
large_size_apps
most_downloads_by_year = dataset_df.groupby(['Last_Updated_Year','Category'])[['Installs']].sum().sort_values('Installs', ascending=False).reset_index()
most_downloads_by_year = most_downloads_by_year.loc[most_downloads_by_year.groupby("Last_Updated_Year")["Installs"].idxmax()]
most_downloads_by_year
most_no_of_reviews = dataset_df.groupby('App')[['Reviews']].mean().sort_values('Reviews', ascending=False).head(10)

most_no_of_reviews
sns.barplot(most_no_of_reviews.index, most_no_of_reviews.Reviews)

plt.title('Top 10 Apps with most number of reviews')
plt.xticks(rotation=75);
plt.ylabel(None);
plt.xlabel('Category');
dataset_df['No_days_before_updated'] = dataset_df['Last_Updated'].max()-dataset_df['Last_Updated']
dataset_df[['App','Category','No_days_before_updated']].sort_values('No_days_before_updated', ascending=False).head(10)
