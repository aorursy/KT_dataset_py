# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from IPython.display import display
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

# Set charts to view inline
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
sns.set(rc={'figure.figsize':(11.7,8.27)})

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/AppleStore.csv')
df2 = pd.read_csv('../input/appleStore_description.csv')
df = df.merge(df2, on='id')
df.drop(['Unnamed: 0', 'track_name_y', 'size_bytes_y'], axis=1, inplace=True)
df.rename(columns = {'track_name_x' :'app_name'}, inplace=True)
df.info()
df.describe()
df.head()
df['free_flag'] = df['price'].apply(lambda x: 'Free' if x == 0 else 'Paid')
df['length_of_app_desc'] = df['app_desc'].apply(len)

sns.countplot(df['free_flag']).set_title('Free vs. Paid')
sns.countplot(df[df['price'] != 0][df['price'] < 10]['price']).set_title('Paid Price Distribution')
# Normalizing reviews for rank ~ https://stackoverflow.com/questions/8542391/how-to-normalize-reviews-based-on-score
df['current_review_rank'] = (df['rating_count_ver'] + (df['rating_count_ver'] + 3000) * df['user_rating_ver'] + (3000 + (df['rating_count_ver'] + 3000)) * df['user_rating_ver'].mean())
df['total_review_rank'] = (df['rating_count_tot'] + (df['rating_count_tot'] + 3000) * df['user_rating_ver'] + (3000 + (df['rating_count_ver'] + 3000)) * df['user_rating_ver'].mean())


def normalize(series):
    # Create x, where x the 'scores' column's values as floats
    x = series.values.reshape(-1, 1)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled
    # Run the normalizer on the dataframe

df['current_review_rank_normalized'] = normalize(df['current_review_rank'])
df['total_review_rank_normalized'] = normalize(df['total_review_rank'])

# 'current_review_rank'
df.nlargest(10, 'total_review_rank_normalized')[['app_name', 'total_review_rank_normalized']][::-1].set_index('app_name').plot(kind='barh', title='Total Reviews')
        
df.nlargest(10, 'current_review_rank_normalized')[['app_name', 'current_review_rank_normalized']][::-1].set_index('app_name').plot(kind='barh', title='Current Reviews')
sns.set(font_scale=1)

df[['prime_genre', 'total_review_rank']].groupby('prime_genre').mean().sort_values(by="total_review_rank").plot(kind='barh', title='Avg. Review Ranking Per Genre', figsize=(10,8))

df[['prime_genre', 'total_review_rank']].groupby('prime_genre').median().sort_values(by="total_review_rank").plot(kind='barh', title='Median Review Ranking Per Genre', figsize=(10,8))

df[['cont_rating', 'total_review_rank']].groupby('cont_rating').mean().sort_values(by="total_review_rank").plot(kind='barh', title='Avg. Review Ranking Per Content Rating', figsize=(10,8))
df[['cont_rating', 'total_review_rank']].groupby('cont_rating').median().sort_values(by="total_review_rank").plot(kind='barh', title='Median Review Ranking Per Content Rating', figsize=(10,8))

f = (
    df[['price', 'total_review_rank', 'user_rating', 'size_bytes_x', 'rating_count_tot', 'prime_genre', 'sup_devices.num', 'lang.num']]
).corr()

sns.heatmap(f, annot=True)