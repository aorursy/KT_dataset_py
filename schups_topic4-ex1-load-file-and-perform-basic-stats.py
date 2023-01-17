%matplotlib inline

import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/hotel-reviews/Datafiniti_Hotel_Reviews.csv')
print(f"DataFrame shape: {df.shape}")
print(f"Oldest review considered: {min(df['dateAdded'])}")
print(f"Newest review considered: {max(df['dateAdded'])}")
print(f"The reviews were posted by {len(df['reviews.username'].unique())} different users")
df.head()
simplified_df = df[['reviews.username', 'reviews.rating']].rename(columns={
    'reviews.username': 'username',
    'reviews.rating': 'rating'
})
simplified_df
(simplified_df.rating.min(), simplified_df.rating.max())
ratings_by_user = simplified_df.groupby(['username']).rating.agg(['count', 'mean'])
ratings_by_user.sort_values(by='count')
fork1 = ratings_by_user[ratings_by_user['count'] == 1]
fork2 = ratings_by_user[ratings_by_user['count'] > 1]
fork3 = ratings_by_user[ratings_by_user['count'] > 5]
fork1_prepared = fork1['mean'].map(lambda mean: round(mean, 1))
fork2_prepared = fork2['mean'].map(lambda mean: round(mean, 1))
fork3_prepared = fork3['mean'].map(lambda mean: round(mean, 1))
fork1_prepared.value_counts()
fork1_prepared.hist()
fork2_prepared.value_counts()
fork2_prepared.hist()
fork3_prepared.value_counts()
fork3_prepared.hist()