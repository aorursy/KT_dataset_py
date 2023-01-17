import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data viz



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/movie_metadata.csv")

df.describe()
df = df[df.title_year > 1980]

df.describe()
df_review = df

# Dropping irrelevant columns

df_review = df_review.drop('color', 1)

df_review = df_review.drop('director_name', 1)

df_review = df_review.drop('duration', 1)

# Last method was inefficient

df_review.drop(df_review.columns[[1,2,3,4,5,6,7,10,11,12,13,14,16,17,18,19,21,23,24]], axis=1, inplace=True)

df_review['review_score'] = (df_review['num_critic_for_reviews'] + df_review['num_user_for_reviews'])/2

df_review.shape