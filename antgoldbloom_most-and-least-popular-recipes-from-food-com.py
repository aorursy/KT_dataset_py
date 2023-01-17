import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('max_colwidth',100)
df_interactions = pd.read_csv('../input/food-com-recipes-and-user-interactions/interactions_train.csv')

gb_interactions = df_interactions.groupby('recipe_id')['rating']

df_rating = pd.concat([gb_interactions.count(),gb_interactions.mean()],axis=1)

df_rating = pd.concat([df_rating,gb_interactions.std()],axis=1)

df_rating.columns = ['Count','Rating','Stdev']



df_rating = df_rating.sort_values(by=['Rating','Count'],ascending=[False,False])

df_recipe_details = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv')

df_rating.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]

df_recipe_details = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv')

df_rating = df_rating.sort_values(by=['Rating','Count'],ascending=[False,False])

df_rating.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]
df_rating_mt5 = df_rating[df_rating['Count'] >=2]

df_rating_mt5 = df_rating_mt5.sort_values(by=['Rating','Count'],ascending=[True,False])

df_rating_mt5.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]
df_rating_mt5 = df_rating[df_rating['Count'] >=5]

df_rating_mt5 = df_rating_mt5.sort_values(by=['Stdev','Count'],ascending=[False,False])

df_rating_mt5.merge(df_recipe_details[['id','name']],how='inner',left_index=True,right_on='id')[:10]