import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
movies = pd. read_csv('../input/movie.csv')
tags = pd.read_csv('../input/tag.csv')
ratings = pd.read_csv('../input/rating.csv')
movies.head()
tags.head()
ratings.head()
movies.isnull().values.any()
movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)
movies['rating'] = ratings['rating']
movies.head()
movies.isnull().values.any()
movies = movies.dropna()
ind_animation = 'Animation'
ind_children = 'Children'

animation1 = movies['genres'].str.contains(ind_animation)
animation0 = ~movies['genres'].str.contains(ind_animation)
children1 = movies['genres'].str.contains(ind_children)
children0 = ~movies['genres'].str.contains(ind_children)

both = movies[animation1 & children1]
just_anim = movies[animation1 & children0]
just_chil = movies[animation0 & children1]
both.head()
just_anim.head()
just_chil.head()
print("The dataset which includes both Animation and Children genres has {0} rows.".format(len(both)))
print("The dataset which includes just Animation genre has {0} rows.".format(len(just_anim)))
print("The dataset which includes just Children genre has {0} rows.".format(len(just_chil)))
just_anim_plt = just_anim[['rating','year']]
just_anim_plt = just_anim_plt.groupby(['year'],as_index=False).mean()

just_anim_plt.head(15)
just_chil_plt = just_chil[['rating','year']]
just_chil_plt = just_chil_plt.groupby(['year'],as_index=False).mean()

just_chil_plt.head(15)
plt.plot(just_anim_plt['year'].values,just_anim_plt['rating'].values)

plt.show()
plt.plot(just_chil_plt['year'].values,just_chil_plt['rating'].values)

plt.show()
just_chil_plt['rating'].corr(just_anim_plt['rating'])