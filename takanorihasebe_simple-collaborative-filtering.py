import pandas as pd
from scipy import stats
import numpy as np
data = pd.read_csv("/kaggle/input/movies-meta-data/movie_metadata.csv")
data.head()
pd.set_option('display.max_columns', 100)
data.head()
data.columns
df = data[["movie_title", "num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes", 
          "actor_1_facebook_likes", "gross", "num_voted_users", "cast_total_facebook_likes", "facenumber_in_poster",
          "num_user_for_reviews", "actor_2_facebook_likes", "imdb_score", "aspect_ratio", "movie_facebook_likes"]]
df.head()
df = df.T
col_lst = list(df[0:1].values[0])
df.columns = col_lst
df = df.drop('movie_title', axis=0)
df = df.fillna(0)
df2 = df.apply(stats.zscore, axis=0)
df2.head()
df2 = df2.loc[:,~df2.columns.duplicated()]
df2.head()
df2.columns
cnt = 0

for name in df2.columns:
    cnt += 1
    result_dict = {}
    # print(name)
    for name_compare in df2.columns:
        # print(name_compare)
        result_dict[name_compare] = np.corrcoef(df2[name].values, df2[name_compare].values)[0][1]
        # result_dict[name_compare] = spearmanr(df[name].values, df[name_compare].values)[0]
    # print(result_dict)
    # print(sorted(result_dict.items(), key=lambda x: x[1], reverse = True))
    print(name)
    for i in sorted(result_dict.items(), key=lambda x: x[1], reverse = True)[1:4]:
        print(i)
    print('')
    
    if cnt >= 3:
        break
