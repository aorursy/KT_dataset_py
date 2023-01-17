!pip install jellyfish
import numpy as np

import pandas as pd

import jellyfish
!ls /kaggle/input
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

df_test
df_train_ext = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')

df_train_ext
set_test = set()

for i in df_test.index:

    set_test.add(df_test.loc[i, 'review'])
list_exact_match = []

for i in df_train_ext.index:

    if df_train_ext.loc[i, 'text'] in set_test:

        list_exact_match.append(df_train_ext.loc[i, 'text'])
len(list_exact_match)
list_exact_match
jellyfish.jaro_distance('great product fast delivery', 'Amazing product, quick delivery')
jellyfish.jaro_distance('easy to use', 'hard to use')
jellyfish.jaro_distance('good', 'goods')
# set_test = set()

# for i in df_test.index:

#     set_test.add(df_test.loc[i, 'review'])

# len(set_test)
# set_train_ext = set()

# for i in df_train_ext.index:

#     set_train_ext.add(df_train_ext.loc[i, 'text'])

# len(set_train_ext)
# df_similar_string = pd.DataFrame(columns=['test', 'train_ext', 'similarity_score'])
# for r_train in set_train_ext:

#     for r_test in set_test:

#         similarity_score = jellyfish.jaro_distance(r_test, r_train)

#         if similarity_score >= 0.9:

#             df_similar_string.loc[df_similar_string.shape[0]] = [r_test, r_train, similarity_score]
# pd.set_option('display.max_rows', None)

# pd.set_option('display.max_columns', None)

# pd.set_option('display.width', None)
# df_similar_string
# df_similar_string.to_csv('similarity.csv', index=None)