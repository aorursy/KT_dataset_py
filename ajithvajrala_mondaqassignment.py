import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

article_topics_df = pd.read_csv("/kaggle/input/mondaq-project/article_topics.csv", engine= 'python')

articles_df = pd.read_csv("/kaggle/input/mondaq-project/articles.csv", engine= 'python')

topic_relationships_df = pd.read_csv("/kaggle/input/mondaq-project/topic_relationships.csv", engine= 'python')
article_topics_df.shape, articles_df.shape, topic_relationships_df.shape
article_topics_df.head()
articles_df.head()
topic_relationships_df.head()
topic_relationships_df[topic_relationships_df['child_topic_id']==1]
article_topics_df.nunique()
topic_relationships_df.nunique()
topic_relationships_df.columns = ['parent_topic_id', 'topic_id']
article_topics_df.head()
df = pd.merge(article_topics_df, topic_relationships_df, on='topic_id', how='left')
df.tail()
df.nunique()
df['parent_topic_id'] = df['parent_topic_id'].fillna(df.topic_id)
df.nunique()
df.head()
articles_df.head()
articles_df = articles_df.dropna()
articles_df.isna().sum()
articles_df.shape, df.shape
articles_df.dtypes, df.dtypes
df['article_id'] = df['article_id'].astype(object)
df.head()
df.nunique()
df['article_id']
df[df['article_id'] ==957346]
df[df['article_id'] ==956082]
#df = df.drop_duplicates(subset='favorite_color', keep="first")

#df
final_df = df[['article_id', 'parent_topic_id']]
final_df['value'] = 1
final_df.head()
result_df = pd.pivot_table(final_df, values = ['value'], index=['article_id'], columns = 'parent_topic_id').reset_index()
result_df.fillna(0, inplace=True)
result_df.head()
f_df = pd.DataFrame(result_df.values)
f_df.columns = ['article_id' , 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7',

                     'topic_9', 'topic_10', 'topic_11', 'topic_12', 'topic_13', 'topic_14', 'topic_15', 'topic_16', 'topic_17',

                     'topic_18', 'topic_19', 'topic_20', 'topic_21', 'topic_22', 'topic_23', 'topic_24', 'topic_25', 'topic_26',

                     'topic_28', 'topic_29', 'topic_30', 'topic_31', 'topic_32']
f_df.head()
f_df.shape, articles_df.shape
f_df['article_id'] = f_df['article_id'].astype(int)

articles_df['article_id'] = articles_df['article_id'].astype(int)
r_df = pd.merge(articles_df, f_df, on='article_id', how='left')
r_df.head()
r_df.isna().sum()