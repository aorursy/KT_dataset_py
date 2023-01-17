import numpy as np
import pandas as pd
ratings_df = pd.read_csv("../input/ratings_small.csv")
ratings_df.shape
ratings_df.head()
movies_df = pd.read_csv("../input/movies_metadata.csv")
ratings_df.head()
movies_df.head()
movies_df.drop(movies_df.index[19730],inplace=True)
movies_df.drop(movies_df.index[29502],inplace=True)
movies_df.drop(movies_df.index[35585],inplace=True)
movies_df.id = movies_df.id.astype(np.int64)
type(movies_df.id[0])
ratings_df.movieId.isin(movies_df.id).sum()
ratings_df = pd.merge(ratings_df,movies_df[['title','id']],left_on='movieId',right_on='id')
ratings_df.head()
ratings_df.drop(['timestamp','id'],axis=1,inplace=True)
ratings_df.shape
ratings_df.sample(5)
ratings_df.isnull().sum()
ratings_count = ratings_df.groupby(by="title")['rating'].count().reset_index().rename(columns={'rating':'totalRatings'})[['title','totalRatings']]
ratings_count.shape[0]
len(ratings_df['title'].unique())
ratings_count.sample(5)
ratings_count.head()
ratings_df.head()
ratings_total = pd.merge(ratings_df,ratings_count,on='title',how='left')
ratings_total.shape
ratings_total.head()
ratings_count['totalRatings'].describe()
ratings_count['totalRatings'].quantile(np.arange(.6,1,0.01))
votes_count_threshold = 20
ratings_top = ratings_total.query('totalRatings > @votes_count_threshold')
ratings_top.shape
ratings_top.head()
if not ratings_top[ratings_top.duplicated(['userId','title'])].empty:
    ratings_top = ratings_top.drop_duplicates(['userId','title'])
ratings_top.shape
df_for_knn = ratings_top.pivot(index='title',columns='userId',values='rating').fillna(0)
df_for_knn.head()
df_for_knn.shape
from scipy.sparse import csr_matrix
df_for_knn_sparse = csr_matrix(df_for_knn.values)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(df_for_knn_sparse)
query_index = np.random.choice(df_for_knn.shape[0])
distances, indices = model_knn.kneighbors(df_for_knn.loc['Batman Returns'].values.reshape(1,-1),n_neighbors=6)
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))
def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
df_for_ar = df_for_knn.T.applymap(encode_units)
df_for_ar.shape
df_for_ar.head()
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemsets = apriori(df_for_ar, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
query_index = df_for_knn.index.get_loc('Batman Returns')
query_index
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print("KNN Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))
all_antecedents = [list(x) for x in rules['antecedants'].values]
desired_indices = [i for i in range(len(all_antecedents)) if len(all_antecedents[i])==1 and all_antecedents[i][0]=='Batman Returns']
apriori_recommendations=rules.iloc[desired_indices,].sort_values(by=['lift'],ascending=False)
apriori_recommendations.head()
apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]
print("Apriori Recommendations for movie: Batman Returns\n")
for i in range(5):
    print("{0}: {1} with lift of {2}".format(i+1,apriori_recommendations_list[i],apriori_recommendations.iloc[i,6]))
apriori_single_recommendations = apriori_recommendations.iloc[[x for x in range(len(apriori_recommendations_list)) if len(apriori_recommendations_list[x])==1],]
apriori_single_recommendations_list = [list(x) for x in apriori_single_recommendations['consequents'].values]
print("Apriori single-movie Recommendations for movie: Batman Returns\n")
for i in range(5):
    print("{0}: {1}, with lift of {2}".format(i+1,apriori_single_recommendations_list[i][0],apriori_single_recommendations.iloc[i,6]))