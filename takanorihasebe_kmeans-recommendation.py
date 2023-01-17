import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines=False)
data.head()
data = data.dropna()
df = data[["title", "average_rating","ratings_count", "text_reviews_count", "ratings_count"]]
df.head()
title_lst = list(df["title"].values)
value_arr = df[["average_rating", "ratings_count", "text_reviews_count", "ratings_count"]].values
sc = StandardScaler()
value_std = sc.fit_transform(value_arr)
kmeans = KMeans(n_clusters=100, random_state=0).fit(value_std)
kmeans.predict([value_std[0]])[0]
label_lst = kmeans.labels_ == kmeans.predict([value_std[0]])[0]
print("your_search:", title_lst[0])
print("")
for i in range(0, len(title_lst)):
    
    if label_lst[i] == True:
        print(title_lst[i])
label_lst = kmeans.labels_ == kmeans.predict([value_std[20]])[0]
print("your_search:", title_lst[20])
print("")
for i in range(0, len(title_lst)):
    
    if label_lst[i] == True:
        print(title_lst[i])
