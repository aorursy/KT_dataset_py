import numpy as np 
import pandas as pd 
from scipy.sparse import csr_matrix, lil_matrix
import sqlite3
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sklearn_pandas
import seaborn as sns
sql_conn = sqlite3.connect('../input/database.sqlite')
pd.read_sql("PRAGMA table_info(May2015)", sql_conn)
pd.read_sql("SELECT COUNT(*) FROM May2015", sql_conn)
pd.read_sql("""
    SELECT author, COUNT(*) as count_comments 
    FROM May2015 
    GROUP BY author 
    ORDER BY count_comments DESC 
    LIMIT 10""", sql_conn)
pd.read_sql("SELECT COUNT(*) as count_comments FROM May2015 WHERE body = '[deleted]'", sql_conn)
%%time
pd.read_sql("SELECT subreddit, COUNT(*) as count_comments FROM May2015 GROUP BY subreddit ORDER BY count_comments DESC LIMIT 10", sql_conn)
data = pd.read_sql("""
    SELECT *
    FROM May2015 WHERE body != '[deleted]' AND author != '[deleted]' AND author NOT LIKE '%bot' AND author != 'AutoModerator' 
    LIMIT 100000
    """, sql_conn, parse_dates=['created_utc', 'retrieved_on', 'edited'])
data.head()
data.describe(include=[np.number])
data.describe(include=[np.object, 'category'])
plt.figure(figsize=(16,8))
plt.hist(data.score, bins=30, range=(-10,20));
plt.title('Score distribution');
data = pd.read_sql("""
    SELECT author, subreddit
    FROM May2015 WHERE body != '[deleted]' AND author != '[deleted]' AND author NOT LIKE '%bot' AND author != 'AutoModerator' 
    LIMIT 100000
    """, sql_conn)
count_comments_by_author = data.groupby('author').size()
authors_with_multiple_comments = set(count_comments_by_author[count_comments_by_author > 1].index.values)
data = data.loc[data['author'].isin(authors_with_multiple_comments)]
print("Comments:", len(data))
authors = sorted(data['author'].unique())
subreddits = sorted(data['subreddit'].unique())
author_to_index = {k: v for v, k in enumerate(authors)}
subreddit_to_index = {k: v for v, k in enumerate(subreddits)}
print("Authors:", len(authors))
print("Subreddits:", len(subreddits))
%%time
users_subreddits_matrix = lil_matrix((len(authors), len(subreddits)), dtype=np.int64)
for i, comment in data.iterrows():
    row = author_to_index[comment['author']]
    col = subreddit_to_index[comment['subreddit']]
    users_subreddits_matrix[row, col] += 1
users_subreddits_matrix
kmeans = cluster.KMeans(n_clusters=1000, random_state=42)
kmeans.fit(users_subreddits_matrix)
metrics.silhouette_score(users_subreddits_matrix, labels=kmeans.labels_)
selected_cluster = 2
subreddit_weights = pd.DataFrame({'subreddit': subreddits, 'cluster_center': kmeans.cluster_centers_[selected_cluster]})
subreddit_weights[subreddit_weights['cluster_center'] > 0].sort_values(by='cluster_center', ascending=False).head(10)
svd = TruncatedSVD(n_components=100)
svd.fit(users_subreddits_matrix)
print("Explained variance by 100 components:", svd.explained_variance_ratio_.sum())
pd.DataFrame({'subreddit': subreddits, 'component1': svd.components_[0]}).sort_values(by='component1', ascending=False).head(10)
pd.DataFrame({'subreddit': subreddits, 'component2': svd.components_[1]}).sort_values(by='component2', ascending=False).head(10)
data = pd.read_sql("""
    SELECT body, score
    FROM May2015 WHERE body != '[deleted]' AND author != '[deleted]' AND author NOT LIKE '%bot' AND author != 'AutoModerator' 
    LIMIT 100000
    """, sql_conn)
min_score = min(data['score'])
data['log_score'] = np.log1p(data['score']-min_score)
def original_score(log_scores):
    return np.expm1(log_scores)+min_score
data.describe()
X_train_body, X_test_body, y_train, y_test = train_test_split(data['body'].values, data['log_score'].values, random_state=42, test_size=0.2)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')
vectorizer.fit(X_train_body)
print("Features:", len(vectorizer.vocabulary_))
X_train = vectorizer.transform(X_train_body)
X_test = vectorizer.transform(X_test_body)
model = Ridge()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict_orig = original_score(y_predict)
y_test_orig = original_score(y_test)
print("Logarithmic scores RMSE", metrics.mean_squared_error(y_predict, y_test) ** 0.5)
print("Logarithmic scores MAE", metrics.mean_absolute_error(y_predict, y_test))
print("Original scores RMSE", metrics.mean_squared_error(y_predict_orig, y_test_orig) ** 0.5)
print("Original scores MAE", metrics.mean_absolute_error(y_predict_orig, y_test_orig))
terms_and_weights = [(term, model.coef_[index]) for term, index in vectorizer.vocabulary_.items()]
pd.DataFrame(terms_and_weights, columns=['term', 'weight']).sort_values(by='weight', ascending=False).head(10)
hourly_comments = pd.read_sql("""
    SELECT 
        strftime('%w', datetime(created_utc, 'unixepoch')) AS weekday, 
        strftime('%H', datetime(created_utc, 'unixepoch')) AS hour, 
        COUNT(*) AS count_comments
    FROM May2015 
    WHERE body != '[deleted]' AND author != '[deleted]' AND author NOT LIKE '%bot' AND author != 'AutoModerator' 
    AND created_utc < 1432857600 -- May 1 - 28 (4 full weeks)
    GROUP BY weekday, hour
    ORDER BY weekday, hour
    """, sql_conn)
hourly_comments_table = pd.pivot_table(hourly_comments, index='weekday', columns='hour', aggfunc=np.sum)
hourly_comments_table.columns = hourly_comments_table.columns.droplevel()
plt.figure(figsize=(16,8))
sns.heatmap(hourly_comments_table);
