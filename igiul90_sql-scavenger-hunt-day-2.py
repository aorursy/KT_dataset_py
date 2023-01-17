# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# Full schema of "full" tables

print(hacker_news.table_schema("full"))
# This query looks in the "full" table and then gets the ranking in descending order of type columns

question1 = """SELECT type, COUNT(id) as type_ranking
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 0  
            ORDER BY COUNT(id) DESC
        """

# Estimation of query question1
print(hacker_news.estimate_query_size(question1))

# I use max_db_scanned = 2 to limit at 2 GB
stories = hacker_news.query_to_pandas_safe(question1, max_gb_scanned=2)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(stories.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "stories"
print(stories)

# This query looks in the "full" table and then gets the comments_delected in descending order of type columns

question2 = """SELECT deleted, COUNT(id) as comments_deleted
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING COUNT(id) > 0  
            ORDER BY COUNT(id) DESC
        """

# Another possible query but more exactly is:

question2_1 = """SELECT deleted, COUNT(id) as comments_deleted
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY deleted
            HAVING COUNT(id) > 0  
            ORDER BY COUNT(id) DESC
        """

# Estimation of query question2_1
print(hacker_news.estimate_query_size(question2_1))

# I use max_db_scanned = 2 to limit at 2 GB
comments_deleted = hacker_news.query_to_pandas_safe(question2_1, max_gb_scanned=2)

# Print Dataframe Size
print('Size of dataframe: {} Bytes'.format(int(comments_deleted.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "comments_deleted"
print(comments_deleted)

# This query looks in the "full" table and then gets MAX and MIN of score columns

question3 = """SELECT type, COUNT(id) as type_ranking, score, MAX(score) as max_score,
                      MIN(score) as mix_score
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type, score
            HAVING COUNT(id) > 0  
            ORDER BY COUNT(id) DESC
        """

# Estimation of query question3
print(hacker_news.estimate_query_size(question3))

# I use max_db_scanned = 2 to limit at 2 GB
best_score = hacker_news.query_to_pandas_safe(question3, max_gb_scanned=2)

# Print Dataframe Size
print('Size of dataframe: {} Bytes'.format(int(best_score.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "comments_deleted"
print(best_score.head())