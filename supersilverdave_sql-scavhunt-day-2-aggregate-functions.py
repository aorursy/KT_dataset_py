import bq_helper
#set up big query after adding data set to data tab of kernel
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",\
                                      dataset_name="hacker_news")

#list the tables to get an idea of the database
hacker_news.list_tables()
#check out the comments table structure
hacker_news.head("comments")
#get stories with more than 10 replies
query = """SELECT parent, COUNT(id) as total_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10;
        """

#Test the size
hacker_news.estimate_query_size(query)
#safe query to limit to 1 gb
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
#more useful to see the highest number of comments first
popular_stories_sorted = popular_stories.sort_values("total_comments", ascending=False)
popular_stories.sort_values("total_comments", ascending=False).head()
query_1 = """SELECT type, COUNT(id) as total_stories
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type;
        """

hacker_news.estimate_query_size(query_1)
answer_1_raw = hacker_news.query_to_pandas_safe(query_1)
answer_1 = answer_1_raw.sort_values("total_stories", ascending=False)
answer_1.to_csv("total_stories_per_type.csv")
answer_1
query_2 = """SELECT COUNT(id) as total_deleted_comments
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY deleted
                HAVING deleted=True;
            """

hacker_news.estimate_query_size(query_2)
answer_2_df = hacker_news.query_to_pandas_safe(query_2)
answer_2 = answer_2_df['total_deleted_comments'][0]
print("There are %d deleted comments in the dataset." % answer_2)
with open('answer_2.txt', 'w') as f:
    f.write("There are %d deleted comments in the dataset." % answer_2)    
#average comment ranking per type
bonus_query = """SELECT type, AVG(score) as average_score
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type;
                """
hacker_news.estimate_query_size(bonus_query)
bonus_answer_df = hacker_news.query_to_pandas_safe(bonus_query)
bonus_answer = bonus_answer_df.sort_values("average_score", ascending = False)
bonus_answer.to_csv('average_score_by_story_type.csv')
bonus_answer