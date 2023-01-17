{"cells":[{"metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","trusted":True,"collapsed":True},"cell_type":"code","source":"import bq_helper\n# create a helper object for our bigquery dataset\nchicago_crime = bq_helper.BigQueryHelper(active_project= \"bigquery-public-data\", \n                                       dataset_name = \"chicago_crime\")","execution_count":1,"outputs":[]},{"metadata":{"_cell_guid":"79c7e3d0-c299-4dcb-8224-4455121ee9b0","collapsed":True,"_uuid":"d629ff2d2480ee46fbb7e2d37f6b5fab8052498a","trusted":False},"cell_type":"code","source":"","execution_count":None,"outputs":[]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.6.5","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":1}
import bq_helper
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("comments")
hacker_news.table_schema("full")
# preview the first couple lines of the "full" table
hacker_news.head("comments")
hacker_news.head("full")
# preview the first ten entries in the by column of the full table
hacker_news.head("full", selected_columns="by", num_rows=10)
# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)

# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)

job_post_scores = hacker_news.query_to_pandas_safe(query)
# average score for job posts
job_post_scores.score.mean()
