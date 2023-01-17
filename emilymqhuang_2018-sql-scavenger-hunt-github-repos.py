# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
# print a list of all the tables in the dataset
github_repos.list_tables()
# print information on all the columns in the "contents" table
# in the github_repos dataset
github_repos.table_schema("contents")
github_repos.table_schema("languages")
# preview the first couple lines of the "..." table
github_repos.head("contents")
github_repos.head("languages")
github_repos.head("commits")
# preview the first ten entries in the by column of the languages table
github_repos.head("languages", selected_columns="language", num_rows=10)
query1 = """SELECT watch_count
            FROM `bigquery-public-data.github_repos.sample_repos`
            WHERE watch_count > 5000"""
github_repos.estimate_query_size(query1)
# check out the count of stars received (if the 
# query is smaller than 1 gig)
stars = github_repos.query_to_pandas_safe(query1,max_gb_scanned=0.5)
stars.watch_count.mean()
# save our dataframe as a .csv 
stars.to_csv("stars.csv")
