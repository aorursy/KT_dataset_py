

# Any results you write to the current directory are saved as output.
# kirahman2
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
hacker_news.list_tables()
