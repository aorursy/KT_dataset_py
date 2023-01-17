from google.cloud import bigquery
import pandas as pd
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

query = """SELECT id, `by`, title, score, time_ts, url, text
            FROM `bigquery-public-data.hacker_news.stories`
            WHERE LOWER(title) LIKE "%functional%programming%" """

print(hacker_news.estimate_query_size(query))

stories = hacker_news.query_to_pandas(query)
stories.to_csv('fp-stories.csv')

stories.head(10)
query = """SELECT
              `comments`.`id`, `comments`.`by`,
              `comments`.`text`, `comments`.`time_ts`,
              `comments`.`ranking`, `comments`.`deleted`,
              `comments`.`dead`, `comments`.`parent` AS story_id
            FROM `bigquery-public-data.hacker_news`.`comments`
            JOIN `bigquery-public-data.hacker_news`.`stories`
            ON `comments`.`parent` = `stories`.`id`
            WHERE LOWER(title) LIKE "%functional%programming%" """

print(hacker_news.estimate_query_size(query))

comments = hacker_news.query_to_pandas(query)
comments.to_csv('fp-comments.csv')

comments.head(10)