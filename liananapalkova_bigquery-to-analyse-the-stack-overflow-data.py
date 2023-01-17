import bq_helper
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from bq_helper import BigQueryHelper

stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()
bq_assistant.head("posts_questions", num_rows=5)
bq_assistant.head("posts_answers", num_rows=5)
bq_assistant.table_schema("posts_questions")
query1 = """SELECT
  EXTRACT(YEAR FROM creation_date) AS Year,
  COUNT(*) AS Number_of_Questions,
  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS Percent_Questions_with_Answers
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Year
HAVING
  Year > 2008 AND Year <= 2018
ORDER BY
  Year;
        """
bq_assistant.estimate_query_size(query1)
answered_questions = stackOverflow.query_to_pandas_safe(query1)
answered_questions.head(5)
ax = sns.barplot(x="Year",y="Percent_Questions_with_Answers",data=answered_questions,palette="coolwarm").set_title("What is the percentage of questions that have been answered over the years?")
query1 = """SELECT
  EXTRACT(YEAR FROM creation_date) AS Year,
  COUNT(*) AS Number_of_Questions,
  SUM(IF(answer_count > 0, 1, 0)) AS Number_Questions_with_Answers
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Year
HAVING
  Year > 2008 AND Year <= 2018
ORDER BY
  Year;
        """

answered_questions = stackOverflow.query_to_pandas_safe(query1)
answered_questions.head(5) 
answered_questions.plot(x="Year",y=["Number_of_Questions","Number_Questions_with_Answers"], 

                    kind="bar",figsize=(14,6), 

                    title='What is the total number of questions and questions that have been answered over the years?')
query2 = """SELECT
  EXTRACT(YEAR FROM creation_date) AS Year,
  COUNT(*) AS Number_of_Questions,
  ROUND(100 * SUM(IF(score > 0, 1, 0)) / COUNT(*), 1) AS Percent_Questions_Scored_Negatively
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Year
HAVING
  Year > 2008 AND Year <= 2018
ORDER BY
  Year;
        """
negatively_scored_questions = stackOverflow.query_to_pandas_safe(query2)
negatively_scored_questions.head(5)
ax = sns.barplot(x="Year",y="Percent_Questions_Scored_Negatively",data=negatively_scored_questions,palette="coolwarm").set_title("What is the percentage of negatively scored questions over years?")
query3 = """SELECT 
    REGEXP_EXTRACT(tags, "spark") AS Tag, 
    EXTRACT(YEAR FROM creation_date) AS Year, 
    COUNT(*) AS Number_Spark_Questions
FROM 
    `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Tag, Year
HAVING
  Year > 2008 AND Year <= 2018 AND Tag IS NOT NULL
ORDER BY
  Year;
"""

bq_assistant.estimate_query_size(query3)
spark_questions = stackOverflow.query_to_pandas_safe(query3)
spark_questions.head(5)
ax = sns.barplot(x="Year",y="Number_Spark_Questions",data=spark_questions,palette="coolwarm").set_title("What is the number of questions about Apache Spark over years?")
query4 = """SELECT tags
FROM 
    `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 200000;
"""

alltags = stackOverflow.query_to_pandas_safe(query4)

tags = ' '.join(alltags.tags).lower()
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=300,
                            relative_scaling=.5).generate(tags)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('stackOverflow.png')
plt.imshow(cloud);
query5 = """SELECT AVG(comment_count) AS Number_Comments, 
    score AS Score, 
    EXTRACT(YEAR FROM creation_date) AS Year
FROM 
    `bigquery-public-data.stackoverflow.posts_answers`
GROUP BY 
    Score, Year
ORDER BY
    Score;
"""

scores_answers = stackOverflow.query_to_pandas_safe(query5)
scores_answers.head(5)
plt.figure(figsize=(20,10))
plt.scatter(scores_answers["Year"], scores_answers["Score"], c=scores_answers["Number_Comments"], alpha=0.3, cmap='viridis')
plt.xlabel("Year")
plt.ylabel("Score")
plt.title("How average score of answers is evolving over years?")
plt.colorbar();  # show color scale