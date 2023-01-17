from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
# Construct a reference to the "posts_answers" table
answers_table_ref = dataset_ref.table("posts_answers")

# API request - fetch the table
answers_table = client.get_table(answers_table_ref)

# Preview the first five lines of the "posts_answers" table
client.list_rows(answers_table, max_results=5).to_dataframe()
questions_table = client.get_table(dataset_ref.table("posts_questions"))
client.list_rows(questions_table, max_results=5).to_dataframe()
query = """
              SELECT q.title AS title, q.body AS questions, a.body AS answers, q.id as q_id, q.tags as tags
              FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                  INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
              ON q.id = a.parent_id
              WHERE (q.tags LIKE '%java|%' OR q.tags='java') and (EXTRACT(MONTH FROM q.creation_date) = 5)
              """

result = client.query(query).result().to_dataframe()
import pandas as pd
df = pd.DataFrame(result)
df.shape
from bs4 import BeautifulSoup

def fix_html(html):
    cleantext = BeautifulSoup(html, 'lxml').text

#     for a in soup.findAll('code'):
#         a.replaceWith("CODE")

    return (cleantext).replace("\\n",'\n')
    
fix_html('''<p>I have <code>Dictionary&lt;int key, int sum</code></p>\n\n<p>asd</p>''')

import re
cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(html):
    cleantext = re.sub(cleanr, '', html)
    return cleantext
cleanhtml('''<p>I have <code>Dictionary&lt;int key, int sum</code></p>\n\n<p>asd</p>''')
df['questions'] = df['questions'].apply(lambda x: cleanhtml(x))
df['answers'] = df['answers'].apply(lambda x: cleanhtml(x))
df.head()
df[df['q_id'] == 6038002]
df[df['q_id'] == 6038002].loc[0][1]
df[df['q_id'] == 6038002].loc[0][2]
# Exporting Data
df.to_csv('java_stackoverflow.csv')
