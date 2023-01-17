import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from bq_helper import BigQueryHelper
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,5)
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
QUERY="""SELECT binary, COUNT(*) AS count      
         FROM `bigquery-public-data.github_repos.contents`
         GROUP BY binary"""
bq_assistant.estimate_query_size(QUERY)
df= bq_assistant.query_to_pandas_safe(QUERY)
df.plot(kind="bar")
QUERY = """
        SELECT REGEXP_EXTRACT(path,r'/\.[0-9a-z]+$'), COUNT(*) AS count
        FROM `bigquery-public-data.github_repos.files`
        GROUP BY REGEXP_EXTRACT(path,r'/\.[0-9a-z]+$')
        ORDER BY COUNT(*) DESC LIMIT 50000
        """
print (bq_assistant.estimate_query_size(QUERY))

df = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=150)
f={}

df["f0_"]=df["f0_"].str.replace("/","")
for i,x in df.iterrows():
    val1=x["f0_"]
    val2=x["count"]
    if val1==None:
        continue
    f[val1]=val2

wordcloud = WordCloud(background_color='black',
                              stopwords=set(STOPWORDS),
                              random_state=42).generate_from_frequencies(f)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


f={}
files_to_ignore=[".gitignore", ".npmignore", ".gitkeep"]
df["f0_"]=df["f0_"].str.replace("/","")
for i,x in df.iterrows():
    val1=x["f0_"]
    val2=x["count"]
    if val1==None:
        continue
    if val1.find(".git")!=-1 or val1 in files_to_ignore:
        continue
    
    f[val1]=val2

wordcloud = WordCloud(background_color='black',
                              stopwords=set(STOPWORDS),
                              random_state=42).generate_from_frequencies(f)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

