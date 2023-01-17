%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime

from matplotlib.pyplot import figure

import bq_helper
from bq_helper import BigQueryHelper

patents_research = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="google_patents_research")

# View table names under the patents data table
bq_assistant = BigQueryHelper("patents-public-data", "google_patents_research")
bq_assistant.list_tables()

# View the first three rows of the publications data table
# bq_assistant.head("publications", num_rows=3)

# View information on all columns in the trials data table
# bq_assistant.table_schema("publications")
query1 = """
SELECT 
  publication_description
  ,COUNT(DISTINCT publication_number) as publications
FROM
  `patents-public-data.google_patents_research.publications`
WHERE country LIKE 'CHINA'
    AND publication_description LIKE 'Granted%'
GROUP BY publication_description
ORDER BY publications DESC;
        """
#bq_assistant.estimate_query_size(query1)

response1 = patents_research.query_to_pandas_safe(query1,max_gb_scanned=6)
response1
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
sns.set(context='poster', style='ticks', font_scale=0.6)

ax = sns.barplot(x='publication_description', y='publications', data=response1, color='grey')

plt.title("Total patents granted to China", loc='center')
plt.ylabel('#Publications')
plt.xlabel('')
sns.despine();
query1 = """
SELECT 
    top_terms
FROM
  `patents-public-data.google_patents_research.publications`
WHERE country LIKE 'CHINA'
   AND publication_description LIKE 'Granted%invention%'
;
        """
#bq_assistant.estimate_query_size(query1)

inventions = patents_research.query_to_pandas_safe(query1,max_gb_scanned=11)
inventions = inventions[inventions.astype(str)['top_terms'] != '[]']
inventions.sample(5)
top_terms = pd.DataFrame(inventions["top_terms"].tolist())

top_terms = pd.DataFrame(top_terms.values.flatten())
top_terms.columns = ['top_terms']

top_terms = top_terms.dropna(axis=0,how='all')
top_terms.shape
# 9 millions terms
top_terms.sample(5)
# 500k unique terms
top_terms.top_terms.nunique()
df_agg = pd.DataFrame(top_terms.groupby('top_terms')['top_terms'].count())

df_agg.columns = ['counter']
df_agg = df_agg.sort_values('counter', ascending=False)
df_agg = df_agg.head(30)

df_agg.tail(5)
figure(num=None, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)
# Reorder it following the values:
my_range=range(1,len(df_agg.index)+1)

# Create a color if the group is "B"
my_color=np.where(df_agg['counter'] >= 30000, '#5ab4ac', '#d8b365')
my_size=np.where(df_agg['counter'] >= 0, 70, 30)

# The vertival plot is made using the hline function
# I load the seaborn library only to benefit the nice looking feature
plt.hlines(y=my_range, xmin=0, xmax=df_agg['counter'], color=my_color, alpha=1)
plt.scatter(df_agg['counter'], my_range, color=my_color, s=my_size, alpha=1)

# Add title and exis names

plt.yticks(my_range, df_agg.index)
plt.title("Top 30 most occuring terms", loc='left')
plt.xlabel('Number of occurances')
plt.ylabel('')
sns.despine();
