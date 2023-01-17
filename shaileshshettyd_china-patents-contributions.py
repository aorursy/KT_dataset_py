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
query0 = """

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



response0 = patents_research.query_to_pandas_safe(query0,max_gb_scanned=7)

response0
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='publication_description', y='publications', data=response0, color='grey')



plt.title("Total patents granted to CHINA", loc='center')

plt.ylabel('#Publications')

plt.xlabel('US')

sns.despine();
query0 = """

SELECT 

    top_terms

FROM

  `patents-public-data.google_patents_research.publications`

WHERE country LIKE 'CHINA'

   AND publication_description LIKE 'Granted%invention%'

;

        """

#bq_assistant.estimate_query_size(query1)



inventions = patents_research.query_to_pandas_safe(query0,max_gb_scanned=15)
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

plt.xlabel('Number of occurences')

plt.ylabel('CHINA')

sns.despine();
query1 = """

SELECT 

  publication_description

  ,COUNT(DISTINCT publication_number) as publications

FROM

  `patents-public-data.google_patents_research.publications_201710`

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

plt.xlabel('OCT-2017')

sns.despine();
query1 = """

SELECT 

    top_terms

FROM

  `patents-public-data.google_patents_research.publications_201710`

WHERE country LIKE 'CHINA'

   AND publication_description LIKE 'Granted%invention%'

;

        """

#bq_assistant.estimate_query_size(query1)



inventions = patents_research.query_to_pandas_safe(query1,max_gb_scanned=13)
inventions = inventions[inventions.astype(str)['top_terms'] != '[]']

inventions.sample(5)
top_terms = pd.DataFrame(inventions["top_terms"].tolist())



top_terms = pd.DataFrame(top_terms.values.flatten())

top_terms.columns = ['top_terms']



top_terms = top_terms.dropna(axis=0,how='all')

top_terms.shape
# 9 millions terms

top_terms.sample(5)
# 572k unique terms

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



# Add title and axis names



plt.yticks(my_range, df_agg.index)

plt.title("Top 30 most occuring terms", loc='left')

plt.xlabel('Number of occurences')

plt.ylabel('CHINA IN OCT-2017')

sns.despine();
query2 = """

SELECT 

  publication_description

  ,COUNT(DISTINCT publication_number) as publications

FROM

  `patents-public-data.google_patents_research.publications_201802`

WHERE country LIKE 'CHINA'

    AND publication_description LIKE 'Granted%'

GROUP BY publication_description

ORDER BY publications DESC;

        """

#bq_assistant.estimate_query_size(query1)



response2 = patents_research.query_to_pandas_safe(query2,max_gb_scanned=6)

response2
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='publication_description', y='publications', data=response2, color='red')



plt.title("Total patents granted to China", loc='center')

plt.ylabel('#Publications')

plt.xlabel('FEB-2018')

sns.despine();
query2 = """

SELECT 

    top_terms

FROM

  `patents-public-data.google_patents_research.publications_201802`

WHERE country LIKE 'CHINA'

   AND publication_description LIKE 'Granted%invention%'

;

        """

#bq_assistant.estimate_query_size(query1)



inventions = patents_research.query_to_pandas_safe(query2,max_gb_scanned=13)
inventions = inventions[inventions.astype(str)['top_terms'] != '[]']

inventions.sample(5)
top_terms = pd.DataFrame(inventions["top_terms"].tolist())



top_terms = pd.DataFrame(top_terms.values.flatten())

top_terms.columns = ['top_terms']



top_terms = top_terms.dropna(axis=0,how='all')

top_terms.shape
# 9 millions terms

top_terms.sample(5)
# 574k unique terms

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



# Add title and axis names



plt.yticks(my_range, df_agg.index)

plt.title("Top 30 most occuring terms", loc='left')

plt.xlabel('Number of occurences')

plt.ylabel('CHINA IN FEB-2018')

sns.despine();
query3 = """

SELECT 

  publication_description

  ,COUNT(DISTINCT publication_number) as publications

FROM

  `patents-public-data.google_patents_research.publications_201809`

WHERE country LIKE 'CHINA'

    AND publication_description LIKE 'Granted%'

GROUP BY publication_description

ORDER BY publications DESC;

        """

#bq_assistant.estimate_query_size(query1)



response3 = patents_research.query_to_pandas_safe(query3,max_gb_scanned=6)

response3
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='publication_description', y='publications', data=response3, color='blue')



plt.title("Total patents granted to China", loc='center')

plt.ylabel('#Publications')

plt.xlabel('SEPT-2018')

sns.despine();
query3 = """

SELECT 

    top_terms

FROM

  `patents-public-data.google_patents_research.publications_201809`

WHERE country LIKE 'CHINA'

   AND publication_description LIKE 'Granted%invention%'

;

        """

#bq_assistant.estimate_query_size(query1)



inventions = patents_research.query_to_pandas_safe(query3,max_gb_scanned=13)
inventions = inventions[inventions.astype(str)['top_terms'] != '[]']

inventions.sample(5)
top_terms = pd.DataFrame(inventions["top_terms"].tolist())



top_terms = pd.DataFrame(top_terms.values.flatten())

top_terms.columns = ['top_terms']



top_terms = top_terms.dropna(axis=0,how='all')

top_terms.shape
# 9 millions terms

top_terms.sample(5)
# 391k unique terms

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

plt.xlabel('Number of occurences')

plt.ylabel('CHINA IN SEPT-2018')

sns.despine();
query4 = """

SELECT 

  publication_description

  ,COUNT(DISTINCT publication_number) as publications

FROM

  `patents-public-data.google_patents_research.publications_201903`

WHERE country LIKE 'CHINA'

    AND publication_description LIKE 'Granted%'

GROUP BY publication_description

ORDER BY publications DESC;

        """

#bq_assistant.estimate_query_size(query1)



response4 = patents_research.query_to_pandas_safe(query4,max_gb_scanned=7)

response4
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='publication_description', y='publications', data=response4, color='cyan')



plt.title("Total patents granted to China", loc='center')

plt.ylabel('#Publications')

plt.xlabel('MARCH-2019')

sns.despine();
query4 = """

SELECT 

    top_terms

FROM

  `patents-public-data.google_patents_research.publications_201903`

WHERE country LIKE 'CHINA'

   AND publication_description LIKE 'Granted%invention%'

;

        """

#bq_assistant.estimate_query_size(query1)



inventions = patents_research.query_to_pandas_safe(query4,max_gb_scanned=15)
inventions = inventions[inventions.astype(str)['top_terms'] != '[]']

inventions.sample(5)
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

plt.xlabel('Number of occurences')

plt.ylabel('CHINA IN MARCH-2019')

sns.despine();