# import our bq_helper package
import bq_helper
# create a helper object for our bigquery dataset
github = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
# print a list of all the tables in the hacker_news dataset
github.list_tables()
github.head('sample_contents', num_rows=3)
q_tab_or_space = ('''
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.py" )
SELECT
  Indentation,
  COUNT(Indentation) AS number_of_occurence
FROM (
  SELECT
    CASE
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^\t+")))>=1 THEN 'Tab'
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +")))>=1 THEN 'Space'
        ELSE 'Other'
    END AS Indentation
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^\s+")
  GROUP BY
    id )
GROUP BY
  Indentation
ORDER BY
  number_of_occurence DESC
''')
github.estimate_query_size(q_tab_or_space)
tab_or_space_df = github.query_to_pandas(q_tab_or_space)
tab_or_space_df
import matplotlib.pyplot as plt
import seaborn as sns
# setting default seaborn aesthetic style
sns.set()
# define plot and figure size
fig = plt.figure(figsize=(10,6))
# Add a subplot
ax = fig.add_subplot(111)
bars = tab_or_space_df.plot(kind='bar', x='Indentation', y='number_of_occurence', ax=ax)
ax.set_ylabel('Occurence',fontsize=12,alpha=0.75)
ax.set_xlabel('Indentation',fontsize=12,alpha=0.75)
ax.set_title('Tabs vs Spaces\n(github python files)')
ax.set_ylim(0, tab_or_space_df['number_of_occurence'].max()*1.2)
plt.xticks(rotation=0);

for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, (height+1500), '{}'.format(height), 
                 ha='center', color='black', fontsize=12, alpha=0.75)

plt.show()