# import package with helper functions 

import bq_helper



# create a helper object for this dataset

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                              dataset_name="github_repos")
github.list_tables()
github.head('commits')
github.head('contents')
github.tables
github.table_schema('files')
github.table_schema('contents')
github.table_schema('commits')
github.head('languages')
query = """SELECT language

            FROM `bigquery-public-data.github_repos.languages`

            LIMIT 5000

        """

github_languages = github.query_to_pandas_safe(query)

all_languages = []

for lang in github_languages.language:

    all_languages.extend(lang)



Languages_count={}

for lan in all_languages:

    if lan["name"] not in Languages_count:

        Languages_count[lan["name"]]=0

    Languages_count[lan["name"]]+=1
import operator

languages_counts = sorted(Languages_count.items(), key=operator.itemgetter(1),reverse=True)

languages_counts[:25]
query = '''SELECT repo_name, watch_count

        FROM `bigquery-public-data.github_repos.sample_repos`

        ORDER BY watch_count DESC 

        LIMIT 1000'''

propular_repos = github.query_to_pandas_safe(query)

propular_repos.head(10)

 
propular_repos.shape
# You can use two dashes (--) to add comments in SQL

query = ("""

        -- Select all the columns we want in our joined table

        SELECT L.license, COUNT(sf.path) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` as sf

        -- Table to merge into sample_files

        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 

            ON sf.repo_name = L.repo_name -- what columns should we join on?

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """)



file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)

file_count_by_license.head()
github.head('files')
# print out all the returned results

print(file_count_by_license)
github.head('sample_commits')
# Your code goes here :)

python_query = """SELECT sf.repo_name AS repo_name, COUNT(sc.commit) AS python_commits_number 

                    FROM `bigquery-public-data.github_repos.sample_files` AS sf JOIN 

                    `bigquery-public-data.github_repos.sample_commits` AS sc 

                    ON sf.repo_name = sc.repo_name

                    WHERE sf.path LIKE '%.py'

                    GROUP BY sf.repo_name  

                    ORDER BY python_commits_number DESC"""
python_commits = github.query_to_pandas_safe(python_query, max_gb_scanned=6)
print("Repos that have python commits" )

print(python_commits)