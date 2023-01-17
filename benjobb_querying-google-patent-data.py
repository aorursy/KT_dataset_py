import bq_helper
from bq_helper import BigQueryHelper
import fuzzywuzzy
patents = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patents")
# View table names under the patents data table
bq_assistant = BigQueryHelper("patents-public-data", "patents")
bq_assistant.list_tables()
# View the first three rows of the publications data table
bq_assistant.head("publications", num_rows=3)
# View information on all columns in the publications data table
bq_assistant.table_schema("publications")
import os
cwd = os.getcwd()

os.listdir(cwd)

#importing company names for future use
import pandas as pd
company_df = pd.read_csv('../input/Accounts_company_name.csv')

replace_list = ['Limited', 'Ltd', 'Inc', 'Corp', 'Corporation', 'Sas', 'Ug', 'Pty', 'Llc', 'Ptv', 'Org', 'Gmbh','LLC','Company']

company_df['cleaned_name'] =company_df['Company Name'].str.replace(r'\b|\b'.join(replace_list),'',case=False)\
.str.replace(',','').str.strip()

# the companies are accessible here
company_df.sample(5)
# accepting dates; dates in the database are integers

import datetime
print('Enter the earliest date in YYYY-MM-DD format')
# date_entry = input()
date_entry = '2011-01-01'
date1 = date_entry.replace('-','')

print('Enter the latest date in YYYY-MM-DD format')
# date_entry = input()
date_entry = '2018-06-01'
date2 = date_entry.replace('-','')


# accepting keywords; to be compared to the text nested in the descripton array
# print('Enter a keyword you want to use:')
# keyword = input()
# keywords = ['retail', 'business to customer', 'software', 'e-commerce', 'database', 'computing', 'cloud']
keywords = ['gaming']
# can add country code or other filters later
print(date1)
print(date2)
print(keywords)

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def highest_similarity_finder(word, company_name):
    similarity = fuzz.token_set_ratio(word, company_name)
    return similarity


query_list = []
### query for finding top assignees/companies in a given time period with a keyword
for keyword in keywords:
    query1 = """
    WITH A AS 
        (WITH temp1 AS (
            SELECT
              DISTINCT
              PUB.country_code,
              PUB.application_number AS patent_number,
              assignee_name,
              PUB.publication_date

            FROM
              `patents-public-data.patents.publications` PUB,
              UNNEST(PUB.title_localized) AS title
            CROSS JOIN
              UNNEST(PUB.assignee) AS assignee_name
            WHERE
              PUB.country_code IS NOT NULL
              AND PUB.application_number IS NOT NULL
              AND PUB.inventor IS NOT NULL
              AND PUB.publication_date >= {}
              AND PUB.publication_date <= {}
              AND title.text LIKE '%{}%' 
        )
        SELECT
          *
        FROM (
            SELECT
             temp1.country_code AS country,
             temp1.assignee_name AS assignee_name,
             COUNT(temp1.patent_number) AS count_of_patents,
             MAX(temp1.publication_date) AS max_pub_date
            FROM temp1
            GROUP BY
             temp1.country_code,
             temp1.assignee_name
             )
        WHERE
         count_of_patents > 10 # Can be adjusted
        )
    SELECT
      A.assignee_name,
      A.count_of_patents,
      A.max_pub_date,
      A.country
    FROM
        A
    ORDER BY assignee_name
    ;
    """.format(date1,date2,keyword)

    print('Amount of memory estimated to be used in query:',bq_assistant.estimate_query_size(query1) )
    query_results = patents.query_to_pandas_safe(query1, max_gb_scanned=16)
    query_results.columns = ['Company_Name', 'Total_Patents', 'Last_Patent_Date', 'Country']
    print("Number of records from {} to {} for keyword '{}':".format(date1,date2, keyword), len(query_results.index))
    query_results.head(10)

    ###
    ### 
    query2 = """
    WITH B AS
    (SELECT DISTINCT
      assignee_name,
      PUB.publication_date as publication_date,
      title.text as title_text,
      PUB.country_code as country
    FROM
      `patents-public-data.patents.publications` PUB,
      UNNEST(PUB.title_localized) AS title
    CROSS JOIN
      UNNEST(PUB.assignee) AS assignee_name
    WHERE
      PUB.country_code IS NOT NULL
      AND PUB.application_number IS NOT NULL
      AND PUB.inventor IS NOT NULL
      AND PUB.publication_date >= {}
      AND PUB.publication_date <= {}
      AND title.text LIKE '%{}%' 
      )
    SELECT
      b.assignee_name,
      b.publication_date,
      b.title_text,
      b.country
    FROM
        B
    ;
    """.format(date1,date2,keyword)

    print('Amount of memory estimated to be used in query:',bq_assistant.estimate_query_size(query2) )
    query_results_2 = patents.query_to_pandas_safe(query2, max_gb_scanned=50)
    print("Number of records from {} to {} for keyword '{}':".format(date1,date2, keyword), len(query_results_2.index))
    query_results_2.head(10)
    print('---')
    new_df = query_results_2.loc[query_results_2.groupby('assignee_name')['publication_date'].idxmax()].sort_values(by='assignee_name')
    new_df.columns = ['Company_Name','Last_Patent_Date','Last_Patent_Name','Country']
    final_df = pd.merge(query_results,new_df, on = ['Company_Name','Last_Patent_Date','Country'])
    final_df = final_df[['Company_Name','Country','Total_Patents','Last_Patent_Date','Last_Patent_Name']]
    for index,row in final_df.iterrows():
        print(row['Company_Name'])
        company_df['similarity_score'] = company_df.apply(lambda s: highest_similarity_finder(s['Company Name'],row['Company_Name']), axis=1)
    #     print(company_df.sort_values(by=['similarity_score'],ascending=False).head())
        similar_list = company_df.sort_values(by=['similarity_score'],ascending=False).nlargest(5,columns=['similarity_score'])[['Company Name','similarity_score']].values.tolist()
        print(similar_list)
        i=1
        for company in similar_list:
            final_df.loc[index,'company_{}'.format(i)] = company[0]
            final_df.loc[index,'company_{}_similarity_score'.format(i)] = company[1]
            i+=1
    #     print('Most Similar Company Name:',company_df.loc[company_df['similarity_score'].idxmax(),'Company Name'])
        print('///')
    print(final_df.head())
    final_df.to_csv('{}_query_v1.csv'.format(keyword),encoding='utf-8-sig')
final_df

# new_df = query_results_2.loc[query_results_2.groupby('assignee_name')['publication_date'].idxmax()].sort_values(by='assignee_name')
# new_df.columns = ['Company_Name','Last_Patent_Date','Last_Patent_Name','Country']
# new_df.head()
# query_results.head()
# #merging of dataframes
# # company, country, count, date, name
# final_df = pd.merge(query_results,new_df, on = ['Company_Name','Last_Patent_Date','Country'])
# final_df = final_df[['Company_Name','Country','Total_Patents','Last_Patent_Date','Last_Patent_Name']]
# final_df
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process

# def highest_similarity_finder(word, company_name):
#     similarity = fuzz.token_set_ratio(word, company_name)
#     return similarity

# for index,row in final_df.iterrows():
#     print(row['Company_Name'])
#     company_df['similarity_score'] = company_df.apply(lambda s: highest_similarity_finder(s['Company Name'],row['Company_Name']), axis=1)
# #     print(company_df.sort_values(by=['similarity_score'],ascending=False).head())
#     similar_list = company_df.sort_values(by=['similarity_score'],ascending=False).nlargest(5,columns=['similarity_score'])[['Company Name','similarity_score']].values.tolist()
#     print(similar_list)
#     i=1
#     for company in similar_list:
#         final_df.loc[index,'company_{}'.format(i)] = company[0]
#         final_df.loc[index,'company_{}_similarity_score'.format(i)] = company[1]
#         i+=1
# #     print('Most Similar Company Name:',company_df.loc[company_df['similarity_score'].idxmax(),'Company Name'])
#     print('///')
# final_df_retail.head()
# final_df_software.head()
# final_df_e_commerce.head()
# final_df_computing.head()
# final_df_retail = final_df
# final_df_software = final_df
# final_df_e_commerce = final_df
# final_df_computing = final_df
# final_df_cloud = final_df
# final_df.to_csv('cloud_query_v1.csv',encoding='utf-8-sig')