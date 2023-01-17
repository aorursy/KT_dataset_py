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



patents = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="patents")



# View table names under the patents data table

bq_assistant = BigQueryHelper("patents-public-data", "patents")

bq_assistant.list_tables()
# View information on all columns in the publications data table

#bq_assistant.table_schema("publications")
# View the first three rows of the publications data table

bq_assistant.head("publications", num_rows=3)
df_sample = bq_assistant.head("publications", num_rows=3)

df_sample.columns
# number of publications by country

query1 = """

SELECT COUNT(*) AS cnt, country_code

FROM (

  SELECT ANY_VALUE(country_code) AS country_code

  FROM `patents-public-data.patents.publications` AS pubs

  GROUP BY application_number

)

GROUP BY country_code

ORDER BY cnt DESC

        """

#bq_assistant.estimate_query_size(query1) 



applications = patents.query_to_pandas_safe(query1, max_gb_scanned=3)

applications.head()
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='country_code', y='cnt', data=applications, color='grey')



plt.title("Total applications per country", loc='center')

plt.ylabel('#Applications')

plt.xlabel('Country')



#ax.set_yscale('log')

sns.despine();
query1 = """

SELECT

  SUM(year_cnt) AS total_count,

  assignee_name,

  ARRAY_AGG(STRUCT<cnt INT64, filing_year INT64, countries STRING>(year_cnt, filing_year, countries) ORDER BY year_cnt DESC LIMIT 1)[SAFE_ORDINAL(1)] AS largest_year

FROM (

  SELECT SUM(year_country_cnt) AS year_cnt, assignee_name, filing_year, STRING_AGG(country_code ORDER BY year_country_cnt DESC LIMIT 5) AS countries

  FROM (

    SELECT COUNT(*) AS year_country_cnt, a.name AS assignee_name, CAST(FLOOR(filing_date / 10000) AS INT64) AS filing_year, apps.country_code

    FROM (

      SELECT ANY_VALUE(assignee_harmonized) AS assignee_harmonized, ANY_VALUE(filing_date) AS filing_date, ANY_VALUE(country_code) AS country_code

      FROM `patents-public-data.patents.publications` AS pubs

      WHERE (SELECT MAX(TRUE) FROM UNNEST(pubs.cpc) AS c WHERE REGEXP_CONTAINS(c.code, "A61K39"))

      GROUP BY application_number

    ) AS apps, UNNEST(assignee_harmonized) AS a

    WHERE filing_date > 0

    GROUP BY a.name, filing_year, country_code

  )

  GROUP BY assignee_name, filing_year

)

GROUP BY assignee_name

ORDER BY total_count DESC

LIMIT 20;

"""

#bq_assistant.estimate_query_size(query1) 



assignee = patents.query_to_pandas_safe(query1, max_gb_scanned=9)

assignee.head()
figure(num=None, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')



sns.set(context='poster', style='ticks', font_scale=0.6)

# Reorder it following the values:

my_range=range(1,len(assignee.index)+1)



# Create a color if the group is "B"

my_color=np.where(assignee['total_count'] >= 1000, '#5ab4ac', '#d8b365')

my_size=np.where(assignee['total_count'] >= 0, 70, 30)



# The vertival plot is made using the hline function

# I load the seaborn library only to benefit the nice looking feature

plt.hlines(y=my_range, xmin=0, xmax=assignee['total_count'], color=my_color, alpha=1)

plt.scatter(assignee['total_count'], my_range, color=my_color, s=my_size, alpha=1)



# Add title and exis names



plt.yticks(my_range, assignee['assignee_name'])

plt.title("Most applications filed per assignee", loc='left')

plt.xlabel('Number of applications')

plt.ylabel('')

sns.despine();
query1 = """

SELECT AVG(num_inventors), COUNT(*) AS cnt, country_code, filing_year, STRING_AGG(publication_number LIMIT 10) AS example_publications

FROM (

  SELECT ANY_VALUE(publication_number) AS publication_number, ANY_VALUE(ARRAY_LENGTH(inventor)) AS num_inventors, ANY_VALUE(country_code) AS country_code, ANY_VALUE(CAST(FLOOR(filing_date / (5*10000)) AS INT64))*5 AS filing_year

  FROM `patents-public-data.patents.publications` AS pubs

  WHERE filing_date > 19000000 AND ARRAY_LENGTH(inventor) > 0

  GROUP BY application_number

)

GROUP BY filing_year, country_code

HAVING cnt > 100

ORDER BY filing_year

"""

# bq_assistant.estimate_query_size(query1) 



inventors = patents.query_to_pandas_safe(query1, max_gb_scanned=8)

inventors.head()
from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')



ax = sns.boxplot(x="country_code", y="f0_",

                 data=inventors, 

                 fliersize=True)



plt.title("Average number of inventors per country  ", loc='center')

plt.ylabel('')

plt.xlabel('')



sns.despine();
from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6, palette='colorblind')



palette = sns.color_palette("deep", 17)



ax = sns.lineplot(x='filing_year', y='f0_', data=inventors, hue='country_code', lw=1, palette=palette)



ax.set_ylim(0, 5)

ax.set_xlim(1900, 2018)

plt.title("Average number of inventors per country  ", loc='center')

plt.ylabel('')

plt.xlabel('Year')

plt.legend(loc=2)

sns.despine();
# number of publications by country

query1 = """

SELECT 

  country_code,

  COUNT(DISTINCT publication_number) AS publications

FROM

  `patents-public-data.patents.publications`

WHERE publication_date > 0

    AND application_kind = 'A'

GROUP BY country_code

ORDER BY publications DESC;

        """

#bq_assistant.estimate_query_size(query1) 



country = patents.query_to_pandas_safe(query1, max_gb_scanned=3)

country.head()
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='country_code', y='publications', data=country, color='grey')



plt.title("Total patents published", loc='center')

plt.ylabel('$log$ #Publications')

plt.xlabel('Country')



#ax.set_yscale('log')

sns.despine();
# number of publications with date

query1 = """

WITH top5 AS (

SELECT 

  country_code,

  COUNT(DISTINCT publication_number) AS publications

FROM

  `patents-public-data.patents.publications`

WHERE publication_date > 0

    AND application_kind = 'A'

GROUP BY country_code

ORDER BY publications DESC

LIMIT 5)



SELECT 

  ROUND(publication_date/10000)  as datum,

  country_code,

  COUNT(DISTINCT publication_number) AS publications

FROM

  `patents-public-data.patents.publications`

WHERE publication_date > 19000000

    AND publication_date < 20180000

    AND application_kind = 'A'

    AND country_code IN (SELECT country_code FROM top5)

GROUP BY datum, country_code

ORDER BY datum DESC, country_code;

        """

# bq_assistant.estimate_query_size(query1) 



yearly = patents.query_to_pandas_safe(query1, max_gb_scanned=3)

yearly['datum'] = np.int64(yearly.datum)

yearly.head()
from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')



ax = sns.boxplot(x="country_code", y="publications",

                 data=yearly, 

                 fliersize=False)



#ax.set_ylim(0, 1e6)

plt.title("Publication of patents per year", loc='center')

plt.ylabel('')

plt.xlabel('')

plt.yscale('log')

sns.despine();
from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6, palette='colorblind')



palette = sns.color_palette("deep", 5)



ax = sns.lineplot(x='datum', y='publications', data=yearly, hue='country_code', lw=4, palette=palette)



ax.set_xlim(1900, 2017)

#ax.set_yscale('log')

plt.title("Publication of patents per year", loc='center')

plt.ylabel('Number of Publication')

plt.xlabel('Year')

plt.legend(loc=2)

sns.despine();
from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6, palette='colorblind')



palette = sns.color_palette("deep", 5)



ax = sns.lineplot(x='datum', y='publications', data=yearly, hue='country_code', lw=4, palette=palette)



ax.set_xlim(2000, 2017)

#ax.set_yscale('log')

plt.title("2011 China emerges as the leader", loc='center')

plt.ylabel('Number of Publication')

plt.xlabel('Year')

plt.legend(loc=2)

sns.despine();
query2 = """

SELECT 

  FLOOR(grant_date/10000) as datum,

  COUNT(DISTINCT publication_number) as publications

FROM

  `patents-public-data.patents.publications`

WHERE country_code = 'CN'

    AND grant_date > 20110000

    AND grant_date < 20180000

    AND application_kind = 'A'

GROUP BY datum, application_kind

ORDER BY application_kind, datum

    ;

        """

#bq_assistant.estimate_query_size(query2) 



cn_yearly = patents.query_to_pandas_safe(query2, max_gb_scanned=3)

cn_yearly['datum'] = np.int64(cn_yearly.datum)

cn_yearly
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)



ax = sns.barplot(x='datum', y='publications', data=cn_yearly, color='grey')



plt.title("Total patents granted to China", loc='center')

plt.ylabel('#Patents')

plt.xlabel('')

sns.despine();
# kind_code = 'B'

# Examined APP. open to Public inspection

query2 = """

SELECT 

    publication_number

FROM

  `patents-public-data.patents.publications`

WHERE country_code = 'CN'

    AND grant_date > 20170000

    AND grant_date < 20180000

    AND application_kind = 'A'

    AND kind_code = 'B'

    ;

        """

#bq_assistant.estimate_query_size(query2) 



cn_publication_number = patents.query_to_pandas_safe(query2, max_gb_scanned=4)

cn_publication_number.head()
cn_publication_number.shape[0]
# Save publication number of Patents granted in 2017

cn_publication_number.to_csv('cn_publication_number.csv', index=False)