import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))



import bq_helper

from bq_helper import BigQueryHelper

medicare = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'cms_medicare')
med_data = BigQueryHelper('bigquery-public-data','cms_medicare')
med_data.list_tables()
med_data.head('home_health_agencies_2013')
query1 = """

SELECT DISTINCT state, SUM(total_episodes_non_lupa) AS non_lupa

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

GROUP BY state

ORDER BY non_lupa DESC

"""

non_lupa_state = medicare.query_to_pandas_safe(query1)

non_lupa_state.head(10)
nonlupa_10 = non_lupa_state.head(10)

nonlupa_10.set_index('state', inplace = True)

nonlupa_10.plot(kind = 'barh', figsize = (12,6), title = '10 states with the most non-LUPA episodes')
query2 = """

SELECT DISTINCT city, SUM(total_episodes_non_lupa) AS non_lupa

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

WHERE state = "TX"

GROUP BY city

ORDER BY non_lupa DESC

"""

non_lupa_TX = medicare.query_to_pandas_safe(query2)

non_lupa_TX.head(20)
texas_10 = non_lupa_TX.head(10)

texas_10.set_index('city', inplace = True)

texas_10.plot(kind = 'barh',

              figsize = (12,6),

              color = 'r',

              title = '10 cities in Texas with most non-LUPA episodes')
query3 = """

SELECT DISTINCT city, SUM(total_episodes_non_lupa) AS non_lupa

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

WHERE state = "FL"

GROUP BY city

ORDER BY non_lupa DESC

"""

non_lupa_FL = medicare.query_to_pandas_safe(query3)

non_lupa_FL.head(20)
florida_10 = non_lupa_FL.head(10)

florida_10.set_index('city', inplace = True)

florida_10.plot(kind = 'barh',

                figsize = (12,6),

                color = 'm',

                title = '10 cities in Florida with most non-LUPA episodes')
query4 = """

SELECT DISTINCT state, SUM(total_lupa_episodes) AS lupa

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

GROUP BY state

ORDER BY lupa DESC

"""



lupa_state = medicare.query_to_pandas_safe(query4)
lupa_10 = lupa_state.head(10)

lupa_10.set_index('state',inplace = True)

lupa_10.plot(kind = 'barh',

             color = 'c',

             figsize = (12,6),

             title = 'Top 10 states with most LUPA episodes')
query5 = """

SELECT DISTINCT city, SUM(total_lupa_episodes) AS lupa

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

WHERE state = "TX"

GROUP BY city

ORDER BY lupa DESC

"""

lupa_TX = medicare.query_to_pandas_safe(query5)

lupa_TX.head(20)
texas_10_lupa = lupa_TX.head(10)

texas_10_lupa.set_index('city', inplace = True)

texas_10_lupa.plot(kind = 'barh',

              figsize = (12,6),

              color = 'g',

              title = '10 cities in Texas with most LUPA episodes')
query6 = """

SELECT 

state,SUM(total_hha_charge_amount_non_lupa) AS hha_charge , SUM(total_hha_medicare_standard_payment_amount_non_lupa) AS medicare

FROM `bigquery-public-data.cms_medicare.home_health_agencies_2013`

GROUP BY state

ORDER BY medicare DESC

"""



non_lupa_charges = medicare.query_to_pandas_safe(query6)

non_lupa_charges.head(10)
non_lupa_charge_10 = non_lupa_charges.head(10)

non_lupa_charge_10.set_index('state', inplace = True)

non_lupa_charge_10.plot(kind = 'bar', figsize = (12,6))

plt.xlabel('Charge(in billions)')
med_data.head('hospice_providers_2014',2)
query7 = """

SELECT

state, SUM(hospice_beneficiaries) AS total_beneficiaries,

SUM(male_hospice_beneficiaries) AS male,

SUM(female_hospice_beneficiaries) AS female

FROM

`bigquery-public-data.cms_medicare.hospice_providers_2014`

GROUP BY state

ORDER BY total_beneficiaries DESC

"""



total_state_ben = medicare.query_to_pandas_safe(query7)

total_state_ben
state_10 = total_state_ben.head(10)

state_10.set_index('state',inplace = True )

state_10.plot(kind = 'bar', figsize = (14,8), title = '10 states with the most no. of beneficiaries')
query8 = """

SELECT state, ROUND(AVG(average_age)) AS mean_age

FROM `bigquery-public-data.cms_medicare.hospice_providers_2014`

GROUP BY state

ORDER BY mean_age

"""

avg_age_state = medicare.query_to_pandas_safe(query8)

avg_age_state
sns.distplot(avg_age_state['mean_age'])

sns.set(rc = {'figure.figsize':(10,8)})
query9 = """

SELECT 

state,

SUM(white_hospice_beneficiaries) AS white, 

SUM(black_hospice_beneficiaries) AS black, 

SUM(asian_hospice_beneficiaries) AS asian, 

SUM(hispanic_hospice_beneficiaries) AS hispanic, 

SUM(other_unknown_race_hospice_beneficiaries) AS others

FROM `bigquery-public-data.cms_medicare.hospice_providers_2014`

GROUP BY  state

ORDER BY white DESC

"""



ben_race_states = medicare.query_to_pandas_safe(query9)

ben_race_states.head(20)
ben_race_states_10 = ben_race_states.head(10)

ben_race_states_10.set_index('state', inplace = True)

ben_race_states_10.plot(kind = 'bar',

                        figsize = (14,8),

                        title = ' Top 10 statewise beneficiaries distributed by race')
query10 = """

SELECT 

state,

SUM(hospice_beneficiaries_with_seven_or_fewer_hospice_care_days) AS week_or_less,

SUM(hospice_beneficiaries_with_more_than_sixty_hospice_care_days) AS two_months_or_less,

SUM(hospice_beneficiaries_with_more_than_one_hundred_eighty_hospice_care_days) AS six_months_or_more

FROM `bigquery-public-data.cms_medicare.hospice_providers_2014`

GROUP BY state

ORDER BY two_months_or_less DESC

"""

ben_by_days_state = medicare.query_to_pandas_safe(query10)

ben_by_days_state.head(20)
ben_by_days_10 = ben_by_days_state.head(10)

ben_by_days_10.set_index('state', inplace = True)

ben_by_days_10.plot(kind = 'bar',

                    figsize = (14,8),

                    title = 'Statewise beneficiaries by no. of days under hospice')