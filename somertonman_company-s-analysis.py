import pandas as pd

import sqlite3

import seaborn as sns





from datetime import datetime

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

conn = sqlite3.connect('/kaggle/input/sellyourstuffdb/company (7).db')
sql = '''

SELECT 

  transaction_date, 

  t.account, 

  clt.residence Country_of_residence, 

  SUM(total_buy)/ 100 Total_profit 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

GROUP BY 

  clt.residence 

ORDER BY 

  Total_profit DESC 

LIMIT 

  10

'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x='Country_of_residence', y='Total_profit',kind='bar', figsize=(20,10))

plt.xlabel('Country')

plt.ylabel('Total profit')

plt.title('Total profits by country of residence on absolute scale')

plt.show()
sql = '''

SELECT 

  transaction_date, 

  t.account, 

  clt.residence Country_of_residence, 

  sum(total_buy)/ 100 Total_profit 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

GROUP BY 

  clt.residence 

ORDER BY 

  Total_profit DESC 

LIMIT 

  10

'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x='Country_of_residence', y='Total_profit',kind='bar', figsize=(20,10), log=True)

plt.xlabel('Country')

plt.ylabel('Total profit on log scale')

plt.title('Total profits by country of residence on log scale')

plt.show()
sql = '''

SELECT 

  DISTINCT clt.residence as Country, 

  avg(total_buy / 100) as Average_profit 

FROM 

  transactions t 

  LEFT JOIN clients clt ON clt.account = t.account 

WHERE 

  total_buy > 0 

GROUP BY 

  Country 

ORDER by 

  Average_profit DESC

LIMIT 

  10

'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x='Country', y='Average_profit',kind='bar', figsize=(20,10), log=True)

plt.xlabel('Country')

plt.ylabel('Average profit')

plt.title('Average profit by country per daily transactions per user')

plt.show()

sql = '''

SELECT 

  DISTINCT clt.indication_coupon as Coupon, 

  total(t.total_buy / 100) Total_profit 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

WHERE 

  total_buy > 0 

GROUP by 

  Coupon 

ORDER by 

  Total_profit DESC 

LIMIT 

  15

'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x='Coupon', y='Total_profit',kind='bar', figsize=(20,10), log=True)

plt.xlabel('Coupon')

plt.ylabel('Average profit')

plt.title('Total profit by indication coupons on a log scale')

plt.show()
sql = '''

SELECT 

  DISTINCT clt.indication_coupon as Coupon, 

  clt.residence as Country, 

  total(t.total_buy / 100) Total_profit 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

WHERE 

  total_buy > 0 

  and indication_coupon = 308 

GROUP by 

  residence 

ORDER by 

  Total_profit DESC

LIMIT 

  15

'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x='Country', y='Total_profit',kind='bar', figsize=(20,10), log=True)

plt.xlabel('Country')

plt.ylabel('Average profit')

plt.title('Total profit by indication coupon 308 on a log scale')

plt.show()
sql = '''

SELECT 

  t.transaction_date as Date, 

  clt.residence as Country, 

  total(total_buy / 100) as Profit 

FROM 

  transactions t 

  LEFT JOIN clients clt ON clt.account = t.account 

WHERE 

  total_buy > 0 

GROUP BY 

  transaction_date, 

  residence

'''



profits_by_dates_countries = pd.read_sql_query(sql,conn)



## since we don't have a pivot feature in SQLite, we have to implement this functionality in Pandas:



profits_by_dates_countries_pivot = pd.pivot_table(profits_by_dates_countries,index='Date',columns='Country',values='Profit')



## converting index to datetime object

profits_by_dates_countries_pivot.index = pd.to_datetime(profits_by_dates_countries_pivot.index)



## adding columns with Month/Year/Year-Month for better trends aggregated visualization



profits_by_dates_countries_pivot['Month'] = profits_by_dates_countries_pivot.index.month

profits_by_dates_countries_pivot['Year'] = profits_by_dates_countries_pivot.index.year

profits_by_dates_countries_pivot['Year-Month'] = profits_by_dates_countries_pivot.index.year.map(str) + '-'+profits_by_dates_countries_pivot.index.month.map(str)



profits_by_dates_countries_pivot[['id','br','ng','ru']].head(10)
### list of the top countries by most accessible values (i.e. least number of missing values for visualization)



NUMBER_OF_COUNTRIES = 5

topcountries = profits_by_dates_countries_pivot.isnull().sum(axis = 0).sort_values(ascending=True)[3:][:NUMBER_OF_COUNTRIES].index.values.tolist() 





## plotting selected number of countries 



fig, axes = plt.subplots(NUMBER_OF_COUNTRIES, 1, figsize=(20, 30), sharex=True)

for name, ax in zip(topcountries, axes):

    sns.boxplot(data=profits_by_dates_countries_pivot, x='Month', y=name, ax=ax)

    ax.set_ylabel('Monthly profit')

    ax.set_title('Boxplots with montly profit for country {}'.format(name))
sql = '''

SELECT 

  DISTINCT CASE WHEN (c.country = '') THEN 'global' WHEN (clt.residence = c.country) THEN 'local' ELSE 'no' END as Is_campaign, 

  avg(t.total_buy / 100) Profit_per_user 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

  LEFT JOIN campaigns c ON t.transaction_date BETWEEN c.start_date 

  AND c.end_date 

WHERE 

  total_buy > 0 

GROUP BY 

  Is_campaign

'''



overall_effect = pd.read_sql_query(sql,conn).set_index('Is_campaign')



##plotting



overall_effect.plot.barh(figsize=(20,5))

plt.ylabel('Campaign status')

plt.xlabel('Average profit per total daily transactions per user')

plt.title('Average profit per total daily transactions per user depending on campaign status')

plt.show()
sql = '''

SELECT 

  DISTINCT t.transaction_date as Date, 

  CASE WHEN (c.country = '') THEN 'global' WHEN (clt.residence = c.country) THEN 'local' ELSE 'no' END as Is_campaign, 

  (t.total_buy / 100) Profit_per_user 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

  LEFT JOIN campaigns c ON t.transaction_date BETWEEN c.start_date 

  AND c.end_date 

WHERE 

  total_buy > 0 

ORDER BY 

  t.transaction_date

'''



campaigns_on_time_lines = pd.read_sql_query(sql,conn) 



##plotting



pd.pivot_table(campaigns_on_time_lines,index='Date',columns='Is_campaign',values='Profit_per_user').plot(kind='line',figsize=(20,10))

plt.xlabel('Date')

plt.ylabel('Average daily profit per user depending on campaign status')

plt.title('Average daily profit per user depending on campaign status on timeline')

plt.show()
sql = '''

SELECT 

  DISTINCT c.start_date as Campaign_start, 

  c.end_date as Campaign_end, 

  julianday(c.end_date) - julianday(c.start_date)+ 1 as Camp_duration, 

  round(

    total(t.total_buy / 100), 

    3

  ) as Average_profit, 

  round(

    total(t.total_buy) /(total_spend), 

    3

  ) as Camp_ratio, 

  c.country as Country, 

  CASE WHEN (c.country = '') THEN 'global' WHEN (clt.residence = c.country) THEN 'local' ELSE 'no' END as Camp_type, 

  round(

    total_spend /(

      julianday(c.end_date) - julianday(c.start_date) + 1

    )

  ) as Camp_costs_daily, 

  (c.total_spend) as Camp_costs_total 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

  LEFT JOIN campaigns c ON t.transaction_date BETWEEN c.start_date 

  AND c.end_date 

WHERE 

  total_buy > 0 

  AND (

    clt.residence = c.country 

    OR c.country = '' 

    OR c.country = 'global'

  ) 

GROUP BY 

  c.start_date 

ORDER BY 

  Camp_ratio

'''



z = pd.read_sql_query(sql,conn)



## plotting



plt.figure(figsize=(20,15))

sns.scatterplot(data=z, x='Camp_ratio', y='Camp_costs_daily', hue = 'Country',size="Average_profit", sizes=(40, 300)).set_title('Average profits by country and campaign sorted by campaign efficiency')

plt.xlabel('Campaign efficiency rank')

plt.ylabel('Campaign costs per day')

plt.title('Campaign efficiency per country, average profit and costs')

plt.show()
sql = '''

SELECT 

  DISTINCT c.start_date as Campaign_started, 

  sum(t.total_buy / 100) Total_profit, 

  avg(t.total_buy / 100) Average_profit, 

  c.country, 

  CASE WHEN (c.country = '') THEN 'global' WHEN (clt.residence = c.country) THEN 'local' ELSE 'no' END as is_camp, 

  julianday(c.end_date) - julianday(c.start_date)+ 1 as camp_duration, 

  total_spend /(

    julianday(c.end_date) - julianday(c.start_date) + 1

  ) as camp_costs_per_day, 

  (c.total_spend) 

FROM 

  transactions t 

  INNER JOIN clients clt ON t.account = clt.account 

  LEFT JOIN campaigns c ON t.transaction_date BETWEEN c.start_date 

  AND c.end_date 

WHERE 

  total_buy > 0 

  AND (

    clt.residence = c.country 

    OR c.country = '' 

    OR c.country = 'global'

  ) 

GROUP BY 

  Campaign_started 

ORDER BY 

  Campaign_started



'''



z = pd.read_sql_query(sql,conn)



## plotting



z.plot(x= 'Campaign_started',y = 'Average_profit',kind='line', figsize=(20,10))

plt.xlabel('Date')

plt.ylabel('Average daily profit')

plt.title('Average daily profits of all campaigns on timeline')

plt.show()