# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', -1)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')



from google.cloud import bigquery
client = bigquery.Client()



dataset_ref = client.dataset('world_bank_health_population', project='bigquery-public-data')



dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



list_of_tables = [table.table_id for table in tables]



print(list_of_tables)
table_ref = dataset_ref.table('country_series_definitions')



table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
table_ref = dataset_ref.table('country_summary')
table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
table_ref = dataset_ref.table('health_nutrition_population')



table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
table_ref = dataset_ref.table('series_summary')



table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
table_ref = dataset_ref.table('series_times')



table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
qseries = '''

     SELECT DISTINCT series_code, topic, indicator_name

     FROM `bigquery-public-data.world_bank_health_population.series_summary`

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

qseries_job = client.query(qseries, job_config=safe_config)



qseries_results = qseries_job.to_dataframe()



qseries_results
q1 = '''

     SELECT country_name, indicator_name, year, value

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

     WHERE indicator_code = 'SP.DYN.AMRT.MA' AND country_name = 'Canada' OR indicator_code = 'SP.DYN.AMRT.FE' AND country_name = 'Canada'

     ORDER BY year

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q1_job = client.query(q1, job_config=safe_config)



q1_results = q1_job.to_dataframe()



q1_results
q1 = q1_results[['year', 'value']]



year_lst = []

for x in q1['year']:

    year_lst.append(x)

year_lst = year_lst[0::2]



male_lst = []

for x in q1['value']:

    male_lst.append(x)

male_lst = male_lst[1::2]



female_lst = []

for x in q1['value']:

    female_lst.append(x)

female_lst = female_lst[0::2]



q1_data = {'Year': year_lst,

           'Male': male_lst,

           'Female': female_lst

          }



q1_df = pd.DataFrame(q1_data, columns = ['Year','Male', 'Female'])

q1_df = q1_df.set_index('Year')



plt.figure(figsize=(10,7))

plt.title('Adult Mortality Rate Based on Gender in Canada')

plt.ylabel('Mortality Rate (per 1000)')

sns.lineplot(data=q1_df)
q2 = '''

     SELECT country_name, indicator_name, year, value

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

     WHERE indicator_code = 'SP.POP.0014.TO.ZS' AND country_name = 'High income' 

         OR indicator_code = 'SP.POP.0014.TO.ZS' AND country_name = 'Middle income' 

         OR indicator_code = 'SP.POP.0014.TO.ZS' AND country_name = 'Low income'

     ORDER BY year

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q2_job = client.query(q2, job_config=safe_config)



q2_results = q2_job.to_dataframe()



q2_results
q2 = q2_results[['year', 'value']]



year_lst = []

for x in q2['year']:

    year_lst.append(x)

year_lst = year_lst[0::3]



high_income_lst = []

for x in q2['value']:

    high_income_lst.append(x)

high_income_lst = high_income_lst[0::3]



low_income_lst = []

for x in q2['value']:

    low_income_lst.append(x)

low_income_lst = low_income_lst[1::3]



middle_income_lst = []

for x in q2['value']:

    middle_income_lst.append(x)

middle_income_lst = middle_income_lst[2::3]



q2_data = {'Year': year_lst,

           'Low Income': low_income_lst,

           'Middle Income': middle_income_lst,

           'High Income': high_income_lst

          }



q2_df = pd.DataFrame(q2_data, columns=['Year', 'Low Income', 'Middle Income', 'High Income'])

q2_df = q2_df.set_index('Year')

q2_df = q2_df.iloc[0::10]

q2_df



plt.figure(figsize=(5,10))

plt.title('Children Population Percentage Based on Income Class')

sns.heatmap(data=q2_df, annot=True)
q3 = '''

     SELECT country_name, indicator_name, year, value

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

     WHERE indicator_code = 'SH.STA.OWAD.FE.ZS' AND country_name = 'Canada' 

         OR indicator_code = 'SH.STA.OWAD.MA.ZS' AND country_name = 'Canada'

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q3_job = client.query(q3, job_config=safe_config)



q3_results = q3_job.to_dataframe()



q3_results
gender = []



for x in range(0,42):

    x = 'Female'

    gender.append(x)

for y in range(0,42):

    y ='Male'

    gender.append(y)



q3_results['Gender'] = gender



q3 = q3_results[['year', 'value', 'Gender']]

q3.columns = ['Year', 'Overweight Percentage', 'Gender']





plt.figure(figsize=(7,5))

plt.title('Canadian Overweight Percentage Based on Gender')

sns.scatterplot(x='Year', y='Overweight Percentage', hue='Gender', data=q3)
q5 = '''

     SELECT country_name, indicator_name, year, value

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

     WHERE indicator_code = 'SL.UEM.TOTL.ZS' AND country_name = 'Canada'

         OR indicator_code = 'SL.UEM.TOTL.ZS' AND country_name = 'United States'

         OR indicator_code = 'SL.UEM.TOTL.ZS' AND country_name = 'Mexico'

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q5_job = client.query(q5, job_config=safe_config)



q5_results = q5_job.to_dataframe()



q5_results
q5_mexico = q5_results.iloc[28:83]

q5_usa = q5_results.iloc[26:27]

q5_canada = q5_results.iloc[0:27]



sns.distplot(a=q5_mexico['value'], label='Mexico', kde=False)

sns.distplot(a=q5_usa['value'], label='United States', kde=False)

sns.distplot(a=q5_canada['value'], label='Canada', kde=False)



plt.title("Histogram of North American Country's Unemployment %")

plt.xlabel('Percentage')

plt.legend()
q6 = '''

     WITH smoking AS

     (

         SELECT year, value AS smoking_prevalence

         FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

         WHERE indicator_code = 'SH.PRV.SMOK' AND country_name = 'Canada'

     )

     SELECT le.country_name, le.indicator_code, le.year, le.value AS life_expectancy, smoking.smoking_prevalence,

     CASE

         WHEN le.indicator_code = 'SP.DYN.LE00.FE.IN' THEN 'Female'

         ELSE 'Male'

     END AS gender

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population` AS le

     RIGHT JOIN smoking

     ON le.year = smoking.year

     WHERE indicator_code = 'SP.DYN.LE00.MA.IN' AND country_name = 'Canada'

         OR indicator_code = 'SP.DYN.LE00.FE.IN' AND country_name = 'Canada'

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q6_job = client.query(q6, job_config=safe_config)



q6_results = q6_job.to_dataframe()



q6_results
sns.lmplot(x="life_expectancy", y="smoking_prevalence", hue="gender", data=q6_results)

plt.title('Life Expectancy vs Smoking Prevalence in Canada')

plt.xlabel('Life Expectancy')

plt.ylabel('Smoking Prevalence')
q6_1 = '''

        SELECT country_name, indicator_code, year, value AS life_expectancy,

        CASE

            WHEN le.indicator_code = 'SP.DYN.LE00.FE.IN' THEN 'Female'

            ELSE 'Male'

        END AS gender

        FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population` AS le

        WHERE indicator_code = 'SP.DYN.LE00.MA.IN' AND country_name = 'Canada'

            OR indicator_code = 'SP.DYN.LE00.FE.IN' AND country_name = 'Canada'

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q6_1_job = client.query(q6_1, job_config=safe_config)



q6_1_results = q6_1_job.to_dataframe()



q6_1_results
sns.swarmplot(x= q6_1_results['gender'], y=q6_1_results['life_expectancy'])

plt.title('Life Expectancy Based on Gender in Canada')

plt.xlabel('Gender')

plt.ylabel('Life Expectancy')
q7 = '''

     SELECT * 

     FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`

     WHERE indicator_code = 'SP.POP.GROW' and country_name = 'Canada'

     '''



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

q7_job = client.query(q7, job_config=safe_config)



q7_results = q7_job.to_dataframe()



q7_results
from xgboost import XGBRegressor



x = q7_results.year

y = q7_results.value



x_lst = []

for num in x:

    num = [num]

    x_lst.append(num)



y_lst = []

for num in y:

    num = [num]

    y_lst.append(num)



x_array = np.array(x_lst)

y_array = np.array(y_lst)



model = XGBRegressor(n_estimators = 2000)

model.fit(x_array, y_array)



prediction = model.predict(np.array([[2019]]))

prediction = prediction.tolist()



plot_data = q7_results[['year', 'value']]

plot_prediction = plot_data.iloc[-1:]

plot_prediction = plot_prediction.append({'year': 2019, 'value':prediction[0]}, ignore_index = True)



plot_data = plot_data.set_index('year')

plot_prediction = plot_prediction.set_index('year')



plt.figure(figsize=(14,6))



plt.title("Canada's Annual Population Growth 2019 Prediction")

sns.lineplot(data=plot_data['value'], label="Canada's Annual Population Growth %")

ax = sns.lineplot(data=plot_prediction['value'], label="Predicted Growth")

ax.lines[1].set_linestyle("--")

plt.xlabel("Year")

plt.ylabel('Growth %')

leg = ax.legend()

leg_lines = leg.get_lines()



print(prediction)