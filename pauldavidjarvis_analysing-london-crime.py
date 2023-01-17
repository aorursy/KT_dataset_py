# Imports
import numpy as np 
import pandas as pd 
import os
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/content/drive/My Drive/Colab Notebooks/google.json" #Google's API
london = bigquery.Client() #Defines the variable "London" to use when referencing dataset
ds_ref = london.dataset("london_crime", project = "bigquery-public-data")
crime_london = london.get_dataset(ds_ref)
tables = list(london.list_tables(crime_london))
for table in tables:
    print(table.table_id)
table_ref = ds_ref.table("crime_by_lsoa")
crime = london.get_table(table_ref)
crime.schema
london.list_rows(crime, max_results = 5).to_dataframe()
# Prints all boroughs
borough_count=  """
SELECT distinct borough 
FROM `bigquery-public-data.london_crime.crime_by_lsoa`;
"""
borough_count = london.query(borough_count).result().to_dataframe()
borough_count.head(5) #Change the number to change the number of boroughs that are outputted or remove to see them all
# Prints all Crime types
crime_type=  """
SELECT distinct major_category, minor_category 
FROM `bigquery-public-data.london_crime.crime_by_lsoa`
ORDER BY major_category ASC;
"""
crime_type = london.query(crime_type).result().to_dataframe()
crime_type
# Calculates the Borough with most crime
crimes_by_borough=  """
SELECT year, borough, SUM(value) AS num_crimes
FROM `bigquery-public-data.london_crime.crime_by_lsoa` 
WHERE year = (select max(year) FROM `bigquery-public-data.london_crime.crime_by_lsoa`)
GROUP BY year, borough ORDER BY num_crimes DESC
"""
crimes_by_borough = london.query(crimes_by_borough).result().to_dataframe()
crimes_by_borough

#Crime Numbers for each Crime Category
crimes_by_type= """
SELECT year, major_category, SUM(value) AS num_crimes  
FROM `bigquery-public-data.london_crime.crime_by_lsoa` 
WHERE year = (select max(year) FROM `bigquery-public-data.london_crime.crime_by_lsoa`)
GROUP BY year, major_category ORDER BY num_crimes DESC
"""
crimes_by_type = london.query(crimes_by_type).result().to_dataframe()
crimes_by_type
crimes_in_boroughs = """
SELECT borough, major_category, minor_category, SUM(value) AS num_crimes
FROM `bigquery-public-data.london_crime.crime_by_lsoa`
WHERE (major_category = "Theft and Handling" OR major_category = "Violence Against the Person" OR major_category ="Burglary") 
AND borough = "Brent" AND year = 2016
GROUP BY  major_category,borough, minor_category
ORDER BY borough, major_category DESC
"""

crimes_in_boroughs = london.query(crimes_in_boroughs).result().to_dataframe()

print(crimes_in_boroughs.head(12)) #Can do 12 before .......
crime_years= """
SELECT year, SUM(value) AS num_crimes  
FROM `bigquery-public-data.london_crime.crime_by_lsoa` 
GROUP BY year ORDER BY year ASC
"""
crime_years = london.query(crime_years).result().to_dataframe()
crime_years 
drug_trafficking_crimes= """
SELECT borough, major_category, minor_category, SUM(value) AS num_crimes
FROM `bigquery-public-data.london_crime.crime_by_lsoa`
WHERE major_category = "Drugs" AND minor_category = "Drug Trafficking"
GROUP BY  major_category,borough, minor_category
ORDER BY num_crimes DESC
"""

drug_trafficking_crimes = london.query(drug_trafficking_crimes).result().to_dataframe()
drug_trafficking_crimes.head(5)
crimes_by_year= """
SELECT year,month, SUM(value) AS num_crimes  
FROM `bigquery-public-data.london_crime.crime_by_lsoa` 
WHERE major_category ="Burglary"
AND year = 2016
GROUP BY year, month ORDER BY year, month ASC
"""    

# run the query, and convert the results to a pandas DataFrame
crimes_by_year = london.query(crimes_by_year).result().to_dataframe()
df = crimes_by_year
df  
months = []
values = []
df.tail(1)
df = df.head(len(df)-1)
df
df.shape
df_month = df.loc[:, 'month']
df_value = df.loc[:, 'num_crimes']
for month in df_month:
  months.append([int(month)])
  


for value in df_value:
  values.append(int(value))
print(months)
print(values)
def predict_crime(months, values, x):

  #Creates 3 SVR models
  svr_lin = SVR(kernel='linear', C=1e3)
  svr_poly = SVR(kernel='poly', C=1e3)
  svr_rbf = SVR(kernel='rbf', C=1e3)

  #Trains the SVR models
  svr_lin.fit(months, values)
  svr_poly.fit(months, values)
  svr_rbf.fit(months, values)

  #Creates & Trains the Linear Regression Model
  lin_reg = LinearRegression()
  lin_reg.fit(months, values)

  #Plots all models and data onto a graph
  plt.scatter(months, values, color='black', label='Data')
  plt.plot(months, svr_rbf.predict(months), color='red', label='SVR RBF')
  plt.plot(months, svr_poly.predict(months), color='blue', label='SVR POLY')
  plt.plot(months, svr_lin.predict(months), color='green', label='SVR LINEAR')
  plt.plot(months, lin_reg.predict(months), color='orange', label='LINEAR REG')
  plt.xlabel('Month')
  plt.ylabel('Number of Crimes')
  plt.title('Regression')
  plt.legend()
  plt.show()

  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0], lin_reg.predict(x)[0]
predicted_crime = predict_crime(months, values, [[10]])
print(predicted_crime)