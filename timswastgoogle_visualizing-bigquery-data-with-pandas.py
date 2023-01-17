# Import data visualization libraries.
import pandas
import matplotlib.pyplot as plt
%matplotlib inline

# We use this helper function to show how much quota a query consumed.
def print_usage(job):
    print('Processed {} bytes, {} MB billed (cache_hit={})'.format(
        job.total_bytes_processed, job.total_bytes_billed / (1024 * 1024), job.cache_hit))

# Connect to BigQuery.
from google.cloud import bigquery
client = bigquery.Client()
sql = """
SELECT year, gender, COUNT(DISTINCT name) / SUM(number) as names_per_person
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY gender, year
"""
job = client.query(sql)  # Run the query.
df = job.to_dataframe()  # Wait for the query to finish, and create a Pandas DataFrame.
print_usage(job)  # How much quota did this query consume?
df.head()
pivot = pandas.pivot_table(
    df, values='names_per_person', index='year', columns='gender')
pivot.head()
# Plot name "uniqueness" (number of distinct names per person) over time.
pivot.plot(fontsize=20, figsize=(15,7))
# Set the font size
# See: https://stackoverflow.com/a/3900167/101923
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)

# Create the plot, and add labels.
plt.figure(figsize=(15, 7))
plt.plot(pivot.index, pivot['F'], label='Female name uniqueness')
plt.plot(pivot.index, pivot['M'], label='Male name uniqueness')
plt.ylabel('Names per Person')
plt.xlabel('Year')
plt.title('US Names Uniqueness')
plt.legend()
sql = """
SELECT grsincfndrsng AS fundraising
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE grsincfndrsng > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)
df.plot.hist(bins=50, fontsize=20, figsize=(15,7))
sql = """
SELECT LOG10(grsincfndrsng) AS log_fundraising
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE grsincfndrsng > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)
df.plot.hist(bins=50, fontsize=20, figsize=(15,7))
sql = """
SELECT
  LOG10(payrolltx) AS log_payrolltx,
  LOG10(noemplyeesw3cnt) AS log_employees,
  LOG10(officexpns) AS log_officeexpns,
  LOG10(legalfees) AS log_legalfees
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE noemplyeesw3cnt > 0
  AND payrolltx > 0
  AND officexpns > 0
  AND legalfees > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)
df.plot.scatter(
    x='log_employees',
    y='log_payrolltx',
    fontsize=20,
    figsize=(15,7))
import pandas.plotting
pandas.plotting.scatter_matrix(df, alpha=0.2, figsize=(15, 15))
