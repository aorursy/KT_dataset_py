from collections import Counter

import bq_helper
import matplotlib.pyplot as plt

from matplotlib import style

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# New query
query = """
        SELECT country 
        FROM `bigquery-public-data.openaq.global_air_quality` 
        WHERE unit != 'ppm'
        """
countries = open_aq.query_to_pandas_safe(query)
# Countries that use measurement other than ppm:
countries.country.unique()
# Create a new query
query = """
        SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

zero_pollutants = open_aq.query_to_pandas(query)
# Pollutants with value 0:
zero_pollutants.pollutant.unique()
# Plot the histogram of the zero pollutants count
style.use('ggplot')

count = Counter(zero_pollutants.pollutant.tolist())

plt.bar(count.keys(), count.values(), width=.55)
plt.title('Pollutants with zero value')
plt.ylabel('count')
plt.show()