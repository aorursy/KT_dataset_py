# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Write the code you need here to figure out the answer
num_tables = len(list(client.list_tables(dataset)))  # Store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
# Write the code to figure out the answer
crime_table_ref = dataset_ref.table('crime')

crime_table = client.get_table(crime_table_ref)

crime_table.schema
timestamp_fields = [schema_field for schema_field in crime_table.schema if schema_field.field_type == 'TIMESTAMP']

num_timestamp_fields = len(timestamp_fields) # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code

query = """SELECT latitude, longitude, primary_type FROM `bigquery-public-data.chicago_crime.crime` LIMIT 80000"""

query_job = client.query(query)

reqd_data = query_job.to_dataframe()

reqd_data.head()
from matplotlib import pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

primary_type = reqd_data[['primary_type']]

primary_type_transformed = ordinal_encoder.fit_transform(primary_type.values)

reqd_data[['primary_type']] = primary_type_transformed
num_primary_types = len(ordinal_encoder.categories_[0])

num_primary_types
reqd_data.head()
reqd_data['primary_type'].hist(bins=num_primary_types)
reqd_data['primary_type'] = reqd_data['primary_type'] / num_primary_types

reqd_data.head()
filtered_indices = reqd_data['latitude'] > 41.0

filtered_data = reqd_data[filtered_indices]
filtered_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=filtered_data['primary_type'], label="Crime Type", figsize=(10, 7),

    c="primary_type", cmap=plt.get_cmap("jet"),

)

plt.title("Crime Map? Huh.")