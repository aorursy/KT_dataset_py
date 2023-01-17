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
# List all the tables in the "chicago_crime" dataset
list_of_tables = list(client.list_tables(dataset))

# Print number of tables in the dataset
print(len(list_of_tables))

num_tables =  1 # Store the answer as num_tables and then run this cell

# Check your answer
q_1.check()
#q_1.hint()
#q_1.solution()
# Write the code to figure out the answer
# Construct a reference to the "crime" table
crime_ref = dataset_ref.table("crime")

# API request - fetch the table
crime_table = client.get_table(crime_ref)

# Use the .schema method to retrieve the table data
print(crime_table.schema)

""" Okay I've done some editing in VSC to get this table.
[SchemaField('unique_key', 'INTEGER', 'REQUIRED', 'Unique identifier for the record.', ()), 
SchemaField('case_number', 'STRING', 'NULLABLE', 'The Chicago Police Department RD Number (Records Division Number), which is unique to the incident.', ()), 
SchemaField('date', 'TIMESTAMP', 'NULLABLE', 'Date when the incident occurred. this is sometimes a best estimate.', ()), 
SchemaField('block', 'STRING', 'NULLABLE', 'The partially redacted address where the incident occurred, placing it on the same block as the actual address.', ()), 
SchemaField('iucr', 'STRING', 'NULLABLE', 'The Illinois Unifrom Crime Reporting code. This is directly linked to the Primary Type and Description. See the list of IUCR codes at https://data.cityofchicago.org/d/c7ck-438e.', ()), 
SchemaField('primary_type', 'STRING', 'NULLABLE', 'The primary description of the IUCR code.', ()), 
SchemaField('description', 'STRING', 'NULLABLE', 'The secondary description of the IUCR code, a subcategory of the primary description.', ()), 
SchemaField('location_description', 'STRING', 'NULLABLE', 'Description of the location where the incident occurred.', ()), 
SchemaField('arrest', 'BOOLEAN', 'NULLABLE', 'Indicates whether an arrest was made.', ()), 
SchemaField('domestic', 'BOOLEAN', 'NULLABLE', 'Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act.', ()), 
SchemaField('beat', 'INTEGER', 'NULLABLE', 'Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. Three to five beats make up a police sector, and three sectors make up a police district. The Chicago Police Department has 22 police districts. See the beats at https://data.cityofchicago.org/d/aerh-rz74.', ()), 
SchemaField('district', 'INTEGER', 'NULLABLE', 'Indicates the police district where the incident occurred. See the districts at https://data.cityofchicago.org/d/fthy-xz3r.', ()), 
SchemaField('ward', 'INTEGER', 'NULLABLE', 'The ward (City Council district) where the incident occurred. See the wards at https://data.cityofchicago.org/d/sp34-6z76.', ()), 
SchemaField('community_area', 'INTEGER', 'NULLABLE', 'Indicates the community area where the incident occurred. Chicago has 77 community areas. See the community areas at https://data.cityofchicago.org/d/cauq-8yn6.', ()), 
SchemaField('fbi_code', 'STRING', 'NULLABLE', "Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). See the Chicago Police Department listing of these classifications at http://gis.chicagopolice.org/clearmap_crime_sums/crime_types.html.", ()), 
SchemaField('x_coordinate', 'FLOAT', 'NULLABLE', 'The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block.', ()), 
SchemaField('y_coordinate', 'FLOAT', 'NULLABLE', 'The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block.', ()), 
SchemaField('year', 'INTEGER', 'NULLABLE', 'Year the incident occurred.', ()), 
SchemaField('updated_on', 'TIMESTAMP', 'NULLABLE', 'Date and time the record was last updated.', ()), 
SchemaField('latitude', 'FLOAT', 'NULLABLE', 'The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.', ()), 
SchemaField('longitude', 'FLOAT', 'NULLABLE', 'The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.', ()), 
SchemaField('location', 'STRING', 'NULLABLE', 'The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block.', ())]
"""
num_timestamp_fields = 2 # Put your answer here

# Check your answer
q_2.check()
#q_2.hint()
#q_2.solution()
# Write the code here to explore the data so you can find the answer

fields_for_plotting = ['latitude', 'longitude'] # Put your answers here

# Check your answer
q_3.check()
#q_3.hint()
#q_3.solution()
# Scratch space for your code