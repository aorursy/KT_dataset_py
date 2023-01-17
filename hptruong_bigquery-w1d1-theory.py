# Import relevant modules



# I need help querrying a huge dataset. Please help.

# Ok, call google, choose cloud as your extension, and talk to the guy name bigquery

from google.cloud import bigquery

# Oh wait, I also need to things to look like Excel.

# Ok, all you need to do is to call a guy name pandas. P.s.: he goes by pd xD

import pandas as pd
# Talking to bigquery...

# Hello Mr. Truong, thank you for using our service. How we may help you?

# Hey big...query?

# Yeah

# Cool, I got your name right. I need to query a huge dataset and I was referred to you.

# Congratulations, you're in good hands, Mr. Truong.

# To use our service for FREE, first you have to sign up to be one of our clients.

client = bigquery.Client()

# Thank you for your prompt action!

# Every time you use our sevice, you have to use your username, which is client in this case.

# That's how we know you.
# As a trial, we provide you with the hacker_news dataset.

# Think of a dataset as an Excel spreadsheet: You have a workbook, but each workbook contains multiple sheets, each sheet is a table.

# Now a question for you, do you need to store multiple datasets for your project?

# Yes, I do. Does your service provide a way to organize multiple datasets, each dataset contains multiple tables, bigquery?

# Greate question! I'm glad you brought up the issue because that's what I'm trying to explain.

# To make querying on huge datasets efficient, we really need to be on top of everything.

# So let me introduce you to the next level after client, project.

# A project contains multiple datasets, each dataset contains multiple tables.

# The trial dataset we gave you, i.e., hacker_news, belongs to a project called bigquery-public-data.

# Thanks for the info... You haven't shown me how to gain access to hacker_news...

# Ahhh, the impatience! I understand, but this knowledge will help you a lot later.

# Anyways, to gain access to hacker_news dataset, first you need to create a handle (or reference) to it.

# Provide the name of the dataset and the project

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

print(type(dataset_ref))

# Good job. Next, we'll use get_dataset() on this handle to fetch the dataset

dataset     = client.get_dataset(dataset_ref)

print(type(dataset))

# Oops, did you have an error?

# Yeah, are you sure you debug your code before publishing it, bigquery?

# We did. All you need to do is go to the Add Dataset tab and add hacker_news data.

# Do it and we'll see you in a bit :)
# Congratulations, now you have access to the dataset hacker_news

# Thanks for your help, bigquery. How do I list all the tables in hacker_news?

# Do you have something sort of like, client.list_table()?

# Greate question, why don't you try it out?

client.list_tables(dataset=dataset)
# What return to you is an iterator object, much like when you use zip

# Now, if you do list on it, you will see four ids, each points to a table in the hacker_news dataset

tables = list(client.list_tables(dataset=dataset))

print(tables)
# You can know these tables' names by using table_id method for table objects

for table in tables:

    print(table.table_id)
# Cool! This is very interesting. Thanks bigquery.

# Now, how do I gain access to a table, say, full?

# It's exactly the same as how you fetch a dataset.

# First, create a table handle (or reference).

# Note that since your table is contained in dataset object, you'll create a handle to your table through dataset_ref

table_ref = dataset_ref.table('full')

print(table_ref)

# Then, you use your username to get access to fetch this table

full = client.get_table(table_ref)

print(full)
# Just so you can trust me more, let's look at the schema of the full table

# A schema is a technical term referring to the structure of a table, much like a data dictionary.

full.schema
# In the last SchemaField, we see that 'ranking' is the name of the column, 

# 'INTEGER' is the data (or field) type of 'ranking'

# 'NULLABLE' is the mode of 'ranking'--here, we allow missing values

# 'Comment ranking' is the description of 'ranking'



# I must say that this is very organized.

# How do I see some rows of the table full then?

# Good question. You can do this via the method list_rows. Please use the argument max_results=5 to get five rows for your experiment

# Try it out by yourself, see if you can tame our service :)

list(client.list_rows(full, max_results=5))

# Good job for using list there. By now, I hope you've seen how our service is structured in general

# Every query you do, you have to use your username--here, it's client

# You gain access to your object by creating a handle and get_ method

# What you will get is an iterator, if that object contains multiple elements
# The printout looks dirty, I know. That's why we give you a beautiful command called to_dataframe

client.list_rows(full, max_results=5).to_dataframe()
# Beautiful. Well, on behalf of XYZ, I welcome you to our BigQuery family.

# The next step we recommend you do to familiarize yourself with BigQuery family members is doing the exercise.

# Also, don't forget to brush up on regex and string manipulations in pandas.

# It's going to be very fun :)