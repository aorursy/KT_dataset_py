# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
# load messages dataset
messages = pd.read_csv("../input/disaster_messages.csv")
messages.head()
# load categories dataset
categories = pd.read_csv("../input/disaster_categories.csv")
categories.head()
categories["categories"][0]
# merge datasets
df = messages.merge(categories, on=["id"])
df.head()
# create a dataframe of the 36 individual category columns
categories = df["categories"].str.split(";", expand=True)
categories.head()
# select the first row of the categories dataframe
row = categories.head(1)

# use this row to extract a list of new column names for categories one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
print(category_colnames)
# rename the columns of categories
categories.columns = category_colnames
categories.head()
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1]
    
    # conversoin - convert column from string to numeric
    categories[column] = categories[column].astype(int)
categories.head()
# drop the original categories column from `data frame - df`
df.drop("categories", axis=1, inplace=True)

df.head()
# concatenate the original dataframe with the new `categories` dataframe df
df = pd.concat([df,categories], axis=1)
df.head()
print("Count of duplicate : {}", format(df[df.duplicated].count()))
# check number of duplicates
df[df.duplicated].shape
#df.duplicated
# drop duplicates
df.drop_duplicates(inplace=True)
# check number of duplicates
df[df.duplicated].count()
#engine = create_engine('sqlite:///db_disaster_messages.db')
#df.to_sql('tbl_disaster_messages', engine, index=False)
# Create database
database = 'database.db'
conn = sqlite3.connect(database)
df.to_sql('table1', conn, index=False)
