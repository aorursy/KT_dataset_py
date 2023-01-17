import pandas as pd

import numpy as np

import csv
# File path shortcut

# Make changes to these as necessary, these aren't fully automated yet. 

# Then everything below this cell should be automated. 

csv_name = '\Detailed_Statistics_Arrivals (10).csv'

dest_name = '\Las Vegas Arrivals'

filename = r'Datasets\Unprocessed'+dest_name+csv_name

filename



filename_cleaned = 'Datasets\Processed'+dest_name+csv_name
# Write to skip first 6 lines

# Problem solved:

# https://www.kite.com/python/answers/how-to-delete-a-line-from-a-file-in-python



# Read unprocessed file 

file = open(filename, 'r+')

lines = file.readlines()

# Close unprocessed file

file.close()



# Remove excess content in CSV

for i in range(6): 

    # print(i, lines[0])

    del lines[0]



# Delete the last line

# print(i, lines[-1])

del lines[-1]



# Write to a new file and dir

new_file = open(filename_cleaned, 'w+')

for line in lines:

    new_file.write(line)



new_file.close()
# Alaska Airlines Destination:(LAX) example, 2019

filename = r'Datasets\Processing'+dest_name+csv_name

my_data = pd.read_csv(filename)
my_data.head()
my_data.tail()
my_data.info()
# Create copy, checkpoint!!

df = my_data.copy()
# Look at uppercase columns

df.columns
# Convert to lowercase columns

df.columns = df.columns.str.lower()

df.columns
df.info()
df_cols = df.columns.str.replace("[()]", "").str.replace(" ", "_").str.replace("date_mm/dd/yyyy", "date")
df_cols.values
df.columns = df_cols
df
# Export to csv

df.to_csv('Datasets/Processed/' + dest_name + csv_name, index=False)