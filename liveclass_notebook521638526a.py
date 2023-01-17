import numpy as np

import pandas as pd

import sqlite3

import matplotlib.pyplot as plt
database = "./input/amazon-fine-food-reviews/database.sqlite"



# Connecting To The DataBase

Connection = sqlite3.connect(database)



# Getting Data using Sql_Query

filtered_data = pd.read_sql_query("SELECT * FROM Reviews WHERE Score < 3 LIMIT 75000", Connection)

filtered_data = filtered_data.append(pd.read_sql_query("SELECT * FROM Reviews WHERE Score > 3 LIMIT 75000", Connection))





def Sentiment(Score):

    return 0 if Score < 3  else 1





# Changing reviews with score less than 3 to positive and vice versa

actual_score = filtered_data['Score']

filtered_data['Score'] = actual_score.map(Sentiment)
filtered_data.head(3)
# DROP ProductID, UserID, ProfileName 
filtered_data['UserId'].value_counts
filtered_data.groupby('UserId').head()
filtered_data.dtypes
print(f"Shape Of Data: {filtered_data.shape}")
filtered_data_num = filtered_data.select_dtypes(include = ['int64'])

filtered_data_num.head()

filtered_data.isnull().sum().sum()
filtered_data['Score'].value_counts()
# Convert characters into lowercase 

def clean_lowercase(text):

    return str(text).lower()



filtered_data.Summary.map(clean_lowercase)

filtered_data.Text.map(clean_lowercase)
filtered_data[filtered_data['UserId'] == 'AR5J8UI46CURR']
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort')
final = sorted_data.drop_duplicates(subset={'UserId', 'ProfileName', 'Time', 'Text'}, keep='first', inplace=False)
def preprocess_col(col):

    col.map(lambda x: str(x).lower())

    return col

print(preprocess_col(filtered_data['UserId']))