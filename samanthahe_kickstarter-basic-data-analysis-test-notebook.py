import numpy as np
import pandas as pd

#Read CSV file and print basic information about data. 
df = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
print(df.info())
#Viewing only specific columns
print(df[['main_category','state']])
#Print Number of Entries per State
print(df['state'].value_counts())
#Replace projects with a state of canceled with "failed" status
df['state'] = df['state'].replace(['canceled'],'failed')
print(df['state'].value_counts())
#Replace projects with a state of suspended with "failed" status
df['state'] = df['state'].replace(['suspended'],'failed')
print(df['state'].value_counts())
#Remove projects with either "undefined" || "live" state
df = df[(df['state'] != 'undefined')&(df['state'] != 'live')]
print(df['state'].value_counts())
#Total number of projects per category.
category_count = df['main_category'].value_counts();
print(category_count)
#Getting Success Count
success_df = df[df['state']=='successful']
success_count = success_df['main_category'].value_counts()
print (success_count)
#Getting Percentages
success_percent = success_count/category_count*100
print(success_percent)
#Sort Values (ASC)
print("Percentage of Successful Project For Each Main Category - ASCENDING")
print(success_percent.sort_values())

print()
#Sort Values (DESC)
print("Percentage of Successful Project For Each Main Category - DESCENDING")
print(success_percent.sort_values(ascending=False))
