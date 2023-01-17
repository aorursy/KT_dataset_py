import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
col_name = ["id", "title", "organizer", "type", "rating", "difficulty", "students"]
df = pd.read_csv("../input/coursera-course-dataset/coursea_data.csv", header=0, names=col_name, index_col="id").sort_values("id")
df.head()
# Split df['student'] into number and value
s = pd.DataFrame()
s['number'] = pd.to_numeric(df['students'].str[:-1])
s['value'] = df['students'].str[-1]
s.head()
# Which type of values do we have?
s['value'].value_counts()
# Ok, simply convert 'k' to thousand and 'm' to million
s.loc[s['value']=='k', 'value'] = 1000 
s.loc[s['value']=='m', 'value'] = 1000000
s.head()
# Multiply number and value, convert to integer and assign it back to df['students']
df['students'] = pd.to_numeric(s['number']*s['value'], downcast='integer')
df['students'].head()
df.info()
df.describe()
mask = df.organizer.value_counts() >= 10
top_organizers = df.organizer.value_counts()[mask]
top_organizers
top_organizers.plot(kind='barh', figsize=(14,6), title="Top Organizers")
particular_organizer = "Google Cloud"
mask = df["organizer"] == particular_organizer
df[mask].sort_values(by='rating', ascending=False)
cert_types = df.type.value_counts()
cert_types
mask= df["type"] == "PROFESSIONAL CERTIFICATE"
df[mask].sort_values(by='students', ascending=False)
df.difficulty.value_counts()
mask = df["difficulty"] == "Advanced"
df[mask].sort_values(by='title', ascending=True)
# 5 star
mask = df['rating']==5.0
df[mask]
mask = df['students']>=500000
df[mask].sort_values(by='students', ascending=False)
keyword = 'Data Science'
mask = df['title'].str.find(keyword) != -1
df[mask]
