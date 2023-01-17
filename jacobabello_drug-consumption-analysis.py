import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../input/drug_consumption_str.data")
len(df)
df.head()
df.info()
print("Number of Unique Countries: ", len(df.country.unique()))
print("Number of Unique Ethnicity: ", len(df.ethnicity.unique()))
print("Number of Unique Age: ", len(df.age.unique()))
print("Number of Unique Education: ", len(df.education.unique()))
countries = df['country'].value_counts().plot(kind='pie', figsize=(8, 8))
ethnicity = df['ethnicity'].value_counts().plot(kind='pie', figsize=(8, 8))
age = df['age'].value_counts().plot(kind='pie', figsize=(8, 8))
gender = df['gender'].value_counts().plot(kind='pie', figsize=(8, 8))
education = df['education'].value_counts().plot(kind='pie', figsize=(8,8))
