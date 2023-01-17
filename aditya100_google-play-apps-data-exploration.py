import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/googleplaystore.csv')
df.head()
df.shape
df = df.dropna() # dropping the null values
df.shape
p = sns.countplot(x='Category', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90) # to rotate the overlapping labels in x-axis
plt.figure(figsize=(25, 10))
p = sns.countplot(x='Genres', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sns.countplot(x='Type', data=df)
p = sns.countplot(x='Content Rating', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sns.countplot(x='Installs', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.figure(figsize=(12, 5))
p = sns.countplot(x='Android Ver', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)