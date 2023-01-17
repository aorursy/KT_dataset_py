import numpy as np 
import pandas as pd 
import seaborn as sbs
import matplotlib.pyplot as plt
df = pd.read_csv('../input/data.csv')
df.head()
df.shape
df.describe
fig = plt.figure(figsize=(25, 10))
p = sbs.countplot(x='Nationality', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.figure(figsize=(40,40))
p = sbs.heatmap(df.corr(), annot=True)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sbs.countplot(x='Preferred Foot', data=df)
p = sbs.countplot(x='Work Rate', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sbs.countplot(x='International Reputation', data=df)
p = sbs.countplot(x='Weak Foot', data=df)
p = sbs.countplot(x='Position', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sbs.countplot(x='Height', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)
top_10 = df.head(10)
print(top_10)
p = sbs.countplot(x='Nationality', data=top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.figure(figsize=(40,40))
p = sbs.heatmap(top_10.corr(), annot=True)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sbs.barplot(x='Name', y='Penalties', data=top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sbs.barplot(x='Name', y='Aggression', data=top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)