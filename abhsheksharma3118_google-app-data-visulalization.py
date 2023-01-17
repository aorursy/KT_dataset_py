import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
df = pd.read_csv("../input/googleplaystore.csv")
df.head()
df.shape
df.info()
Rating = df.Rating.value_counts()

df.Rating.value_counts()
Category = df.Category.value_counts()

df.Category.value_counts()
Genres = df.Genres.value_counts()
df.Genres.value_counts()
ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x=Category)
plt.xticks(rotation=90)
plt.show()
ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(x=Genres)
plt.xticks(rotation=90)
plt.show()
#visualization_settings()
ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(df['Category'])
sns.despine()

# Label customizing
plt.ylabel('Count', fontsize=16)
plt.xlabel('Category', fontsize=16)
plt.title("Category Frequency", fontsize=16);

category_count  = dataset['Category'].value_counts();
plt.figure(figsize=(20,8))
sns.barplot(category_count.index, category_count.values, alpha=0.8)
plt.ylabel('Count', fontsize=16)
plt.title("App Categories", fontsize=16);
plt.xticks(rotation=90);
category_count  = df['Genres'].value_counts();
plt.figure(figsize=(20,8))
sns.barplot(category_count.index, category_count.values, alpha=0.8)
plt.ylabel('Count', fontsize=16)
plt.title("App Categories", fontsize=16);
plt.xticks(rotation=90);
category_count  = df['Rating'].value_counts();
plt.figure(figsize=(20,8))
sns.barplot(category_count.index, category_count.values, alpha=0.8)
plt.ylabel('Count', fontsize=16)
plt.title("App Categories", fontsize=16);
plt.xticks(rotation=90);
